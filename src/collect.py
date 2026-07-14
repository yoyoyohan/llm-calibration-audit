"""
Collect model answers + verbal confidence ratings.

Usage:
  # Smoke test (~few minutes)
  python src/collect.py --smoke

  # Full run on frozen question bank
  python src/collect.py

  # Resume / continue writing checkpoints
  python src/collect.py --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clients import call_model
from parse import index_to_letter, letter_to_index, parse_response
from paths import ROOT, ensure_dirs, load_experiment
from prompts import build_confidence_prompt


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_bank(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Question bank not found at {path}. Run: python src/build_question_bank.py"
        )
    return pd.read_csv(path)


def already_done_keys(df: pd.DataFrame) -> set[tuple]:
    if df.empty:
        return set()
    return set(zip(df["question_id"], df["model_id"], df["trial"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run 3 questions × selected models × 1 trial")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cells already in results CSV (recommended when adding a model)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="DANGEROUS: ignore existing final_results.csv and replace it",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model ids")
    parser.add_argument("--limit-questions", type=int, default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_experiment()
    bank_path = ROOT / cfg["question_bank_path"]
    out_path = ROOT / cfg["raw_results_path"]
    ckpt_dir = ROOT / cfg["checkpoint_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    bank = load_bank(bank_path)
    if args.limit_questions:
        bank = bank.head(args.limit_questions)
    if args.smoke:
        bank = bank.head(3)

    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["id"] in args.models]
        if not models:
            raise ValueError("No matching models after --models filter")

    trials = 1 if args.smoke else int(cfg["trials_per_question"])
    gemini_sleep = float(cfg.get("gemini_sleep_seconds", 4.0))
    anthropic_sleep = float(cfg.get("anthropic_sleep_seconds", 1.0))
    ollama_sleep = float(cfg.get("ollama_sleep_seconds", 0.1))

    # Always preserve existing results unless --overwrite (prevents wiping Llama/Mistral)
    existing = pd.DataFrame()
    done: set[tuple] = set()
    if out_path.exists() and not args.overwrite:
        existing = pd.read_csv(out_path)
        done = already_done_keys(existing)
        print(f"Found existing results: {len(existing)} rows ({len(done)} cells). Will APPEND new work.")
        print("Skipping cells that already exist (safe merge mode).")
    elif args.overwrite and out_path.exists():
        print("WARNING: --overwrite set; existing final_results.csv will be replaced.")

    rows: list[dict] = []
    total = len(bank) * len(models) * trials
    pbar = tqdm(total=total, desc="collect")

    for _, q in bank.iterrows():
        choices = [q["choice_A"], q["choice_B"], q["choice_C"], q["choice_D"]]
        prompt = build_confidence_prompt(q["question"], choices)
        correct_letter = q["answer_letter"] if "answer_letter" in q else index_to_letter(int(q["answer_index"]))

        for model in models:
            for trial in range(trials):
                key = (q["question_id"], model["id"], trial)
                if key in done:
                    pbar.update(1)
                    continue

                raw = None
                err = None
                try:
                    raw = call_model(model["backend"], model["model_name"], prompt)
                except Exception as e:  # noqa: BLE001
                    err = str(e)

                answer, confidence, parse_ok = parse_response(raw)
                pred_idx = letter_to_index(answer)
                is_correct = bool(pred_idx == int(q["answer_index"])) if pred_idx is not None else False

                rows.append(
                    {
                        "question_id": q["question_id"],
                        "subject": q["subject"],
                        "domain": q["domain"],
                        "difficulty": q["difficulty"],
                        "model_id": model["id"],
                        "backend": model["backend"],
                        "model_name": model["model_name"],
                        "trial": trial,
                        "temperature": cfg["temperature"],
                        "prompt": prompt,
                        "raw_response": raw,
                        "error": err,
                        "predicted_letter": answer,
                        "correct_letter": correct_letter,
                        "is_correct": is_correct,
                        "confidence": confidence,
                        "parse_ok": parse_ok,
                        "timestamp_utc": utc_now(),
                    }
                )

                if model["backend"] == "gemini":
                    time.sleep(gemini_sleep)
                elif model["backend"] == "anthropic":
                    time.sleep(anthropic_sleep)
                else:
                    time.sleep(ollama_sleep)

                pbar.update(1)

        # Checkpoint after each question
        if rows:
            partial = pd.DataFrame(rows)
            if not existing.empty:
                combined = pd.concat([existing, partial], ignore_index=True)
            else:
                combined = partial
            combined.to_csv(out_path, index=False)
            combined.to_csv(ckpt_dir / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    pbar.close()

    final = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame(rows)
    print(f"\nWrote {len(final)} rows → {out_path}")
    if len(final):
        print("Parse OK rate:", final["parse_ok"].mean())
        print("Accuracy by model:")
        print(final.groupby("model_id")["is_correct"].mean())


if __name__ == "__main__":
    main()
