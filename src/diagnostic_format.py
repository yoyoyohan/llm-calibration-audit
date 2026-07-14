"""
Supplementary diagnostic: constrained vs unconstrained prompting.

NOT part of the main experiment. Run after the main collect.
Uses a few bank questions to show whether forbidding chain-of-thought
hurts accuracy — for Methodology / Limitations discussion only.

Usage:
  python src/diagnostic_format.py
  python src/diagnostic_format.py --models llama3.1 mistral   # local only
  python src/diagnostic_format.py --models claude_haiku       # API arm
  python src/diagnostic_format.py --n 5
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clients import call_model
from paths import ROOT, ensure_dirs, load_experiment
from prompts import build_confidence_prompt


def build_unconstrained_prompt(question: str, choices: list[str]) -> str:
    letters = ["A", "B", "C", "D"]
    block = "\n".join(f"{L}. {c}" for L, c in zip(letters, choices))
    return f"""Answer the following multiple-choice question. Think through it step by step, then give your final answer.

Question: {question}

{block}

Show brief reasoning, then end with exactly one line:
FINAL_ANSWER: [A/B/C/D]
"""


def extract_final_letter(text: str | None) -> str | None:
    if not text:
        return None
    import re

    m = re.search(r"FINAL_ANSWER:\s*([A-Da-d])", text)
    if m:
        return m.group(1).upper()
    # fallback: last standalone A-D
    found = re.findall(r"\b([A-D])\b", text.upper())
    return found[-1] if found else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=6, help="Number of questions from bank")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_experiment()
    bank = pd.read_csv(ROOT / cfg["question_bank_path"])
    # Prefer a mix of difficulties if possible
    sample_parts = []
    for d in ["easy", "medium", "hard"]:
        sub = bank[bank["difficulty"] == d]
        if len(sub):
            sample_parts.append(sub.sample(n=min(2, len(sub)), random_state=7))
    sample = pd.concat(sample_parts).head(args.n)

    models = cfg["models"]
    if args.models:
        models = [m for m in models if m["id"] in args.models]

    rows = []
    print("=" * 70)
    print("DIAGNOSTIC: constrained vs unconstrained (supplementary only)")
    print("=" * 70)

    for _, q in sample.iterrows():
        choices = [q["choice_A"], q["choice_B"], q["choice_C"], q["choice_D"]]
        constrained = build_confidence_prompt(q["question"], choices)
        unconstrained = build_unconstrained_prompt(q["question"], choices)
        print(f"\nQ {q['question_id']} ({q['difficulty']}) correct={q['answer_letter']}")
        print(q["question"][:100], "...")

        for model in models:
            for label, prompt in [("constrained", constrained), ("unconstrained", unconstrained)]:
                err = None
                raw = None
                try:
                    raw = call_model(model["backend"], model["model_name"], prompt)
                except Exception as e:  # noqa: BLE001
                    err = str(e)

                if label == "constrained":
                    from parse import parse_response

                    pred, conf, ok = parse_response(raw)
                else:
                    pred = extract_final_letter(raw)
                    conf = None
                    ok = pred is not None

                correct = q["answer_letter"]
                is_correct = pred == correct if pred else False
                rows.append(
                    {
                        "question_id": q["question_id"],
                        "difficulty": q["difficulty"],
                        "model_id": model["id"],
                        "format": label,
                        "predicted": pred,
                        "correct": correct,
                        "is_correct": is_correct,
                        "confidence": conf,
                        "parse_ok": ok,
                        "error": err,
                        "raw_response": (raw or "")[:800],
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                mark = "✓" if is_correct else "✗"
                print(f"  {model['id']:12s} {label:14s} {pred} {mark}  {(raw or err or '')[:80]!r}")
                if model["backend"] == "gemini":
                    time.sleep(float(cfg.get("gemini_sleep_seconds", 4)))
                else:
                    time.sleep(0.2)

    out = ROOT / "data" / "processed" / "diagnostic_format_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("Accuracy by model × format:")
    print(df.groupby(["model_id", "format"])["is_correct"].mean().unstack().round(3))
    print(f"\nSaved → {out}")
    print(
        """
HOW TO USE THIS:
- If unconstrained >> constrained: document that format suppresses CoT;
  constrained = deliberate deployment-like elicitation (keep main results).
- Do NOT replace the main 1080-row experiment with unconstrained data.
- Cite this CSV in Methods as a supplementary methodological check.
"""
    )


if __name__ == "__main__":
    main()
