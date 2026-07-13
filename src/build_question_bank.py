"""
Build a frozen, difficulty-stratified MMLU question bank.

Usage (from repo root, with venv active):
  python src/build_question_bank.py
  python src/build_question_bank.py --per-cell 5   # smaller smoke bank
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# Allow `python src/build_question_bank.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT, ensure_dirs, load_experiment, load_yaml


def subject_to_domain(subject: str, domains: dict) -> str | None:
    for domain, subjects in domains.items():
        if subject in subjects:
            return domain
    return None


def build_difficulty_lookup(diff_cfg: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for tier, subjects in diff_cfg["tiers"].items():
        for s in subjects:
            lookup[s] = tier
    return lookup


def normalize_choices(value) -> list[str]:
    # HuggingFace/pandas may yield list, tuple, or numpy array
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            value = value.tolist()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        # datasets sometimes serializes oddly after to_pandas
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:  # noqa: BLE001
            pass
    raise ValueError(f"Could not parse choices: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stratified MMLU question bank")
    parser.add_argument(
        "--per-cell",
        type=int,
        default=None,
        help="Override questions per domain×difficulty (default from experiment.yaml)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default from experiment.yaml)",
    )
    args = parser.parse_args()

    ensure_dirs()
    cfg = load_experiment()
    domains_cfg = load_yaml(ROOT / "config" / "domains.yaml")["domains"]
    diff_cfg = load_yaml(ROOT / "config" / "difficulty_map.yaml")
    difficulty_lookup = build_difficulty_lookup(diff_cfg)

    per_cell = args.per_cell if args.per_cell is not None else int(cfg["questions_per_cell"])
    seed = int(cfg["random_seed"])
    out_path = ROOT / (args.out or cfg["question_bank_path"])

    print("Loading MMLU from HuggingFace (cais/mmlu)…")
    print("First run downloads data — may take a few minutes.")

    all_subjects = sorted({s for subs in domains_cfg.values() for s in subs})
    frames = []
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        frames.append(ds.to_pandas())
        print("Loaded config='all'")
    except Exception as e:  # noqa: BLE001
        print(f"config='all' unavailable ({e}); loading subjects individually…")
        for subject in all_subjects:
            try:
                ds_s = load_dataset("cais/mmlu", subject, split="test")
                part = ds_s.to_pandas()
                if "subject" not in part.columns:
                    part["subject"] = subject
                frames.append(part)
                print(f"  loaded {subject}: {len(part)} rows")
            except Exception as se:  # noqa: BLE001
                print(f"  skip {subject}: {se}")

    if not frames:
        raise RuntimeError("Could not load any MMLU subjects.")

    df = pd.concat(frames, ignore_index=True)

    # Expected columns: question, subject, choices, answer
    required = {"question", "subject", "choices", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"MMLU missing columns: {missing}. Got: {list(df.columns)}")

    df["domain"] = df["subject"].apply(lambda s: subject_to_domain(s, domains_cfg))
    df["difficulty"] = df["subject"].map(difficulty_lookup)
    df = df.dropna(subset=["domain", "difficulty"]).copy()

    samples = []
    print("\nSampling plan (domain × difficulty):")
    for domain in domains_cfg.keys():
        for difficulty in ["easy", "medium", "hard"]:
            subset = df[(df["domain"] == domain) & (df["difficulty"] == difficulty)]
            n = min(per_cell, len(subset))
            print(f"  {domain:16s} {difficulty:6s} available={len(subset):4d} sample={n}")
            if n == 0:
                print(f"    WARNING: empty cell {domain}/{difficulty}")
                continue
            samples.append(subset.sample(n=n, random_state=seed))

    if not samples:
        raise RuntimeError("No questions sampled. Check domain/difficulty maps.")

    bank = pd.concat(samples, ignore_index=True)

    # Flatten choices into stable columns for CSV friendliness
    choice_lists = bank["choices"].apply(normalize_choices)
    for i, letter in enumerate(["A", "B", "C", "D"]):
        bank[f"choice_{letter}"] = choice_lists.apply(lambda c, idx=i: c[idx])

    bank["answer_index"] = bank["answer"].astype(int)
    bank["answer_letter"] = bank["answer_index"].map({0: "A", 1: "B", 2: "C", 3: "D"})
    bank["question_id"] = [f"q{i:04d}" for i in range(len(bank))]

    keep = [
        "question_id",
        "subject",
        "domain",
        "difficulty",
        "question",
        "choice_A",
        "choice_B",
        "choice_C",
        "choice_D",
        "answer_index",
        "answer_letter",
    ]
    bank = bank[keep]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bank.to_csv(out_path, index=False)

    print(f"\nWrote {len(bank)} questions → {out_path}")
    print(bank.groupby(["domain", "difficulty"]).size().unstack(fill_value=0))
    print("\nFROZEN: do not regenerate with a new seed after data collection starts.")


if __name__ == "__main__":
    main()
