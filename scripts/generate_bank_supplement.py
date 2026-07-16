"""
Generate a non-overlapping MMLU bank supplement for v2.

Reuses config/domains.yaml and config/difficulty_map.yaml exactly.
Excludes items already present in the frozen original 180 by content key
(subject + question text), not by bank-local question_id.

Usage (from repo root):
  python scripts/generate_bank_supplement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_question_bank import (  # noqa: E402
    build_difficulty_lookup,
    normalize_choices,
    subject_to_domain,
)
from paths import load_yaml  # noqa: E402

ORIGINAL_CSV = ROOT / "data" / "v1_frozen" / "question_bank_original_180.csv"
ORIGINAL_JSON = ROOT / "data" / "v1_frozen" / "bank_original_180.json"
OUT_SUPPLEMENT = ROOT / "data" / "v2" / "bank_supplement_360.json"
OUT_FULL = ROOT / "data" / "v2" / "bank_full_v2.json"
SEED = 43
PER_CELL = 40


def content_key(subject: str, question: str) -> str:
    return f"{subject}||{question.strip()}"


def load_original() -> pd.DataFrame:
    if ORIGINAL_CSV.exists():
        return pd.read_csv(ORIGINAL_CSV)
    return pd.DataFrame(json.loads(ORIGINAL_JSON.read_text(encoding="utf-8")))


def load_mmlu_pool(domains_cfg: dict) -> pd.DataFrame:
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
    return pd.concat(frames, ignore_index=True)


def flatten_bank(bank: pd.DataFrame) -> pd.DataFrame:
    choice_lists = bank["choices"].apply(normalize_choices)
    for i, letter in enumerate(["A", "B", "C", "D"]):
        bank[f"choice_{letter}"] = choice_lists.apply(lambda c, idx=i: c[idx])
    bank["answer_index"] = bank["answer"].astype(int)
    bank["answer_letter"] = bank["answer_index"].map({0: "A", 1: "B", 2: "C", 3: "D"})
    keep = [
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
    return bank[keep].copy()


def main() -> None:
    domains_cfg = load_yaml(ROOT / "config" / "domains.yaml")["domains"]
    diff_cfg = load_yaml(ROOT / "config" / "difficulty_map.yaml")
    difficulty_lookup = build_difficulty_lookup(diff_cfg)

    original = load_original()
    used_keys = {
        content_key(str(r.subject), str(r.question)) for r in original.itertuples(index=False)
    }
    print(f"Original bank: {len(original)} items; unique content keys: {len(used_keys)}")

    df = load_mmlu_pool(domains_cfg)
    df["domain"] = df["subject"].apply(lambda s: subject_to_domain(s, domains_cfg))
    df["difficulty"] = df["subject"].map(difficulty_lookup)
    df = df.dropna(subset=["domain", "difficulty"]).copy()
    df["_key"] = [
        content_key(str(s), str(q)) for s, q in zip(df["subject"], df["question"])
    ]
    df = df[~df["_key"].isin(used_keys)].copy()
    print(f"Pool after excluding original overlap: {len(df)} rows")

    samples = []
    shortfalls: list[str] = []
    print("\nSupplement sampling plan (seed=43, target=40/cell):")
    for domain in domains_cfg.keys():
        for difficulty in ["easy", "medium", "hard"]:
            subset = df[(df["domain"] == domain) & (df["difficulty"] == difficulty)]
            n = min(PER_CELL, len(subset))
            print(f"  {domain:16s} {difficulty:6s} available={len(subset):4d} sample={n}")
            if n < PER_CELL:
                msg = (
                    f"SHORTFALL: {domain}/{difficulty} — available {len(subset)}, "
                    f"target {PER_CELL}, sampled {n} (short by {PER_CELL - n})"
                )
                print(f"    WARNING: {msg}")
                shortfalls.append(msg)
            if n == 0:
                continue
            samples.append(subset.sample(n=n, random_state=SEED))

    if not samples:
        raise RuntimeError("No supplement questions sampled.")

    supplement = flatten_bank(pd.concat(samples, ignore_index=True))
    # Verify no content overlap with original
    supp_keys = {
        content_key(str(r.subject), str(r.question)) for r in supplement.itertuples(index=False)
    }
    overlap = used_keys & supp_keys
    if overlap:
        raise RuntimeError(f"Overlap detected after sampling: {len(overlap)} keys")

    supplement = supplement.copy()
    supplement["question_id"] = [f"s{i:04d}" for i in range(len(supplement))]
    supplement["bank_source"] = "supplement_v2"

    original_out = original.copy()
    if "bank_source" not in original_out.columns:
        original_out["bank_source"] = "original_180"
    # Ensure question_id preserved from original
    if "question_id" not in original_out.columns:
        original_out["question_id"] = [f"q{i:04d}" for i in range(len(original_out))]

    cols = [
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
        "bank_source",
    ]
    supplement = supplement[cols]
    original_out = original_out[cols]

    OUT_SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUPPLEMENT.write_text(
        json.dumps(supplement.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    full = pd.concat([original_out, supplement], ignore_index=True)
    OUT_FULL.write_text(
        json.dumps(full.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nWrote {len(supplement)} supplement items → {OUT_SUPPLEMENT}")
    print(f"Wrote {len(full)} full-bank items → {OUT_FULL}")
    print(supplement.groupby(["domain", "difficulty"]).size().unstack(fill_value=0))
    if shortfalls:
        print("\nSHORTFALL SUMMARY:")
        for s in shortfalls:
            print(f"  - {s}")
    else:
        print("\nAll cells met target of 40 non-overlapping items.")


if __name__ == "__main__":
    main()
