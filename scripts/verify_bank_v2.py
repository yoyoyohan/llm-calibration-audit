"""
Verify v2 combined question bank integrity.

Usage (from repo root):
  python scripts/verify_bank_v2.py
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "v2" / "bank_full_v2.json"
ORIGINAL = ROOT / "data" / "v1_frozen" / "bank_original_180.json"
SUPPLEMENT = ROOT / "data" / "v2" / "bank_supplement_360.json"


def content_key(item: dict) -> str:
    return f"{item['subject']}||{str(item['question']).strip()}"


def main() -> None:
    full = json.loads(FULL.read_text(encoding="utf-8"))
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    supplement = json.loads(SUPPLEMENT.read_text(encoding="utf-8"))

    print("=" * 72)
    print("bank_full_v2 verification report")
    print("=" * 72)

    print(f"\n(a) Total items in bank_full_v2.json: {len(full)}")
    print(f"    original_180 file count: {len(original)}")
    print(f"    supplement_v2 file count: {len(supplement)}")
    print(f"    expected sum: {len(original) + len(supplement)}")

    ids = [x["question_id"] for x in full]
    id_counts = Counter(ids)
    dup_ids = [i for i, c in id_counts.items() if c > 1]
    print(f"\n(b) Duplicate question_ids in full bank: {len(dup_ids)}")
    if dup_ids:
        print(f"    EXAMPLES: {dup_ids[:10]}")
    else:
        print("    OK — zero duplicate question_ids")

    orig_keys = {content_key(x) for x in original}
    supp_keys = {content_key(x) for x in supplement}
    overlap = orig_keys & supp_keys
    print(f"\n(c) Content-key overlap (subject||question) original ∩ supplement: {len(overlap)}")
    if overlap:
        print(f"    EXAMPLES: {list(overlap)[:3]}")
    else:
        print("    OK — zero content overlap between original_180 and supplement_v2")

    # Also check bank_source splits inside full
    by_source = Counter(x.get("bank_source") for x in full)
    print(f"\n    bank_source counts in full: {dict(by_source)}")

    print("\n(d) Breakdown: domain × difficulty × bank_source")
    print(f"{'domain':16s} {'diff':7s} {'source':14s} {'n':>5s}")
    cells = Counter((x["domain"], x["difficulty"], x.get("bank_source")) for x in full)
    for domain in sorted({x["domain"] for x in full}):
        for diff in ["easy", "medium", "hard"]:
            for source in ["original_180", "supplement_v2"]:
                n = cells.get((domain, diff, source), 0)
                print(f"{domain:16s} {diff:7s} {source:14s} {n:5d}")

    print("\n(e) Spot-check: 3 randomly selected supplement items (seed=7)")
    rng = random.Random(7)
    picks = rng.sample(supplement, k=min(3, len(supplement)))
    for i, item in enumerate(picks, 1):
        print("-" * 72)
        print(f"SPOT {i}: {item['question_id']} | {item['domain']} / {item['difficulty']} / {item['subject']}")
        print(f"Q: {item['question']}")
        print(f"  A. {item['choice_A']}")
        print(f"  B. {item['choice_B']}")
        print(f"  C. {item['choice_C']}")
        print(f"  D. {item['choice_D']}")
        print(f"Correct: {item['answer_letter']}")

    print("\n" + "=" * 72)
    print("END REPORT")
    print("=" * 72)


if __name__ == "__main__":
    main()
