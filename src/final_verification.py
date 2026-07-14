"""
Final numbers dump for the Research Brief Results section.

Usage (after collect + analyze):
  python src/final_verification.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, load_experiment


def main() -> None:
    cfg = load_experiment()
    raw_path = ROOT / cfg["raw_results_path"]
    processed = ROOT / cfg["processed_dir"]

    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}. Run collect.py first.")

    df = pd.read_csv(raw_path)
    primary = cfg.get("primary_models") or [m["id"] for m in cfg["models"]]
    parse_ok = df["parse_ok"].astype(str).str.lower().isin(["1", "true", "yes"])
    df["parse_ok"] = parse_ok
    df["is_correct"] = df["is_correct"].astype(str).str.lower().isin(["1", "true", "yes"])
    df = df[df["model_id"].isin(primary)].copy()

    print("=== FINAL DATA VERIFICATION (primary models only) ===")
    print(f"Primary models: {primary}")
    print(f"Total rows: {len(df)}")
    print(f"Parse OK rate: {df['parse_ok'].mean():.1%}")
    print("\nRows by model:")
    print(df.groupby("model_id").size())
    print("\nDifficulty distribution:")
    print(df.groupby("difficulty").size())

    clean = df[df["parse_ok"] & df["confidence"].notna()].copy()
    clean["confidence"] = pd.to_numeric(clean["confidence"], errors="coerce")
    clean = clean.dropna(subset=["confidence"])

    print("\nAccuracy by model × difficulty:")
    print(clean.groupby(["model_id", "difficulty"])["is_correct"].mean().unstack().round(3))

    print("\nMean confidence by model × difficulty:")
    print(clean.groupby(["model_id", "difficulty"])["confidence"].mean().unstack().round(1))

    print("\nOverconfidence gap (conf/100 - accuracy) by model × difficulty:")
    rows = []
    for (model, diff), g in clean.groupby(["model_id", "difficulty"]):
        gap = g["confidence"].mean() / 100 - g["is_correct"].mean()
        rows.append({"model_id": model, "difficulty": diff, "overconfidence_gap": gap})
    print(pd.DataFrame(rows).pivot(index="model_id", columns="difficulty", values="overconfidence_gap").round(3))

    print("\nDK-style check (hard gap > easy gap?):")
    for model in clean["model_id"].unique():
        m = clean[clean["model_id"] == model]
        easy = m[m["difficulty"] == "easy"]
        hard = m[m["difficulty"] == "hard"]
        if easy.empty or hard.empty:
            print(f"  {model}: incomplete difficulty coverage")
            continue
        easy_g = easy["confidence"].mean() / 100 - easy["is_correct"].mean()
        hard_g = hard["confidence"].mean() / 100 - hard["is_correct"].mean()
        print(f"  {model}: {'YES' if hard_g > easy_g else 'NO'} (easy={easy_g:.3f}, hard={hard_g:.3f})")

    for name in [
        "overall_by_model.csv",
        "dk_hard_easy_slopes.csv",
        "summary_by_model_difficulty.csv",
    ]:
        p = processed / name
        if p.exists():
            print(f"\n--- {name} ---")
            print(pd.read_csv(p).to_string(index=False))

    print("\n=== VERIFICATION COMPLETE — send this output to Akul ===")


if __name__ == "__main__":
    main()
