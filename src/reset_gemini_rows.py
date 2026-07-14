"""
Drop Gemini rows so you can recollect only Gemini after quota resets.

Usage:
  python src/reset_gemini_rows.py
  python src/collect.py --models gemini_flash --resume
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import ROOT, load_experiment


def main() -> None:
    cfg = load_experiment()
    path = ROOT / cfg["raw_results_path"]
    df = pd.read_csv(path)
    before = len(df)
    kept = df[df["model_id"] != "gemini_flash"].copy()
    kept.to_csv(path, index=False)
    print(f"Removed {before - len(kept)} gemini_flash rows; kept {len(kept)} rows in {path}")
    print("Next: python src/collect.py --models gemini_flash --resume")


if __name__ == "__main__":
    main()
