"""
Analyze calibration results and write figures + summary tables.

Usage:
  python src/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT, ensure_dirs, load_experiment

DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DIFFICULTY_NUMERIC = {"easy": 1, "medium": 2, "hard": 3}


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    """confidence in [0,1], correct as 0/1."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence >= lo) & (confidence < hi)
        if in_bin.sum() == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidence[in_bin].mean()
        weight = in_bin.mean()
        ece += weight * abs(acc - conf)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "bin_center": (lo + hi) / 2,
                "accuracy": acc,
                "confidence": conf,
                "count": int(in_bin.sum()),
            }
        )
    return float(ece), pd.DataFrame(rows)


def load_clean_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["parse_ok"] = df["parse_ok"].astype(str).str.lower().isin(["1", "true", "yes"])
    df["is_correct"] = df["is_correct"].astype(str).str.lower().isin(["1", "true", "yes"])
    return df


def main() -> None:
    ensure_dirs()
    cfg = load_experiment()
    raw_path = ROOT / cfg["raw_results_path"]
    processed_dir = ROOT / cfg["processed_dir"]
    figures_dir = ROOT / cfg["figures_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run collect.py first.")

    raw = load_clean_results(raw_path)
    parse_rate = raw["parse_ok"].mean() if len(raw) else 0.0
    df = raw[raw["parse_ok"] & raw["confidence"].notna()].copy()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["confidence"])
    df["confidence_norm"] = df["confidence"] / 100.0
    df["difficulty_numeric"] = df["difficulty"].map(DIFFICULTY_NUMERIC)

    if df.empty:
        raise RuntimeError("No parseable rows. Check collect output / prompt format.")

    # ---- Summary by model × difficulty ----
    group_cols = ["model_id", "difficulty"]
    summary = (
        df.groupby(group_cols)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence_norm", "mean"),
        )
        .reset_index()
    )
    summary["overconfidence_gap"] = summary["mean_confidence"] - summary["accuracy"]

    ece_rows = []
    for (model_id, difficulty), sub in df.groupby(group_cols):
        ece, _ = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        ece_rows.append({"model_id": model_id, "difficulty": difficulty, "ece": ece})
    ece_df = pd.DataFrame(ece_rows)
    summary = summary.merge(ece_df, on=group_cols, how="left")
    summary["difficulty"] = pd.Categorical(summary["difficulty"], DIFFICULTY_ORDER, ordered=True)
    summary = summary.sort_values(["model_id", "difficulty"])

    # Domain ECE
    domain_ece_rows = []
    for (model_id, domain), sub in df.groupby(["model_id", "domain"]):
        ece, _ = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        domain_ece_rows.append({"model_id": model_id, "domain": domain, "ece": ece, "n": len(sub)})
    domain_ece = pd.DataFrame(domain_ece_rows)

    # DK / hard-easy slope: overconfidence ~ difficulty_numeric, per model
    dk_rows = []
    for model_id, sub in summary.groupby("model_id"):
        sub = sub.dropna(subset=["overconfidence_gap"])
        if len(sub) < 2:
            continue
        x = sub["difficulty"].map(DIFFICULTY_NUMERIC).astype(float)
        y = sub["overconfidence_gap"].astype(float)
        X = sm.add_constant(x)
        try:
            model = sm.OLS(y, X).fit()
            slope = float(model.params.iloc[1])
            p_value = float(model.pvalues.iloc[1])
            r2 = float(model.rsquared)
        except Exception:  # noqa: BLE001
            slope, p_value, r2 = np.nan, np.nan, np.nan
        corr, corr_p = stats.pearsonr(x, y) if len(sub) >= 3 else (np.nan, np.nan)
        dk_rows.append(
            {
                "model_id": model_id,
                "dk_slope": slope,
                "dk_p_value": p_value,
                "dk_r_squared": r2,
                "hard_easy_corr": corr,
                "hard_easy_corr_p": corr_p,
            }
        )
    dk_df = pd.DataFrame(dk_rows)

    # Overall ECE by model
    overall_rows = []
    for model_id, sub in df.groupby("model_id"):
        ece, bins = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        overall_rows.append(
            {
                "model_id": model_id,
                "n": len(sub),
                "accuracy": sub["is_correct"].mean(),
                "mean_confidence": sub["confidence_norm"].mean(),
                "overconfidence_gap": sub["confidence_norm"].mean() - sub["is_correct"].mean(),
                "ece": ece,
            }
        )
        bins.to_csv(processed_dir / f"reliability_bins_{model_id}.csv", index=False)
    overall = pd.DataFrame(overall_rows)

    # Save tables
    summary.to_csv(processed_dir / "summary_by_model_difficulty.csv", index=False)
    domain_ece.to_csv(processed_dir / "ece_by_model_domain.csv", index=False)
    dk_df.to_csv(processed_dir / "dk_hard_easy_slopes.csv", index=False)
    overall.to_csv(processed_dir / "overall_by_model.csv", index=False)

    meta = {
        "n_raw_rows": int(len(raw)),
        "n_analyzed_rows": int(len(df)),
        "parse_ok_rate": float(parse_rate),
        "temperature": cfg["temperature"],
        "trials_per_question": cfg["trials_per_question"],
    }
    (processed_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- Figures ----
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11})

    # Figure 1: reliability diagrams
    model_ids = list(df["model_id"].unique())
    n = len(model_ids)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.2), squeeze=False)
    for ax, model_id in zip(axes[0], model_ids):
        sub = df[df["model_id"] == model_id]
        ece, bins = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        if len(bins):
            ax.plot(bins["confidence"], bins["accuracy"], "o-", label=model_id)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"{model_id}\nECE={ece:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams (verbal confidence)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_reliability_diagrams.png")
    plt.close(fig)

    # Figure 2: overconfidence vs difficulty
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_id, sub in summary.groupby("model_id"):
        sub = sub.sort_values("difficulty")
        xs = [DIFFICULTY_ORDER.index(d) for d in sub["difficulty"]]
        ax.plot(xs, sub["overconfidence_gap"], "o-", linewidth=2, markersize=8, label=model_id)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(len(DIFFICULTY_ORDER)))
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTY_ORDER])
    ax.set_ylabel("Overconfidence gap (confidence − accuracy)")
    ax.set_xlabel("Difficulty (pre-registered subject tiers)")
    ax.set_title("Does overconfidence rise with difficulty?", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_overconfidence_by_difficulty.png")
    plt.close(fig)

    # Figure 3: domain ECE heatmap
    if not domain_ece.empty:
        pivot = domain_ece.pivot(index="model_id", columns="domain", values="ece")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax)
        ax.set_title("ECE by model × domain (lower is better)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figures_dir / "figure3_ece_domain_heatmap.png")
        plt.close(fig)

    print("=== Analysis complete ===")
    print(json.dumps(meta, indent=2))
    print("\nOverall by model:")
    print(overall.to_string(index=False))
    print("\nDK / hard-easy slopes:")
    print(dk_df.to_string(index=False) if len(dk_df) else "(insufficient difficulty levels)")
    print(f"\nTables → {processed_dir}")
    print(f"Figures → {figures_dir}")


if __name__ == "__main__":
    main()
