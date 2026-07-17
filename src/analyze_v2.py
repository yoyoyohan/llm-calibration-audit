"""
v2 analysis — Claude Haiku, GPT-4o-mini, Gemini Flash on 540×5 collection.

Usage (from repo root):
  python src/analyze_v2.py
  python src/analyze_v2.py --stamp 20260716T175808Z
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT  # noqa: E402

DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DIFFICULTY_NUMERIC = {"easy": 1, "medium": 2, "hard": 3}
PRIMARY_MODELS = ["claude_haiku", "gpt4o_mini", "gemini_flash"]
DISPLAY = {
    "claude_haiku": "Claude Haiku",
    "gpt4o_mini": "GPT-4o-mini",
    "gemini_flash": "Gemini Flash",
}


def compute_ece(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
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
        acc = float(correct[in_bin].mean())
        conf = float(confidence[in_bin].mean())
        weight = float(in_bin.mean())
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


def holm_bonferroni(p_values: dict[str, float]) -> pd.DataFrame:
    items = sorted(p_values.items(), key=lambda kv: (kv[1] if kv[1] == kv[1] else 1.0))
    m = len(items)
    rows = []
    running_max = 0.0
    for rank, (name, p) in enumerate(items, start=1):
        if p != p:
            adj = np.nan
        else:
            adj = min(1.0, float(p) * (m - rank + 1))
            running_max = max(running_max, adj)
            adj = running_max
        rows.append({"test": name, "p_raw": p, "p_holm": adj, "rank": rank})
    return pd.DataFrame(rows)


def bootstrap_ci(
    values: pd.DataFrame,
    metric_fn: Callable[[pd.DataFrame], float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = float(metric_fn(values))
    n = len(values)
    if n == 0:
        return point, np.nan, np.nan
    samples = np.empty(n_bootstrap)
    idx = np.arange(n)
    for i in range(n_bootstrap):
        draw = values.iloc[rng.choice(idx, size=n, replace=True)]
        samples[i] = float(metric_fn(draw))
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    return point, float(lo), float(hi)


def load_jsonl_dir(raw_dir: Path, stamp: str, models: list[str]) -> pd.DataFrame:
    frames = []
    for model_id in models:
        path = raw_dir / f"{model_id}_{stamp}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df["parse_ok"] = df["parse_ok"].astype(bool)
    df["is_correct"] = df["is_correct"].astype(bool)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    return df


def response_level_regression(sub: pd.DataFrame) -> dict:
    x = sub["difficulty_numeric"].astype(float)
    y = sub["overconfidence_gap_row"].astype(float)
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    slope = float(model.params.iloc[1])
    se = float(model.bse.iloc[1])
    p = float(model.pvalues.iloc[1])
    r2 = float(model.rsquared)
    ci_lo, ci_hi = [float(v) for v in model.conf_int().iloc[1].tolist()]
    rho, rho_p = stats.spearmanr(x, y)
    return {
        "n": int(len(sub)),
        "slope": slope,
        "se": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p,
        "r_squared": r2,
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
    }


def hard_minus_easy_gap(sub: pd.DataFrame) -> float:
    easy = sub[sub["difficulty"] == "easy"]
    hard = sub[sub["difficulty"] == "hard"]
    if easy.empty or hard.empty:
        return np.nan
    g_easy = easy["confidence_norm"].mean() - easy["is_correct"].astype(float).mean()
    g_hard = hard["confidence_norm"].mean() - hard["is_correct"].astype(float).mean()
    return float(g_hard - g_easy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stamp", default=None, help="Collection stamp (default: CURRENT_RUN_STAMP.txt)")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    raw_dir = ROOT / "data" / "v2" / "raw"
    stamp = args.stamp
    if not stamp:
        stamp_path = raw_dir / "CURRENT_RUN_STAMP.txt"
        stamp = stamp_path.read_text(encoding="utf-8").strip() if stamp_path.exists() else "20260716T175808Z"

    processed_dir = ROOT / "data" / "v2" / "processed"
    figures_dir = ROOT / "figures" / "v2"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    raw = load_jsonl_dir(raw_dir, stamp, PRIMARY_MODELS)
    # Drop bulky text from the published CSV; keep a slim analysis table.
    export_cols = [
        c
        for c in raw.columns
        if c
        not in {
            "raw_response",
            "internal_prob_alternatives",
            "error",
        }
    ]
    raw[export_cols].to_csv(processed_dir / "final_results_v2.csv", index=False)

    parse_summary = (
        raw.groupby("model_id", observed=False)
        .agg(n=("parse_ok", "size"), parse_ok_rate=("parse_ok", "mean"))
        .reset_index()
    )
    parse_summary.to_csv(processed_dir / "parse_rates_v2.csv", index=False)

    df = raw[raw["parse_ok"] & raw["confidence"].notna()].copy()
    df["confidence_norm"] = df["confidence"] / 100.0
    df["difficulty_numeric"] = df["difficulty"].map(DIFFICULTY_NUMERIC)
    df["overconfidence_gap_row"] = df["confidence_norm"] - df["is_correct"].astype(float)
    df["model_id"] = pd.Categorical(df["model_id"], categories=PRIMARY_MODELS, ordered=True)

    # Overall metrics + bootstrap CIs
    overall_rows = []
    for model_id, sub in df.groupby("model_id", observed=False):
        def acc_fn(d: pd.DataFrame) -> float:
            return float(d["is_correct"].astype(float).mean())

        def ece_fn(d: pd.DataFrame) -> float:
            ece, _ = compute_ece(d["confidence_norm"].to_numpy(), d["is_correct"].astype(float).to_numpy())
            return ece

        def gap_fn(d: pd.DataFrame) -> float:
            return float(d["confidence_norm"].mean() - d["is_correct"].astype(float).mean())

        acc, acc_lo, acc_hi = bootstrap_ci(sub, acc_fn, n_bootstrap=args.n_bootstrap, seed=42)
        ece, ece_lo, ece_hi = bootstrap_ci(sub, ece_fn, n_bootstrap=args.n_bootstrap, seed=43)
        gap, gap_lo, gap_hi = bootstrap_ci(sub, gap_fn, n_bootstrap=args.n_bootstrap, seed=44)
        overall_rows.append(
            {
                "model_id": model_id,
                "n": int(len(sub)),
                "accuracy": acc,
                "accuracy_ci_lo": acc_lo,
                "accuracy_ci_hi": acc_hi,
                "mean_confidence": float(sub["confidence_norm"].mean()),
                "overconfidence_gap": gap,
                "gap_ci_lo": gap_lo,
                "gap_ci_hi": gap_hi,
                "ece": ece,
                "ece_ci_lo": ece_lo,
                "ece_ci_hi": ece_hi,
                "conf_sd": float(sub["confidence"].std()),
                "pct_conf_ge_95": float((sub["confidence"] >= 95).mean()),
            }
        )
        _, bins = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        bins.to_csv(processed_dir / f"reliability_bins_{model_id}.csv", index=False)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(processed_dir / "overall_by_model_v2.csv", index=False)

    # By difficulty
    summary = (
        df.groupby(["model_id", "difficulty"], observed=False)
        .agg(
            n=("is_correct", "size"),
            accuracy=("is_correct", "mean"),
            mean_confidence=("confidence_norm", "mean"),
        )
        .reset_index()
    )
    summary["overconfidence_gap"] = summary["mean_confidence"] - summary["accuracy"]
    ece_rows = []
    for (model_id, difficulty), sub in df.groupby(["model_id", "difficulty"], observed=False):
        ece, _ = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        ece_rows.append({"model_id": model_id, "difficulty": difficulty, "ece": ece})
    summary = summary.merge(pd.DataFrame(ece_rows), on=["model_id", "difficulty"], how="left")
    summary["difficulty"] = pd.Categorical(summary["difficulty"], DIFFICULTY_ORDER, ordered=True)
    summary = summary.sort_values(["model_id", "difficulty"])
    summary.to_csv(processed_dir / "summary_by_model_difficulty_v2.csv", index=False)

    # Domain ECE
    domain_rows = []
    for (model_id, domain), sub in df.groupby(["model_id", "domain"], observed=False):
        ece, _ = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        domain_rows.append({"model_id": model_id, "domain": domain, "ece": ece, "n": len(sub)})
    domain_ece = pd.DataFrame(domain_rows)
    domain_ece.to_csv(processed_dir / "ece_by_model_domain_v2.csv", index=False)

    # Response-level regression + hard>easy contrast
    reg_rows = []
    p_tests: dict[str, float] = {}
    for model_id, sub in df.groupby("model_id", observed=False):
        reg = response_level_regression(sub)
        reg["model_id"] = model_id
        hard_easy = hard_minus_easy_gap(sub)
        reg["hard_minus_easy_gap"] = hard_easy
        # one-sided bootstrap p for hard_gap > easy_gap
        rng = np.random.default_rng(100 + PRIMARY_MODELS.index(str(model_id)))
        n = len(sub)
        diffs = []
        for _ in range(args.n_bootstrap):
            draw = sub.iloc[rng.choice(n, size=n, replace=True)]
            diffs.append(hard_minus_easy_gap(draw))
        diffs_arr = np.asarray(diffs, dtype=float)
        p_hard = float(np.mean(diffs_arr <= 0.0))  # one-sided
        reg["hard_gt_easy_p_boot"] = p_hard
        reg_rows.append(reg)
        p_tests[f"{model_id}_slope_ne_0"] = reg["p_value"]
        p_tests[f"{model_id}_hard_gt_easy"] = p_hard
    reg_df = pd.DataFrame(reg_rows)
    reg_df.to_csv(processed_dir / "response_level_difficulty_regression_v2.csv", index=False)
    holm_df = holm_bonferroni(p_tests)
    holm_df.to_csv(processed_dir / "holm_corrected_tests_v2.csv", index=False)

    # original_180 vs full_540 robustness
    rob_rows = []
    for bank_filter, label in [("original_180", "original_180"), (None, "full_540")]:
        sub_all = df if bank_filter is None else df[df["bank_source"] == bank_filter]
        for model_id, sub in sub_all.groupby("model_id", observed=False):
            ece, _ = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
            rob_rows.append(
                {
                    "bank": label,
                    "model_id": model_id,
                    "n": int(len(sub)),
                    "accuracy": float(sub["is_correct"].mean()),
                    "mean_confidence": float(sub["confidence_norm"].mean()),
                    "overconfidence_gap": float(sub["confidence_norm"].mean() - sub["is_correct"].mean()),
                    "ece": ece,
                }
            )
    rob = pd.DataFrame(rob_rows)
    rob.to_csv(processed_dir / "original180_vs_full540_v2.csv", index=False)

    # Verbal vs internal (GPT-4o-mini primarily)
    vip_rows = []
    for model_id, sub in df.groupby("model_id", observed=False):
        has_ip = sub["internal_prob_answer"].notna()
        if has_ip.sum() < 30:
            vip_rows.append(
                {
                    "model_id": model_id,
                    "n_with_internal": int(has_ip.sum()),
                    "note": "insufficient / unavailable internal probs",
                }
            )
            continue
        s = sub[has_ip].copy()
        s["internal_prob_answer"] = pd.to_numeric(s["internal_prob_answer"], errors="coerce")
        s = s.dropna(subset=["internal_prob_answer"])
        r, rp = stats.pearsonr(s["confidence_norm"], s["internal_prob_answer"])
        ece_v, _ = compute_ece(s["confidence_norm"].to_numpy(), s["is_correct"].astype(float).to_numpy())
        ece_i, _ = compute_ece(s["internal_prob_answer"].to_numpy(), s["is_correct"].astype(float).to_numpy())
        vip_rows.append(
            {
                "model_id": model_id,
                "n_with_internal": int(len(s)),
                "pearson_r_verbal_internal": float(r),
                "pearson_p": float(rp),
                "ece_verbal": float(ece_v),
                "ece_internal": float(ece_i),
            }
        )
    vip = pd.DataFrame(vip_rows)
    vip.to_csv(processed_dir / "verbal_vs_internal_v2.csv", index=False)

    meta = {
        "stamp": stamp,
        "n_raw": int(len(raw)),
        "n_analyzed": int(len(df)),
        "primary_models": PRIMARY_MODELS,
        "n_bootstrap": args.n_bootstrap,
        "max_output_tokens": 1024,
        "trials_per_question": 5,
        "temperature": 0.3,
        "ollama_models_collected": False,
        "note": "Empirical difficulty via Llama/Mistral omitted — Ollama not collected in this submission.",
    }
    (processed_dir / "run_metadata_v2.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- Figures ----
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11})
    model_ids = PRIMARY_MODELS

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), squeeze=False)
    for ax, model_id in zip(axes[0], model_ids):
        sub = df[df["model_id"] == model_id]
        ece, bins = compute_ece(sub["confidence_norm"].to_numpy(), sub["is_correct"].astype(float).to_numpy())
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        if len(bins):
            ax.plot(bins["confidence"], bins["accuracy"], "o-", label=DISPLAY[model_id])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean confidence")
        ax.set_ylabel("Fraction correct")
        ax.set_title(f"{DISPLAY[model_id]}\nECE={ece:.3f}")
        ax.legend(fontsize=8)
    fig.suptitle("Reliability diagrams (v2 verbal confidence)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_reliability_diagrams.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model_id in model_ids:
        sub = summary[summary["model_id"] == model_id].sort_values("difficulty")
        xs = [DIFFICULTY_ORDER.index(str(d)) for d in sub["difficulty"]]
        ax.plot(xs, sub["overconfidence_gap"], "o-", linewidth=2, markersize=8, label=DISPLAY[model_id])
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Easy", "Medium", "Hard"])
    ax.set_ylabel("Overconfidence gap (confidence − accuracy)")
    ax.set_xlabel("Difficulty (predetermined subject tiers)")
    ax.set_title("Does overconfidence rise with difficulty? (v2)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_overconfidence_by_difficulty.png")
    plt.close(fig)

    if not domain_ece.empty:
        pivot = domain_ece.pivot(index="model_id", columns="domain", values="ece")
        pivot.index = [DISPLAY.get(str(i), str(i)) for i in pivot.index]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=ax)
        ax.set_title("ECE by model × domain (v2; lower is better)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(figures_dir / "figure3_ece_domain_heatmap.png")
        plt.close(fig)

    # Accuracy vs ECE scatter
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, row in overall.iterrows():
        ax.scatter(row["accuracy"], row["ece"], s=120, label=DISPLAY[str(row["model_id"])])
        ax.annotate(DISPLAY[str(row["model_id"])], (row["accuracy"], row["ece"]), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("ECE (lower is better)")
    ax.set_title("Capability vs calibration (v2)", fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_accuracy_vs_ece.png")
    plt.close(fig)

    print("=== v2 analysis complete ===")
    print(json.dumps(meta, indent=2))
    print("\nParse rates:")
    print(parse_summary.to_string(index=False))
    print("\nOverall:")
    print(overall.to_string(index=False))
    print("\nBy difficulty:")
    print(summary.to_string(index=False))
    print("\nResponse-level regression:")
    print(reg_df.to_string(index=False))
    print("\nHolm-corrected tests:")
    print(holm_df.to_string(index=False))
    print("\nOriginal-180 vs full-540:")
    print(rob.to_string(index=False))
    print("\nVerbal vs internal:")
    print(vip.to_string(index=False))
    print(f"\nTables → {processed_dir}")
    print(f"Figures → {figures_dir}")


if __name__ == "__main__":
    main()
