# Revision Patch (apply to PDF manuscript)

Generated from `data/raw/final_results.csv` after the external review.
New quantitative files:
- `data/processed/response_level_difficulty_regression.csv`
- (recomputed below in this note)

---

## A. Replace Table 6 (and §4.5 regression text)

**Old:** cell-level OLS on 3 difficulty means (invalid df).

**New Table 6. Response-level OLS of per-response overconfidence gap on difficulty (easy=1, medium=2, hard=3).**

| Model | N | Slope | SE | 95% CI | *p* | *R*² |
|---|---:|---:|---:|---|---:|---:|
| Claude Haiku | 267 | 0.028 | 0.020 | [−0.011, 0.068] | 0.158 | 0.008 |
| Llama 3.1 | 360 | 0.121 | 0.030 | [0.062, 0.181] | **6.7×10⁻⁵** | 0.044 |
| Mistral | 359 | 0.095 | 0.032 | [0.032, 0.158] | **0.003** | 0.024 |

Spearman *ρ* (gap vs difficulty): Claude −0.035 (*p*=0.57); Llama 0.170 (*p*=0.001); Mistral 0.208 (*p*<0.001).

**Drop-in Results paragraph:**

> To test whether overconfidence increases with difficulty using adequate sample size, we regressed the **per-response** overconfidence gap (normalized confidence − correctness) on ordinal difficulty coded 1–3 (Table 6). Unlike the three-point cell-mean regression, this analysis uses all parseable responses. Difficulty significantly predicted overconfidence for Llama 3.1 (slope = 0.121, *p* = 6.7×10⁻⁵, *n* = 360) and Mistral (slope = 0.095, *p* = 0.003, *n* = 359). For Claude Haiku the slope was positive but not significant (0.028, *p* = 0.158, *n* = 267). Effect sizes are small (*R*² = 0.02–0.04 for open models), indicating a reliable but modest association once response-level variance is retained.

---

## B. Add confidence SD / Mistral “confidence collapse”

**New Table (optional Table 7) or Results paragraph:**

| Model | Mean conf. | SD | % ≥ 95 |
|---|---:|---:|---:|
| Claude Haiku | 90.1 | **10.8** | 47.2% |
| Llama 3.1 | 86.7 | **5.4** | 7.5% |
| Mistral | 97.9 | **3.0** | **97.2%** |

By difficulty, Mistral SD stays 2.4–3.7; ≥95% of scores are ≥95 on every tier.

**Drop-in:**

> Mean confidence for Mistral remained near ceiling across easy, medium, and hard items (0.978 / 0.970 / 0.988) with very low dispersion (overall SD = 3.0; 97.2% of scores ≥ 95). This pattern is better described as **confidence collapse**—minimal modulation of stated confidence by item content—than as proportional under-adjustment of an otherwise informative confidence signal. Llama 3.1 showed modest dispersion (SD = 5.4); Claude Haiku showed the widest range (SD = 10.8; scores from 15–99).

---

## C. Mistral non-monotonic accuracy (Results + Limitations)

**Drop-in Results:**

> Mistral accuracy was not monotonic in the predetermined tiers (easy 0.583, medium 0.625, hard 0.403). Medium exceeded easy, indicating that subject-level difficulty labels—fixed prior to evaluation—do not perfectly track this model’s empirical error rates. We therefore treat difficulty primarily as a pre-fixed stratification factor rather than a validated per-model hardness ranking (see Limitations).

---

## D. Claude parse missingness by difficulty (Results + Limitations)

Parse OK: easy 99/120 (82.5%), medium 98/120 (81.7%), hard 70/120 (58.3%).  
χ²(2) = 23.57, *p* = 7.6×10⁻⁶.

**Drop-in:**

> Claude Haiku parse failures were significantly concentrated on hard items (58.3% parse OK vs. ~82% on easy/medium; χ²(2) = 23.57, *p* < 0.001). Because hard-tier analyses condition on successfully parsed outputs, surviving hard responses may under-represent verbose/truncated cases and could bias Claude’s hard-tier calibration estimates toward the better-formatted subset. We report hard-tier Claude metrics with this non-random missingness caveat.

---

## E. Soften open-weight vs proprietary (Abstract / Discussion / Conclusion)

**Replace phrases like** “weaker open-weight models” / “proprietary vs open-weight” **with:**

> Particularly for the **weaker-performing models in this study** (Llama 3.1 and Mistral). Because model capability and access type (API vs local open-weight) are **confounded**—Claude Haiku is both the only API model and the highest-accuracy model (92% vs 64% vs 54%)—we **cannot attribute** calibration differences to open-weight architecture specifically rather than overall capability.

---

## F. “Pre-registered” → “predetermined / fixed prior to evaluation”

Global find-replace:
- “pre-registered” → “predetermined” or “fixed prior to model evaluation”
- Keep “frozen” for the question bank (accurate).

Optional credibility boost: cite the Git commit that introduced `config/difficulty_map.yaml` before collect runs.

---

## G. Bootstrap 95% CIs (optional Table 3 footnote)

| Model | Accuracy [95% CI] | ECE [95% CI] |
|---|---|---|
| Claude Haiku | 0.921 [0.891, 0.951] | 0.048 [0.029, 0.081] |
| Llama 3.1 | 0.636 [0.586, 0.683] | 0.231 [0.182, 0.281] |
| Mistral | 0.538 [0.485, 0.588] | 0.446 [0.396, 0.498] |

(1000 nonparametric bootstrap resamples.)

---

## H. Citation / bibliography cleanup

1. **Xiong:** use **2023** everywhere *or* “Xiong et al. (2023/2024)” once; arXiv is 2023, ICLR 2024. Recommend: **(2023)** to match arXiv:2306.13063, and note ICLR 2024 in the reference entry.
2. Delete orphan BibTeX keys between references (`farquhar2024semantic`, `guo2017calibration`, etc.).
3. Fix Yin author list to: Yin, Z., Sun, Q., Guo, Q., Wu, J., Qiu, X., & Huang, X. (2023).
4. Fill header placeholders (corresponding author email; leave Revised/Accepted blank or remove).

---

## I. Discussion tone adjustments (short)

1. Lead with response-level evidence: difficulty→overconfidence is **significant for Llama and Mistral**, not Claude.
2. Describe Mistral as confidence collapse + rising gap driven by falling accuracy, not “confidence rising with difficulty.”
3. Keep hard–easy comparison **directional only**; add capability confound caveat.
4. Explicitly discuss Claude hard-tier parse bias as a threat to the “Claude stays calibrated on hard” claim.

---

## J. Trial consistency (optional one sentence)

Same-question trial–trial confidence correlation: Claude *r*=0.97, Mistral *r*=0.89, Llama *r*=0.61; mean |Δ| ≈ 0.6 / 0.3 / 2.3 points. Confidence is highly stable across the two trials at temperature 0.3.
