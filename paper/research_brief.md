# Confidence Without Competence? Difficulty-Stratified Calibration of Large Language Models

**Authors:** Akul [Last], Yohan Deshpande  
**Track:** AI, Data Science & Computing  
**Event:** NSRI Summer Research Hackathon 2026  

---

## Abstract (≤250 words)

Large language models are increasingly used as decision-support tools, yet users often treat fluent answers as reliable. While LLM calibration has been studied, fewer student-accessible audits jointly (i) elicit verbal confidence, (ii) stratify items by predetermined difficulty, (iii) compare multiple contemporary API models under a locked protocol, and (iv) relate patterns to classic human hard–easy findings. We evaluate Claude Haiku (Anthropic), GPT-4o-mini (OpenAI), and Gemini 3.1 Flash-Lite (Google) on an expanded frozen MMLU bank (540 items = original 180 + 360-item supplement; 3 domains × 3 difficulties; 5 trials; temperature 0.3; max 1024 output tokens). Models must answer A–D and report confidence 0–100. With parse-repair retries, parse success was 100% (8,100/8,100). Claude was near-calibrated and slightly underconfident (accuracy 0.895; ECE 0.033; gap −0.017). GPT-4o-mini and Gemini were overconfident (gaps +0.156 and +0.116; ECE 0.157 and 0.117), with overconfidence rising from easy to hard for all three models—most sharply for GPT-4o-mini (gap 0.028 → 0.354). Response-level regressions of gap on difficulty were significant after Holm–Bonferroni correction for all models. Gemini showed confidence collapse (SD 2.5; 95.6% of scores ≥95). For GPT-4o-mini, verbal confidence correlated only weakly with token-level answer probability (*r* = 0.22). Verbal confidence is an imperfect reliability signal—especially on harder items—and human review remains necessary. Limitations include verbal ≠ internal probabilities for Claude/Gemini, MMLU multiple-choice vs open-ended deployment, and no open-weight local models in this submission.

Word count: ~230

---

## Research question

Do large language models’ verbally elicited confidence ratings become more miscalibrated as task difficulty increases, and does the structure of that miscalibration resemble human hard–easy / overconfidence patterns documented in cognitive psychology?

---

## Motivation

Deployed LLM assistants often accompany answers with implicit or explicit confidence. Overconfident errors are especially costly in education, advice, and professional workflows. Classic psychology documents a **hard–easy effect**: people tend to be overconfident on hard items and better calibrated (or underconfident) on easy items (Lichtenstein, Fischhoff & Phillips, 1982). The Dunning–Kruger framing further highlights mismatches between competence and metacognitive insight (Kruger & Dunning, 1999). Separately, ML work studies whether models “know what they know” (Kadavath et al., 2022) and how to elicit confidence (Xiong et al., 2023). We contribute a compact, fully reproducible multi-model audit that connects these threads with difficulty tiers fixed prior to evaluation on MMLU (Hendrycks et al., 2021).

---

## Method

### Question bank
- Source: MMLU test split (`cais/mmlu`).  
- Domains: STEM, Social Science, Humanities (`config/domains.yaml`).  
- Difficulty: subject-level tiers easy/medium/hard fixed prior to evaluation in `config/difficulty_map.yaml` using MMLU difficulty patterns (not chosen after seeing our model scores).  
- Sampling: original 180 items (seed 42; frozen under `data/v1_frozen/`) plus a 360-item content-disjoint supplement (seed 43) → **540 items** (`data/v2/bank_full_v2.json`). One true duplicate pair in the original bank (`q0060`/`q0073`) is retained for v1 comparability and noted in Limitations.

### Models & decoding
- Claude Haiku (`claude-haiku-4-5-20251001`, Anthropic API)  
- GPT-4o-mini (OpenAI API; token logprobs collected)  
- Gemini 3.1 Flash-Lite (Google Gemini API; per-token logprobs not available)  
- Temperature 0.3; **max_output_tokens = 1024**; **5** independent trials per question×model.  
- Up to 3 parse attempts with a format-repair suffix on retries (`src/parse_retry.py`); primary attempt uses the locked v1 prompt unchanged.

### Confidence elicitation prompt
Exact template in `src/prompts.py` / `config/v2/experiment_v2.yaml`. Models must return:

```text
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]
```

### Metrics
- Accuracy  
- Mean confidence (normalized to [0,1])  
- Overconfidence gap = mean confidence − accuracy  
- Expected Calibration Error (ECE), 10 bins  
- Per-model OLS slope of **per-response** overconfidence gap on difficulty (1=easy … 3=hard), with Holm–Bonferroni correction across 6 predeclared tests (slope ≠ 0 and hard−easy gap > 0, per model)  
- Bootstrap 95% CIs (2,000 resamples) for accuracy, gap, and ECE  
- For GPT-4o-mini only: correlation of verbal confidence with `internal_prob_answer` (exp of answer-token logprob)

### Exclusions
Parse failures would be excluded from metric computation; in this v2 run parse success was **100%** for all models, so no rows were dropped.

---

## Evidence / Results

All numbers below are from `data/v2/processed/*_v2.csv` produced by `src/analyze_v2.py` on stamp `20260716T175808Z` (8,100 responses).

### Overall calibration
| Model | N | Accuracy [95% CI] | Mean conf. | Overconf. gap [95% CI] | ECE [95% CI] |
|---|---:|---|---:|---|---|
| Claude Haiku | 2700 | 0.895 [0.883, 0.907] | 0.878 | −0.017 [−0.027, −0.006] | 0.033 [0.023, 0.044] |
| GPT-4o-mini | 2700 | 0.715 [0.697, 0.731] | 0.871 | +0.156 [0.140, 0.173] | 0.157 [0.141, 0.175] |
| Gemini Flash | 2700 | 0.876 [0.863, 0.889] | 0.992 | +0.115 [0.104, 0.129] | 0.117 [0.104, 0.129] |

Parse success: **2700/2700 (100%)** for each model.

### Difficulty pattern
Overconfidence gap rises easy→hard for all three models (Figure 2 in `figures/v2/`):

| Model | Easy gap | Medium gap | Hard gap |
|---|---:|---:|---:|
| Claude Haiku | −0.041 | −0.021 | +0.011 |
| GPT-4o-mini | +0.028 | +0.086 | +0.354 |
| Gemini Flash | +0.041 | +0.117 | +0.189 |

### Response-level difficulty regression (Table 6 replacement)
Per-response OLS of (confidence − correctness) on difficulty coded 1–3:

| Model | N | Slope | SE | 95% CI | *p* (raw) | *p* (Holm) | *R*² |
|---|---:|---:|---:|---|---:|---:|---:|
| Claude Haiku | 2700 | 0.026 | 0.007 | [0.013, 0.039] | 7.8×10⁻⁵ | 7.8×10⁻⁵ | 0.006 |
| GPT-4o-mini | 2700 | 0.163 | 0.010 | [0.143, 0.183] | 1.0×10⁻⁵⁶ | 3.1×10⁻⁵⁶ | 0.089 |
| Gemini Flash | 2700 | 0.074 | 0.008 | [0.059, 0.088] | 3.6×10⁻²² | 7.3×10⁻²² | 0.034 |

Hard−easy gap contrasts were also significant after Holm correction for all three models (bootstrap one-sided *p* = 0 for each).

### Confidence dispersion / collapse
| Model | Mean conf. (0–100) | SD | % ≥ 95 |
|---|---:|---:|---:|
| Claude Haiku | 87.8 | **13.4** | 42.6% |
| GPT-4o-mini | 87.1 | 3.6 | 4.9% |
| Gemini Flash | 99.2 | **2.5** | **95.6%** |

Gemini’s near-ceiling, low-dispersion confidence is better described as **confidence collapse** than as informative metacognitive modulation.

### Domain pattern
ECE by domain (`ece_by_model_domain_v2.csv`): Claude remains low across Humanities/STEM/Social Science (0.047 / 0.012 / 0.057). GPT-4o-mini is worst on STEM (0.247). Gemini is intermediate (0.081–0.141).

### Verbal vs internal confidence (GPT-4o-mini only)
Among 2,700 GPT-4o-mini responses with answer-token logprobs: Pearson *r*(verbal, internal) = **0.222** (*p* ≈ 1.7×10⁻³¹). ECE using verbal confidence = 0.157; ECE using internal answer probability = **0.243**. Claude and Gemini are excluded by design (Anthropic public API has no token logprobs; Gemini returned only aggregate `avgLogprobs`, not per-token answer probabilities).

### Robustness: original-180 vs full-540
| Bank | Model | Acc | Gap | ECE |
|---|---|---:|---:|---:|
| original_180 | Claude | 0.911 | −0.020 | 0.037 |
| full_540 | Claude | 0.895 | −0.017 | 0.033 |
| original_180 | GPT-4o-mini | 0.726 | +0.148 | 0.152 |
| full_540 | GPT-4o-mini | 0.715 | +0.156 | 0.157 |
| original_180 | Gemini | 0.874 | +0.117 | 0.119 |
| full_540 | Gemini | 0.876 | +0.115 | 0.117 |

Direction and ranking of calibration are stable under the bank expansion.

### Capability vs calibration
Across this three-model sample, higher accuracy tracks lower ECE (Claude best on both; GPT-4o-mini lowest accuracy and highest ECE). We **do not** claim a causal proprietary-vs-open distinction: all three arms here are API models, and capability is confounded with provider.

---

## Limitations

1. Verbal confidence ≠ token probabilities for Claude and Gemini; internal-probability analyses are scoped to GPT-4o-mini.  
2. Constrained `ANSWER`/`CONFIDENCE` format suppresses chain-of-thought; claims concern miscalibration under this elicitation protocol, not absolute MMLU leaderboard scores.  
3. MMLU multiple choice ≠ open-ended deployment.  
4. Subject-tier difficulty is a proxy, not item-level IRT; one duplicate item pair (`q0060`/`q0073`) exists in the original 180.  
5. Human hard–easy baselines use different paradigms — compare directionally only.  
6. Fixed temperature (0.3); model versions dated July 2026.  
7. Open-weight local models (Llama 3.1, Mistral) were **not** collected in this submission; earlier v1 results on those models are archival only.  
8. No formal human-subjects baseline in this brief.

---

## Conclusion / Impact

Overconfidence rises with predetermined difficulty for all three API models in this audit, with the steepest response-level slope for GPT-4o-mini and near-ceiling confidence collapse for Gemini Flash. Claude Haiku remains near-calibrated overall. Practical implication: treat verbal self-reports as a weak standalone safety signal—especially on harder items—and prefer abstention, retrieval, or human review when stakes are high. The open pipeline (`src/collect_v2.py` / `src/analyze_v2.py`) makes the audit re-runnable as models update.

---

## References

1. Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR*.  
2. Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.  
3. Kruger, J., & Dunning, D. (1999). Unskilled and unaware of it. *Journal of Personality and Social Psychology*, *77*(6), 1121–1134.  
4. Lichtenstein, S., Fischhoff, B., & Phillips, L. D. (1982). Calibration of probabilities. In Kahneman, Slovic, & Tversky (Eds.), *Judgment under Uncertainty*.  
5. Xiong, M., et al. (2023). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. arXiv:2306.13063 (ICLR 2024).  
6. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.

---

## Appendix: AI Use Transparency Statement

**Tools used**
- Claude Haiku, GPT-4o-mini, Gemini 3.1 Flash-Lite — experimental subjects  
- Cursor / coding assistants — pipeline scaffolding, debugging, and brief drafting from verified tables  
- HuggingFace `datasets` — MMLU access  

**How AI was used**
Helped write pipeline code and polish wording. Did **not** fabricate results, citations, or conclusions. All quantitative claims in this brief were copied from `src/analyze_v2.py` output tables.

**How claims were verified**
Citations checked via Google Scholar / arXiv. Metrics recomputed from `data/v2/processed/final_results_v2.csv` via `src/analyze_v2.py`.

**Responsibility**
Akul and Yohan take full responsibility and can explain every method step and figure.
