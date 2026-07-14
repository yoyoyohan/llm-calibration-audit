# Confidence Without Competence? Difficulty-Stratified Calibration of Large Language Models

**Authors:** [Akul Last], [Yohan Last]  
**Track:** AI, Data Science & Computing  
**Event:** NSRI Summer Research Hackathon 2026  

---

## Abstract (≤250 words — write LAST)

[Problem] Large language models are increasingly used as decision-support tools, yet users often treat fluent answers as reliable.  
[Gap] While LLM calibration has been studied, fewer student-accessible audits jointly (i) elicit verbal confidence, (ii) stratify items by pre-registered difficulty, (iii) compare multiple free models, and (iv) relate patterns to classic human hard–easy findings.  
[Method] We evaluate Claude Haiku (Anthropic API), Llama 3.1 8B, and Mistral 7B on a frozen stratified MMLU subset (~180 items; 3 domains × 3 difficulties; 2 trials; temperature 0.3). Models must answer A–D and report confidence 0–100. We compute accuracy, overconfidence gap, ECE, and difficulty–overconfidence slopes.  
[Result] Overconfidence rose from easy to hard for all three models. Claude Haiku was near-calibrated (ECE ≈ 0.05; slight underconfidence overall), while Llama 3.1 and especially Mistral were overconfident (ECE ≈ 0.23 and 0.45), with the largest gaps on hard items.  
[Implication] Verbal confidence is an imperfect reliability signal—particularly for smaller open models on harder items; human review remains necessary.  
[Limitation] Verbal ratings ≠ token probabilities; Claude format-parse rate was ~74% (long answers truncated); MMLU ≠ open-ended deployment.

Word count: ____

---

## Research question

Do large language models’ verbally elicited confidence ratings become more miscalibrated as task difficulty increases, and does the structure of that miscalibration resemble human hard–easy / overconfidence patterns documented in cognitive psychology?

---

## Motivation

Deployed LLM assistants often accompany answers with implicit or explicit confidence. Overconfident errors are especially costly in education, advice, and professional workflows. Classic psychology documents a **hard–easy effect**: people tend to be overconfident on hard items and better calibrated (or underconfident) on easy items (Lichtenstein, Fischhoff & Phillips, 1982). The Dunning–Kruger framing further highlights mismatches between competence and metacognitive insight (Kruger & Dunning, 1999). Separately, ML work studies whether models “know what they know” (Kadavath et al., 2022) and how to elicit confidence (Xiong et al., 2024). We contribute a compact, fully reproducible multi-model audit that connects these threads with pre-registered difficulty tiers on MMLU (Hendrycks et al., 2021).

---

## Method

### Question bank
- Source: MMLU test split (`cais/mmlu`).  
- Domains: STEM, Social Science, Humanities (`config/domains.yaml`).  
- Difficulty: subject-level tiers easy/medium/hard pre-registered in `config/difficulty_map.yaml` using MMLU difficulty patterns (not chosen after seeing our model scores).  
- Sampling: up to 20 questions per domain × difficulty (seed=42) → target ~180 items; report actual N from `data/question_bank.csv`.

### Models & decoding
- Claude Haiku (Anthropic API)  
- Llama 3.1 (Ollama)  
- Mistral (Ollama)  
- Temperature 0.3; 2 independent trials per question×model.

### Confidence elicitation prompt
Exact template in `src/prompts.py` / README. Models must return:

```text
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]
```

### Metrics
- Accuracy  
- Mean confidence (normalized to [0,1])  
- Overconfidence gap = mean confidence − accuracy  
- Expected Calibration Error (ECE), 10 bins  
- Per-model OLS slope of overconfidence gap on difficulty (1=easy … 3=hard)

### Exclusions
Rows with failed parse (missing ANSWER or CONFIDENCE) are excluded from metric computation; parse-failure rate is reported.

---

## Evidence / Results

### Overall calibration
| Model | N | Accuracy | Mean conf. | Overconf. gap | ECE |
|---|---|---|---|---|---|
| claude_haiku | 267 | 0.921 | 0.901 | −0.020 | 0.048 |
| llama3.1 | 360 | 0.636 | 0.867 | +0.231 | 0.231 |
| mistral | 359 | 0.538 | 0.979 | +0.441 | 0.446 |

### Difficulty pattern
Figure 2: overconfidence gap rises easy→hard for all three models. Claude stays near zero (best calibrated). Llama rises ~0.12→0.36. Mistral is high throughout and peaks on hard (~0.59).

### Domain pattern
Figure 3: Claude ECE is low across Humanities/STEM/Social Science (~0.06–0.08). Llama and Mistral are worse, especially STEM (Llama ~0.41; Mistral ~0.66).

### Qualitative human baseline note
Published hard–easy work predicts higher overconfidence on harder items. We compare **directionally**, acknowledging paradigm differences (different item formats and confidence elicitation).

---

## Supplementary finding — constrained output format (add to Method)

We use a constrained output format (`ANSWER` + `CONFIDENCE` only), forbidding
free-form chain-of-thought. This matches common deployed structured/API use,
keeps confidence elicitation comparable across models, and avoids confounding
where visible reasoning text itself changes stated confidence.

We ran a small supplementary diagnostic (`src/diagnostic_format.py`) comparing
constrained vs unconstrained prompting on a handful of bank items. If accuracy
rises when models may reason step-by-step, that supports a format effect on
*accuracy* without invalidating the main calibration analysis: calibration is
about whether confidence tracks accuracy *under the same elicitation protocol*.

## Limitations

1. Verbal confidence ≠ token probabilities.  
2. Constrained `ANSWER`/`CONFIDENCE` format suppresses chain-of-thought and can lower accuracy vs standard MMLU CoT-style evals (especially multi-step math). Our claims concern **miscalibration** (confidence − accuracy) under this elicitation protocol, not absolute leaderboard scores.  
3. MMLU multiple choice ≠ open-ended deployment.  
4. Subject-tier difficulty is a proxy, not item-level IRT.  
5. Human hard–easy baselines use different paradigms — compare directionally only.  
6. Fixed temperature (0.3).  
7. Model-version / date specific (July 2026).  
8. Parse failures reduce usable N (Claude Haiku ~74% parse OK due to verbose replies hitting the token cap before `ANSWER`/`CONFIDENCE`); primary analyses use successfully parsed rows and report parse rates.  


---

## Conclusion / Impact

If overconfidence rises with difficulty, verbal self-reports are a weak standalone safety signal. Practical implication: require abstention, retrieval, or human review when tasks are hard or stakes are high. Our open pipeline (`collect.py` / `analyze.py`) makes the audit re-runnable as models update.

---

## References (seed list — verify + add pages/links)

1. Hendrycks, D., et al. (2021). Measuring Massive Multitask Language Understanding. *ICLR*.  
2. Kadavath, S., et al. (2022). Language Models (Mostly) Know What They Know. arXiv:2207.05221.  
3. Kruger, J., & Dunning, D. (1999). Unskilled and unaware of it. *Journal of Personality and Social Psychology*.  
4. Lichtenstein, S., Fischhoff, B., & Phillips, L. D. (1982). Calibration of probabilities. In Kahneman, Slovic, Tversky (Eds.), *Judgment under Uncertainty*.  
5. Xiong, M., et al. (2024). Can LLMs Express Their Uncertainty? arXiv / EMNLP-related confidence elicitation work — verify exact citation.  
6. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*. (ECE background)

---

## Appendix: AI Use Transparency Statement

**Tools used**
- Claude Haiku (Anthropic) — experimental subject  
- Llama 3.1 / Mistral via Ollama — experimental subjects  
- Cursor / other coding assistants — code scaffolding & debugging  
- HuggingFace `datasets` — MMLU access

**How AI was used**
Helped write pipeline code and polish wording. Did **not** fabricate results, citations, or conclusions.

**How claims were verified**
Citations checked via Google Scholar / Semantic Scholar. Metrics recomputed from `data/raw/final_results.csv` via `src/analyze.py`.

**Responsibility**
[Akul] and [Yohan] take full responsibility and can explain every method step and figure.
