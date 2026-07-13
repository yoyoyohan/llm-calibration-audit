# LLM Calibration Audit (NSRI Summer Research Hackathon 2026)

**Title:** Confidence Without Competence? Difficulty-Stratified Calibration of Large Language Models  
**Track:** AI, Data Science & Computing  
**Team:** [Akul] & [Yohan]  

## Research question

Do LLMs’ verbally elicited confidence ratings become more miscalibrated as task difficulty increases, and does this pattern resemble human hard–easy / overconfidence patterns from cognitive psychology?

## Method (one paragraph)

We sample a frozen, difficulty-stratified subset of [MMLU](https://huggingface.co/datasets/cais/mmlu) across three domains (STEM, Social Science, Humanities). For each question we prompt Gemini 1.5 Flash, Llama 3.1 (Ollama), and Mistral (Ollama) to return `ANSWER` + `CONFIDENCE` (0–100). We measure accuracy, overconfidence gap (confidence − accuracy), Expected Calibration Error (ECE), and how overconfidence scales with pre-registered difficulty tiers.

## Prompt (exact)

```text
Answer the following multiple-choice question. Then rate your confidence.

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Respond in EXACTLY this format with no other text:
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]
```

## Follow this for the hackathon

1. **[`HACKATHON_PLAYBOOK.md`](./HACKATHON_PLAYBOOK.md)** — merged Akul+Yohan day-by-day (start here)  
2. [`EXECUTION.md`](./EXECUTION.md) — short technical checklist  
3. [`paper/research_brief.md`](./paper/research_brief.md) — brief draft  
4. [`docs/`](./docs/) — Day 1 / Day 2 writing templates  

## Quickstart

```bash
# 1) Setup
cd llm-calibration-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste GEMINI_API_KEY

# 2) Install Ollama models (separate terminal/app)
ollama pull llama3.1
ollama pull mistral

# 3) Verify + smoke-test
python src/setup_verify.py
python src/smoke_test.py --both

# 4) Build frozen question bank (~180 Qs at default settings)
python src/build_question_bank.py

# 5) Collect model responses
python src/collect.py --smoke          # tiny test first
python src/collect.py                  # full run
python src/collect.py --resume         # if interrupted

# 6) Analyze + figures (+ Day 5 dump for Akul)
python src/analyze.py
python src/final_verification.py
```

Default scale ≈ `180 questions × 3 models × 2 trials ≈ 1,080` calls.

## Repo layout

```
config/          experiment + domain + difficulty maps (pre-registered)
src/             build / collect / analyze pipeline
data/            question bank + raw/processed CSVs
figures/         publication figures
paper/           research brief draft
docs/            Day 1 / Day 2 templates for Akul
HACKATHON_PLAYBOOK.md   merged full guide (Akul org + Yohan code)
EXECUTION.md            short technical checklist
```

## Key outputs

| Path | What |
|---|---|
| `data/question_bank.csv` | Frozen evaluation set |
| `data/raw/final_results.csv` | All model responses |
| `data/processed/*.csv` | Summary tables |
| `figures/figure1_reliability_diagrams.png` | Calibration curves |
| `figures/figure2_overconfidence_by_difficulty.png` | Signature DK/hard-easy style plot |
| `figures/figure3_ece_domain_heatmap.png` | Domain ECE |

## Citation honesty

We claim models may **resemble** hard–easy / overconfidence *patterns*. We do **not** claim to have proven the Dunning–Kruger psychological mechanism in LLMs.

## License

Code for hackathon reuse. MMLU remains under its original dataset license.
