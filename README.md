# LLM Calibration Audit (NSRI Summer Research Hackathon 2026)

**Title:** Confidence Without Competence? Difficulty-Stratified Calibration of Large Language Models  
**Track:** AI, Data Science & Computing  
**Team:** [Akul] & [Yohan]  
**Repo:** https://github.com/yoyoyohan/llm-calibration-audit  

## Research question

Do LLMs’ verbally elicited confidence ratings become more miscalibrated as task difficulty increases, and does this pattern resemble human hard–easy / overconfidence patterns from cognitive psychology?

## Method (one paragraph)

We sample a frozen, difficulty-stratified subset of [MMLU](https://huggingface.co/datasets/cais/mmlu) across three domains (STEM, Social Science, Humanities). For each question we prompt **Claude Haiku** (Anthropic API), **Llama 3.1** (Ollama), and **Mistral** (Ollama) to return `ANSWER` + `CONFIDENCE` (0–100). We measure accuracy, overconfidence gap (confidence − accuracy), Expected Calibration Error (ECE), and how overconfidence scales with pre-registered difficulty tiers.

## Primary models

| Model | Access |
|---|---|
| Claude Haiku (`claude_haiku`) | Anthropic API |
| Llama 3.1 | Ollama (local) |
| Mistral | Ollama (local) |

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

1. **[`HACKATHON_PLAYBOOK.md`](./HACKATHON_PLAYBOOK.md)** — day-by-day team guide  
2. **[`docs/OVERNIGHT_CLAUDE.md`](./docs/OVERNIGHT_CLAUDE.md)** — Claude collect + merge into existing results  
3. [`paper/research_brief.md`](./paper/research_brief.md) — brief draft  

## Quickstart

```bash
cd llm-calibration-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# set ANTHROPIC_API_KEY=... in .env

ollama pull llama3.1
ollama pull mistral

python src/smoke_claude.py
python src/build_question_bank.py   # only if bank not already frozen
python src/collect.py --models claude_haiku   # appends; does not wipe existing rows
python src/analyze.py
```

Default main scale ≈ `180 questions × 3 models × 2 trials`.

## Repo layout

```
config/          experiment + domain + difficulty maps (pre-registered)
src/             build / collect / analyze pipeline
data/            question bank + raw/processed CSVs
figures/         publication figures
paper/           research brief draft
docs/            Day templates + overnight Claude guide
```

## Key outputs

| Path | What |
|---|---|
| `data/question_bank.csv` | Frozen evaluation set |
| `data/raw/final_results.csv` | All model responses |
| `data/processed/*.csv` | Summary tables |
| `figures/figure2_overconfidence_by_difficulty.png` | Signature hard–easy / overconfidence plot |

## Citation honesty

We claim models may **resemble** hard–easy / overconfidence *patterns*. We do **not** claim to have proven the Dunning–Kruger psychological mechanism in LLMs.

## License

Code for hackathon reuse. MMLU remains under its original dataset license.
