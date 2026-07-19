# LLM Calibration Audit (NSRI Summer Research Hackathon 2026)

**Title:** Difficulty-Stratified Evaluation of Confidence Calibration in Large Language Models  
**Track:** AI, Data Science & Computing  
**Team:** Yohan Deshpande & Akul Bajpai  
**Repo:** https://github.com/yoyoyohan/llm-calibration-audit  

## Final submission (v2)

| Item | Path |
|---|---|
| LaTeX manuscript | [`paper/manuscript_v2.tex`](./paper/manuscript_v2.tex) |
| Manuscript figures | [`paper/figures/`](./paper/figures/) |
| Research brief | [`paper/research_brief.md`](./paper/research_brief.md) |
| Processed results (8,100 rows) | [`data/v2/processed/final_results_v2.csv`](./data/v2/processed/final_results_v2.csv) |
| Analysis figures | [`figures/v2/`](./figures/v2/) |

**Models:** Claude Haiku · GPT-4o-mini · Gemini 3.1 Flash-Lite  
**Bank:** 540 MMLU items (180 frozen + 360 supplement) × **5** trials · temperature 0.3 · max 1024 tokens · **100%** parse  

Open-weight local models (Llama 3.1 / Mistral) were **not** collected in this final submission. Archival v1 results remain under `data/v1_frozen/` and `data/raw/final_results.csv`.

## Research question

Do LLMs’ verbally elicited confidence ratings become more miscalibrated as task difficulty increases, and does this pattern resemble human hard–easy / overconfidence patterns from cognitive psychology?

## Headline results (full 540 × 5)

| Model | N | Accuracy | Overconf. gap | ECE |
|---|---:|---:|---:|---:|
| Claude Haiku | 2700 | 0.895 | −0.017 | 0.033 |
| Gemini Flash-Lite | 2700 | 0.876 | +0.115 | 0.117 |
| GPT-4o-mini | 2700 | 0.715 | +0.156 | 0.157 |

## Reproduce analysis

```bash
cd llm-calibration-audit
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Requires local raw JSONL under data/v2/raw/ (gitignored) OR use published CSV:
python src/analyze_v2.py --stamp 20260716T175808Z
```

Collection (API keys in `.env` only — never commit):

```bash
cp .env.example .env
python src/collect_v2.py --dry-run --models claude_haiku gpt4o_mini gemini_flash
python src/collect_v2.py --models claude_haiku gpt4o_mini gemini_flash --max-run-cost 40
```

## Repo layout

```
config/v2/          v2 experiment config (locked prompt + protocol)
src/                collect_v2 / analyze_v2 / clients / parse_retry
data/v1_frozen/     archival hackathon v1 bank + results
data/v2/            540-item bank + processed v2 tables
figures/v2/         analysis figures
paper/              manuscript_v2.tex, figures/, research brief
docs/changelog/     Phase 1–4 audit logs
```

## Citation honesty

We claim models may **resemble** hard–easy / overconfidence *patterns*. We do **not** claim to have proven the Dunning–Kruger psychological mechanism in LLMs.

## License

Code for hackathon reuse. MMLU remains under its original dataset license.
