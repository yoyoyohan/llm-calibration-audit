# Datasets in this repository

This repo **does not** redistribute the full MMLU benchmark.  
It **does** include the frozen evaluation subset used in our study (questions + choices + labels).

## Primary evaluation bank (v2 submission)

| File | What |
|---|---|
| [`v2/bank_full_v2.json`](./v2/bank_full_v2.json) | **540 items** used for collection (canonical) |
| [`v2/bank_supplement_360.json`](./v2/bank_supplement_360.json) | 360-item supplement (seed 43), content-disjoint from original 180 |
| [`v1_frozen/bank_original_180.json`](./v1_frozen/bank_original_180.json) | Original 180 items (seed 42), frozen |
| [`v1_frozen/question_bank_original_180.csv`](./v1_frozen/question_bank_original_180.csv) | Same 180 as CSV |
| [`question_bank.csv`](./question_bank.csv) | Working copy of the original 180 CSV |

Each item includes: `question_id`, `subject`, `domain`, `difficulty`, `question`, `choice_A`–`choice_D`, `answer_index` / `answer_letter`, and (in v2 JSON) `bank_source` (`original_180` or `supplement_v2`).

## Results

| File | What |
|---|---|
| [`v2/processed/final_results_v2.csv`](./v2/processed/final_results_v2.csv) | 8,100 model responses (Claude / GPT-4o-mini / Gemini) |
| [`v1_frozen/final_results_v1.csv`](./v1_frozen/final_results_v1.csv) | Archival v1 results |
| [`raw/final_results.csv`](./raw/final_results.csv) | Archival v1 combined results |

Raw per-call JSONL under `v2/raw/` is local-only (gitignored).

## Source / license

- **Upstream:** [MMLU](https://huggingface.co/datasets/cais/mmlu) (`cais/mmlu`, test split) — Hendrycks et al. (2021).  
- Sampling / domain / difficulty maps: `config/domains.yaml`, `config/difficulty_map.yaml`.  
- Rebuild scripts (download full MMLU from Hugging Face, then resample):  
  `src/build_question_bank.py`, `scripts/generate_bank_supplement.py`.  
- MMLU remains under its original dataset license; we only ship our stratified subset for reproducibility.
