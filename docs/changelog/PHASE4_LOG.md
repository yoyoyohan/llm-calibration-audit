# Phase 4 log — analysis (3 API models)

Stamp: `20260716T175808Z`  
Script: `src/analyze_v2.py`  
Models: `claude_haiku`, `gpt4o_mini`, `gemini_flash` (Llama/Mistral not collected)

## Outputs
- `data/v2/processed/final_results_v2.csv` (8,100 rows; raw response text omitted)
- Tables: `overall_by_model_v2.csv`, `summary_by_model_difficulty_v2.csv`, `response_level_difficulty_regression_v2.csv`, `holm_corrected_tests_v2.csv`, `original180_vs_full540_v2.csv`, `verbal_vs_internal_v2.csv`
- Figures: `figures/v2/figure{1–4}_*.png`

## Headline
- Parse rate 100% all models.
- Claude near-calibrated (ECE≈0.033); GPT-4o-mini and Gemini overconfident.
- Difficulty→overconfidence slopes significant after Holm for all three.
- GPT-4o-mini verbal↔internal *r*≈0.22; internal ECE worse than verbal ECE.
