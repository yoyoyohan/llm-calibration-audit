# Phase 3 log — full collection

## Dry-run (2026-07-16) — Claude + GPT-4o-mini + Gemini
Bank: `data/v2/bank_full_v2.json` (540) × 5 trials = 2700 calls/model.

| model | calls | est_cost_usd | est_hours |
|---|---:|---:|---:|
| claude_haiku | 2700 | 2.03 | 2.62 |
| gpt4o_mini | 2700 | 0.27 | 1.50 |
| gemini_flash (`gemini-3.1-flash-lite`) | 2700 | 0.56 | 4.12 |
| **TOTAL** | **8100** | **~$2.86** | **~8.25** |

MAX_RUN_COST_USD set to **$40** for live run.
Ollama (Llama/Mistral): **not collected** for this submission (scope decision 2026-07-17).

## Live run (stamp `20260716T175808Z`)
Command:
```
python src/collect_v2.py --models claude_haiku gpt4o_mini gemini_flash \
  --max-run-cost 40 --run-stamp 20260716T175808Z
```
Outputs: `data/v2/raw/{model}_20260716T175808Z.jsonl` (resumable).

### Post-collection integrity
| model | n | expected | parse_ok | bad_cfg | bad_tok |
|---|---:|---:|---:|---:|---:|
| claude_haiku | 2700 | 2700 | 1.000 | 0 | 0 |
| gpt4o_mini | 2700 | 2700 | 1.000 | 0 | 0 |
| gemini_flash | 2700 | 2700 | 1.000 | 0 | 0 |

Total tracked run cost: **$2.7076**

Published analysis table (no raw response text): `data/v2/processed/final_results_v2.csv`
