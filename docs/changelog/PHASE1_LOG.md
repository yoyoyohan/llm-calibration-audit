# Phase 1 log (v2-ultimate foundation)

Running audit trail for Phase 1. Append-only decisions.

## 2026-07-16 — Branch / start
- Confirmed branch: `v2-ultimate` (not main).
- Created directories: `config/v2/`, `data/v1_frozen/`, `data/v2/`, `docs/changelog/`, `scripts/`.

## 2026-07-16 — Task 1: Hygiene
- Updated `.gitignore` to cover `.env`, `.env.*` (except `.env.example`), `*.key`.
- Updated `.env.example` with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY` placeholders only.
- Ran `git log --all --full-history -- .env` and `git ls-files` for key files: **`.env` was never committed**; no API key files tracked.

## 2026-07-16 — Task 2: Preserve v1 frozen data
- **Assumption:** Original bank lives at `data/question_bank.csv` (CSV), not JSON. Copied byte-for-byte to `data/v1_frozen/question_bank_original_180.csv` (SHA256 match with source).
- Also wrote lossless JSON mirror `data/v1_frozen/bank_original_180.json` for the path referenced by `experiment_v2.yaml`.
- Copied v1 results to `data/v1_frozen/final_results_v1.csv` (do not move originals).
- Set `chmod 444` on frozen copies.
- Originals under `data/question_bank.csv` and `data/raw/final_results.csv` left in place.
- Note: original 180-item CSV has **180 rows but 179 unique (subject||question) keys** — one duplicated stem already present in the hackathon bank. Overlap filtering uses content keys, so that duplicate is treated as already-used.

## 2026-07-16 — Task 3: Unified config
- Created `config/v2/experiment_v2.yaml` with temperature 0.3, max_output_tokens 512, trials_per_question 5, exact v1 prompt template (including example reply from `src/prompts.py` `build_confidence_prompt`).
- Did **not** modify `config/experiment.yaml`, `config/domains.yaml`, or `config/difficulty_map.yaml`.

## 2026-07-16 — Task 4: Bank supplement
- Created `scripts/generate_bank_supplement.py` and `scripts/verify_bank_v2.py`.
- Sampling: seed **43**, 40/cell, same `domains.yaml` + `difficulty_map.yaml`, exclude content keys already in original 180.
- Result: **360** supplement items; **0** shortfalls; full bank **540** items.
- Social_Science/hard pool after exclusion was only 94 available (still enough for 40).
- Outputs: `data/v2/bank_supplement_360.json`, `data/v2/bank_full_v2.json` with `bank_source` field.

## 2026-07-16 — Task 5: Docs
- Created `docs/METHODS_CHANGELOG.md` (plain-English methods trail).
- No API client / collect / analyze code modified in this phase.

## 2026-07-16 — Pre-commit review: duplicate-item diagnosis (q0060 / q0073)
Reviewer flagged the "179 unique keys" note for a real diagnosis before commit.
Full field-by-field comparison run (read-only, no files modified):

| Field | q0060 | q0073 |
|---|---|---|
| subject | high_school_psychology | high_school_psychology |
| domain | Social_Science | Social_Science |
| difficulty | easy | easy |
| question | "Knowledge of different categories of trees and where they grow best is an example of what kind of long-term memory?" | *(identical)* |
| choice_A | episodic memory | episodic memory |
| choice_B | semantic memory | semantic memory |
| choice_C | procedural memory | procedural memory |
| choice_D | eidetic memory | eidetic memory |
| answer_index / letter | 1 / B | 1 / B |

**Verdict: Possibility A — TRUE DUPLICATE.** All evaluated fields identical.

**Root cause traced to source, not our pipeline.** The raw `cais/mmlu` `all`
test split (14,042 rows) contains this exact record **twice**, at dataset
indices **5267** and **5283** (verified by direct dataset scan). The v1 bank
builder sampled both source rows; it did not accidentally copy one item. So the
original hackathon bank contains **179 functionally unique items, not 180**.

**Full-bank near-duplicate scan (all 540 items):**
- Exactly **1** exact-normalized duplicate group: `q0060` / `q0073` (the above).
- **4** fuzzy pairs ≥0.90 similarity, all confirmed **legitimate distinct
  questions** that share a reading passage / boilerplate:
  - `q0163` / `s0357` (formal_logic) — different logical arguments
  - `q0127` / `s0252` (high_school_world_history) — shared al-Bakri passage, different questions
  - `q0133` / `s0279` (high_school_world_history) — shared Ibn Battuta passage, different questions
  - `q0122` / `s0244` (high_school_world_history) — shared Nkrumah passage, different questions
- No other true duplicates. The 360-item supplement introduces **no** exact
  duplicate of any original item (content-key overlap = 0, re-confirmed).

**Resolution (decided):**
- v1 frozen bank and v1 results are **left untouched** (they are the submitted
  record; editing would break the audit trail).
- For v2 **item-level** analyses that assume item independence (e.g. empirical
  difficulty terciles), **one of {q0060, q0073} is to be dropped**. Both are
  **kept** in aggregate accuracy / ECE (immaterial at 1/180 scale).
- Disclosure text added to `METHODS_CHANGELOG.md` (and to be carried into the
  paper Limitations section).

## 2026-07-16 — Pre-commit review: prompt lock verified (byte-level)
- Compared the **live** runtime prompt from `src/prompts.py`
  `build_confidence_prompt()` against the `prompt_template` string in
  `config/v2/experiment_v2.yaml`.
- Both strings: **303 characters**, **character-identical**, **UTF-8
  byte-identical**; unified diff empty. **No v1↔v2 prompt confound.**
- Side finding: the documentation constant `PROMPT_TEMPLATE_FOR_PAPER` in
  `src/prompts.py` is **stale** — it omits the 4-line "Example of a valid
  reply:" block that the live prompt (and v2 YAML) contain. This constant is
  **not used for collection**, so it is not a data confound, but it should be
  synced before it is quoted in the paper appendix. (Not modified in Phase 1.)

## 2026-07-16 — Manual spot-check (human sign-off)
- Protocol: ≥15 supplement items from `data/v2/bank_supplement_360.json`,
  2–3 per each of the 9 domain×difficulty cells; Yohan + Akul independently.
- RESULT: Manually verified 15+ supplement items across cells — no formatting,
  domain-assignment, answer-validity, or encoding issues found. Sign-off:
  Yohan (spot-check clean, 2026-07-16). Frozen copies under `data/v1_frozen/`
  also opened and confirmed readable.
