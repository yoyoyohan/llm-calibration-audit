# Methods changelog (v2 expansion)

Plain-English record of what changed after the hackathon-submitted (v1) study.
Written so a stranger can understand where each dataset came from.

## Study versions

| Version | What it is |
|---|---|
| **v1 (hackathon)** | Frozen 180-item MMLU bank (seed 42); Claude Haiku + Llama 3.1 + Mistral; temperature 0.3; max_output_tokens 256; 2 trials; constrained `ANSWER`/`CONFIDENCE` prompt |
| **v2 (in progress)** | Same domains/difficulty maps and same prompt text; expanded bank + standardized decoding knobs; new models deferred to Phase 2 |

## Question bank

### What stayed the same
- Source corpus: MMLU test split (`cais/mmlu`) via Hugging Face.
- Domain grouping: `config/domains.yaml` (STEM, Social_Science, Humanities) — **unchanged**.
- Difficulty tiers: `config/difficulty_map.yaml` (subject-level easy/medium/hard) — **unchanged**.
- The original 180 items used in the hackathon collection are **preserved byte-for-byte** under `data/v1_frozen/` and were **not edited**.

### What changed (Phase 1)
- Added a **separate** sampling pass with fixed random seed **43** (not 42) to draw up to **40 new questions per domain × difficulty cell** that do **not** overlap the original 180 (overlap checked by `subject + question text`, not by bank-local `question_id`).
- Target supplement size: 3 × 3 × 40 = **360** new items.
- Combined file: `data/v2/bank_full_v2.json` (up to **540** items).
- Every item carries `bank_source`:
  - `"original_180"` — hackathon frozen items
  - `"supplement_v2"` — new Phase 1 items  
  This field must be kept through collection and analysis so subsets can be analyzed separately or together.

### Why expand the bank
Cell sizes of 20 in v1 produce wide uncertainty for bootstrap CIs and empirical difficulty work. Expanding coverage improves statistical power without rewriting or discarding the original frozen bank.

### Known data quirk: duplicate item in the original 180 (q0060 / q0073)
Post-hoc review during v2 setup identified that two items in the original 180-item bank, `q0060` and `q0073`, share **identical** question content — same stem, same four answer choices, and same correct answer (`high_school_psychology`, Social_Science, easy; answer B, "semantic memory"). This is not a sampling bug on our side: the source `cais/mmlu` test split itself contains this exact record twice (dataset indices 5267 and 5283), and the v1 builder drew both. The original evaluation set therefore contained **179 functionally unique items rather than 180**.

Handling:
- The original 180-item bank and the v1 collected results are **frozen and not edited**, to preserve the submitted audit trail.
- The effect on original aggregate results is negligible (1 of 180 items), so it is **retained** in aggregate accuracy/ECE.
- For v2 analyses that treat each item as an independent unit (e.g., empirical difficulty terciles), **one of the two duplicate IDs is excluded**.
- A full near-duplicate scan of the combined 540-item bank found **no other exact duplicates**; four high-similarity pairs are legitimate distinct questions that share a reading passage (world history) or boilerplate (formal logic).

Suggested paper Limitations wording: "Post-hoc review identified that two items in the original 180-item bank (q0060, q0073) share identical question content, meaning the original evaluation set contained 179 functionally unique items rather than 180. The duplication originates in the source MMLU dataset. It was retained in aggregate metrics given its scale (1 of 180 items) but excluded from item-level independence analyses in the expanded study; it is disclosed here for transparency."

## Decoding / elicitation settings (standardized for v2 collection)

| Setting | v1 | v2 | Why |
|---|---|---|---|
| Temperature | 0.3 | **0.3** (unchanged) | Keep elicitation comparable |
| max_output_tokens | 256 | **512** | Reduce truncation before `ANSWER`/`CONFIDENCE` tags |
| trials_per_question | 2 | **5** | More stable per-item estimates |
| Prompt template | constrained ANSWER + CONFIDENCE | **byte-identical text** | Avoid protocol confound |

Config file: `config/v2/experiment_v2.yaml`.

The v2 prompt template was verified **byte-for-byte identical** (303 characters, UTF-8) against the live runtime prompt produced by `src/prompts.py` `build_confidence_prompt()`. Note: the standalone documentation constant `PROMPT_TEMPLATE_FOR_PAPER` in `src/prompts.py` is currently stale (missing the "Example of a valid reply" block) and should be synced before being quoted in the paper appendix; it is not used during collection and is not a data confound.

## Models

### v1 (already collected; archived)
- Claude Haiku (Anthropic API)
- Llama 3.1 (Ollama)
- Mistral (Ollama)
- Raw responses archived at `data/v1_frozen/final_results_v1.csv`

### Phase 2+ (placeholder — not implemented in Phase 1)
- Planned additions such as GPT-4o-mini and Gemini Flash (API clients, collection, and logprob extraction belong in later phases).
- When added, they will use the **same** v2 token budget, temperature, trial count, and prompt template.

## Analysis upgrades (placeholder — later phases)
- Response-level difficulty regression (already prototyped on v1 data during hackathon revision).
- Bootstrap CIs; Holm–Bonferroni for planned contrasts.
- Empirical item difficulty (with circularity caveats).
- Verbal vs token-probability confidence comparison where logprobs are available.

## What Phase 1 explicitly did **not** do
- No new API client code.
- No new model collection runs.
- No edits to `config/domains.yaml` or `config/difficulty_map.yaml`.
- No deletion of v1 originals or `data/v1_frozen/` safety copies.
