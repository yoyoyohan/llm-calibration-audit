# Phase 2 log (v2-phase2-clients)

Running audit trail. Append-only.

## Constraint (confirmed)
Phase 2 implements clients + logprobs + tiny smoke tests only.
**No full-bank (540) collection** in this phase.

## 2026-07-16 — Branch / env
- Created branch `v2-phase2-clients` from `v2-ultimate` @ `v2-phase1-complete`.
- `.env` gitignored; never staged/committed.
- Key presence check (lengths only, values never printed):
  - `ANTHROPIC_API_KEY`: present
  - `OPENAI_API_KEY`: **MISSING** — smoke test blocked until Yohan adds it to local `.env`
  - `GEMINI_API_KEY`: **MISSING** — same

## 2026-07-16 — Task 1: cost_tracker
- Added `src/cost_tracker.py` → writes `data/v2/cost_log.json` after every call.
- Hard halt via `CostLimitExceeded` when run cost > `MAX_RUN_COST_USD` (default **$5.00**).
- Smoke test uses a tighter **$1.00** run cap.
- Pricing used (verified 2026-07-16):
  - **gpt-4o-mini** (OpenAI official model page): $0.15 / 1M in, $0.60 / 1M out
  - **gemini-2.5-flash** paid tier ([Google pricing](https://ai.google.dev/gemini-api/docs/pricing)): $0.30 / 1M in, $2.50 / 1M out
    - Free tier is $0; tracker always uses **paid-tier rates as a conservative overestimate**
  - Anthropic: conservative overestimate $1.00 / $5.00 per 1M (tracked for completeness)
  - Ollama: $0
- `data/v2/cost_log.json` added to `.gitignore`.

## 2026-07-16 — Task 2: OpenAI client
- Added `call_openai()` → Chat Completions API with `logprobs=True`, `top_logprobs=5`.
- Reuses `parse.parse_response` (same regex as all other models).
- Extracts answer-letter token logprob → `exp(logprob)` → `internal_prob_answer`.
- Ambiguous extraction emits `warnings.warn` rather than silent guess.
- Retry: exponential backoff (1s×2^n), max 5, on 429/5xx only; other 4xx raised immediately.
- `record_call` uses real `prompt_tokens` / `completion_tokens` from usage.

## 2026-07-16 — Task 3: Gemini client (pre-modification state → update)
### Pre-modification state
Existing `call_gemini()` in v1 `clients.py`:
- REST `v1beta/models/{model}:generateContent`
- Default model `gemini-flash-latest`
- Optional `thinkingConfig.thinkingBudget=0`
- Returned raw string only; no cost tracker; no logprobs; 3 retries with linear-ish sleep

### Updates
- Default model set to **`gemini-2.5-flash`** (`gemini-2.0-flash` shut down 2026-06-01).
- Unified `ClientResponse` return path; legacy `prompt=` string path kept for `call_model`.
- Requests `responseLogprobs` + `logprobs=5` when supported; strips on 400 and retries.
- If only `avgLogprobs` present → `internal_prob_answer=None` with note
  `"not available at per-token granularity for this model/API version"`.
- Cost tracker + 429/5xx exponential backoff.
- **Paid-tier verification:** API does not return an explicit "free vs paid" flag.
  Verification plan for smoke: (1) billing enabled in Google AI Studio / Cloud,
  (2) successful completion of a request using paid-tier model id `gemini-2.5-flash`,
  (3) cost_tracker bills at paid rates regardless. Confirm dashboard spend after smoke.

## 2026-07-16 — Task 4: Ollama logprobs (Llama 3.1, Mistral)
- Local Ollama version: **0.31.2**
- Docs confirm `/api/generate` supports `logprobs` + `top_logprobs`.
- Live probe on `llama3.1`: **WORKS** — `logprobs` list returned (12 tokens in probe).
- End-to-end extraction test: `internal_prob_answer ≈ 0.999996` for letter B;
  alternatives dict populated. **Ollama included in logprob comparison.**
- Implemented in `_ollama_generate` / `call_ollama` unified path.
- Did not change trials_per_question or collection behavior (Phase 3).

## 2026-07-16 — Task 5: Claude update
- `call_claude()` reads `max_output_tokens` from `config/v2/experiment_v2.yaml` (**512**).
- Model id / no-system-prompt behavior kept identical to v1 Anthropic calls
  (no system message added — verified against prior `call_anthropic`).
- `internal_prob_answer = None` hardcoded with explicit by-design note for Limitations.

## 2026-07-16 — Task 6: Interface consistency test
- `tests/test_client_interface_consistency.py` — **5/5 PASS** (mocked, no API spend).
- Required fields on every client:
  `raw_text, parsed_answer, parsed_confidence, parse_ok, internal_prob_answer,
  internal_prob_alternatives, model_id, backend, config_version`.

## 2026-07-16 — Task 7: Smoke test status
- Script: `smoke_test_v2.py`
- Uses **only** `data/v1_frozen/question_bank_original_180.csv` (3 STEM easy/med/hard).
- **NOT RUN YET** — blocked on missing `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env`.
- After keys are added locally, run: `python smoke_test_v2.py`
- Expected: 15 calls, full raw dumps, summary table, total cost < $1.00.

## 2026-07-16 — Parse-rate hardening (post-smoke)
Smoke failures were: Claude hard truncation before tags (512 tokens), and
occasional Mistral off-format refusals.

Fixes applied:
1. `config/v2/experiment_v2.yaml` `max_output_tokens`: **512 → 1024**
2. New `src/parse_retry.py`: up to 3 attempts; attempt 1 = locked primary
   prompt; attempts 2–3 append a format-repair footer only.
3. Smoke + future collection should call via `call_with_parse_retry`.

Re-smoke result: **15/15 parse_ok**. Claude hard used attempt 2/3 (repair).
Total smoke cost ≈ **$0.0069**. Model for Gemini: `gemini-3.1-flash-lite`.
