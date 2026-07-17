"""
Phase 2 cost-safe smoke test.

3 questions from the ORIGINAL 180-item bank × 5 models × 1 trial = 15 calls.
Does NOT touch the 360 supplement or 540 full bank.

Usage (from repo root):
  python smoke_test_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from clients import (  # noqa: E402
    ClientResponse,
    call_claude,
    call_gemini,
    call_ollama,
    call_openai,
)
from cost_tracker import get_run_cost, reset_run_cost  # noqa: E402
from parse_retry import call_with_parse_retry  # noqa: E402

ORIGINAL_BANK = ROOT / "data" / "v1_frozen" / "question_bank_original_180.csv"
SMOKE_COST_CAP = 1.00

MODELS = [
    ("claude_haiku", "anthropic", call_claude, {"model_id": "claude_haiku"}),
    ("gpt4o_mini", "openai", call_openai, {"model_id": "gpt4o_mini"}),
    ("gemini_flash", "gemini", call_gemini, {"model_id": "gemini_flash", "model_name": "gemini-3.1-flash-lite"}),
    ("llama3.1", "ollama", call_ollama, {"model_id": "llama3.1", "model_name": "llama3.1"}),
    ("mistral", "ollama", call_ollama, {"model_id": "mistral", "model_name": "mistral"}),
]


def pick_three(bank: pd.DataFrame) -> pd.DataFrame:
    """One easy / medium / hard from a single domain (STEM if available)."""
    domain = "STEM" if (bank["domain"] == "STEM").any() else bank["domain"].iloc[0]
    subset = bank[bank["domain"] == domain]
    rows = []
    for diff in ("easy", "medium", "hard"):
        cell = subset[subset["difficulty"] == diff]
        if cell.empty:
            raise RuntimeError(f"No {domain}/{diff} items in original bank")
        rows.append(cell.iloc[0])
    out = pd.DataFrame(rows)
    assert len(out) == 3
    # Safety: every id must be q#### from original bank
    for qid in out["question_id"]:
        if not str(qid).startswith("q"):
            raise RuntimeError(f"Smoke test must use original-bank ids only; got {qid}")
    return out


def fmt_prob(resp: ClientResponse) -> str:
    if resp.internal_prob_answer is None:
        note = resp.internal_prob_note or "not available"
        return f"None - {note}"
    return f"{resp.internal_prob_answer:.6f}"


def main() -> None:
    if not ORIGINAL_BANK.exists():
        raise SystemExit(f"Missing {ORIGINAL_BANK}")

    # Fail fast on missing keys BEFORE any spend
    from dotenv import load_dotenv
    import os
    load_dotenv(ROOT / ".env")
    missing = []
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        v = os.getenv(k) or ""
        if not v or v.startswith("your_"):
            missing.append(k)
    if missing:
        raise SystemExit(
            "Missing/empty API keys in .env (add them yourself — do not paste into chat):\n  "
            + "\n  ".join(missing)
            + "\nThen re-run: python smoke_test_v2.py"
        )

    bank = pd.read_csv(ORIGINAL_BANK)
    assert len(bank) == 180, f"Expected 180 original items, got {len(bank)}"
    questions = pick_three(bank)

    print("=" * 72)
    print("PHASE 2 SMOKE TEST")
    print(f"Bank: {ORIGINAL_BANK} (ORIGINAL 180 ONLY)")
    print(f"Questions: {list(questions['question_id'])} "
          f"({questions.iloc[0]['domain']}: easy/medium/hard)")
    print(f"Models: {[m[0] for m in MODELS]}")
    print(f"Trials: 1 each → {3 * len(MODELS)} logical cells (with parse-retry up to 3 attempts)")
    print(f"max_output_tokens: 1024 (from experiment_v2.yaml)")
    print(f"Cost cap for this run: ${SMOKE_COST_CAP:.2f}")
    print("=" * 72)

    reset_run_cost(max_run_cost_usd=SMOKE_COST_CAP)

    results: list[dict] = []
    for _, q in questions.iterrows():
        choices = [q["choice_A"], q["choice_B"], q["choice_C"], q["choice_D"]]
        for model_id, backend, fn, kwargs in MODELS:
            print("\n" + "-" * 72)
            print(f"CALL  question={q['question_id']}  difficulty={q['difficulty']}  "
                  f"model={model_id}  backend={backend}")
            print(f"Q: {q['question'][:120]}...")
            try:
                resp = call_with_parse_retry(fn, q["question"], choices, max_attempts=3, **kwargs)
                assert isinstance(resp, ClientResponse)
                print("RAW RESPONSE (final attempt):")
                print(resp.raw_text)
                print(
                    f"parsed_answer={resp.parsed_answer}  "
                    f"parsed_confidence={resp.parsed_confidence}  "
                    f"parse_ok={resp.parse_ok}"
                )
                print(f"internal_prob_answer: {fmt_prob(resp)}")
                print(f"call_cost_usd (incl. retries): {resp.call_cost_usd}")
                print(f"running_cumulative_cost_usd: {get_run_cost():.6f}")
                if not resp.parse_ok:
                    print("!!! PARSE STILL FAILED AFTER RETRIES !!!")
                results.append(
                    {
                        "question_id": q["question_id"],
                        "difficulty": q["difficulty"],
                        "model_id": model_id,
                        "parse_ok": resp.parse_ok,
                        "internal_prob_answer": resp.internal_prob_answer,
                        "internal_prob_note": resp.internal_prob_note,
                        "cost": resp.call_cost_usd or 0.0,
                        "error": None,
                        "raw": resp.raw_text,
                    }
                )
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: {type(e).__name__}: {e}")
                results.append(
                    {
                        "question_id": q["question_id"],
                        "difficulty": q["difficulty"],
                        "model_id": model_id,
                        "parse_ok": False,
                        "internal_prob_answer": None,
                        "internal_prob_note": None,
                        "cost": 0.0,
                        "error": str(e),
                        "raw": "",
                    }
                )

    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    print(f"{'model':<14} {'parse_ok':>10} {'internal_prob':>14} {'cost_usd':>12} {'errors':>8}")
    total_cost = 0.0
    for model_id, *_ in MODELS:
        rows = [r for r in results if r["model_id"] == model_id]
        n_ok = sum(1 for r in rows if r["parse_ok"])
        has_prob = any(r["internal_prob_answer"] is not None for r in rows)
        cost = sum(r["cost"] for r in rows)
        errs = sum(1 for r in rows if r["error"])
        total_cost += cost
        print(
            f"{model_id:<14} {n_ok}/3{'':>6} "
            f"{'yes' if has_prob else 'no':>14} "
            f"{cost:>12.6f} {errs:>8}"
        )

    print(f"\nTOTAL estimated smoke cost: ${total_cost:.6f}")
    print(f"cost_tracker run accumulator: ${get_run_cost():.6f}")
    n_ok_all = sum(1 for r in results if r["parse_ok"])
    print(f"OVERALL PARSE SUCCESS: {n_ok_all}/{len(results)}")
    if total_cost > SMOKE_COST_CAP:
        print(
            f"FLAG: smoke cost ${total_cost:.4f} exceeded ${SMOKE_COST_CAP:.2f} — "
            "stop and investigate before any larger run."
        )
        raise SystemExit(2)

    # Confirm bank provenance
    print("\nBank provenance check: ONLY original 180 used:", ORIGINAL_BANK)
    print("Did NOT load data/v2/bank_supplement_360.json or bank_full_v2.json.")

    if n_ok_all != len(results):
        print("\nPARSE RATE NOT PERFECT — fix before Phase 3 collection.")
        raise SystemExit(3)
    print("\nPARSE RATE PERFECT: 15/15 after retries.")


if __name__ == "__main__":
    main()
