"""
v2 full-scale collection (Phase 3).

Usage (from repo root):
  # Dry-run cost / call estimate (NO API calls)
  python src/collect_v2.py --dry-run --models claude_haiku gpt4o_mini gemini_flash

  # Real collection (resumable)
  python src/collect_v2.py --models claude_haiku gpt4o_mini gemini_flash \\
      --max-run-cost 40

  # Ollama overnight later
  python src/collect_v2.py --models llama3.1 mistral --max-run-cost 40
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clients import (  # noqa: E402
    ClientResponse,
    call_claude,
    call_gemini,
    call_ollama,
    call_openai,
)
from cost_tracker import (  # noqa: E402
    CostLimitExceeded,
    estimate_cost_usd,
    get_run_cost,
    reset_run_cost,
)
from parse import letter_to_index  # noqa: E402
from parse_retry import call_with_parse_retry  # noqa: E402
from paths import ROOT, load_experiment_v2, load_yaml  # noqa: E402

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "claude_haiku": {
        "backend": "anthropic",
        "model_name": "claude-haiku-4-5-20251001",
        "fn": call_claude,
        "kwargs": {"model_id": "claude_haiku"},
        "sleep_key": "anthropic_sleep_seconds",
        "provider": "anthropic",
        # conservative overestimate for dry-run (Haiku-class)
        "pricing_model": "default",
    },
    "gpt4o_mini": {
        "backend": "openai",
        "model_name": "gpt-4o-mini",
        "fn": call_openai,
        "kwargs": {"model_id": "gpt4o_mini"},
        "sleep_key": "openai_sleep_seconds",
        "provider": "openai",
        "pricing_model": "gpt-4o-mini",
    },
    "gemini_flash": {
        "backend": "gemini",
        "model_name": "gemini-3.1-flash-lite",
        "fn": call_gemini,
        "kwargs": {"model_id": "gemini_flash", "model_name": "gemini-3.1-flash-lite"},
        "sleep_key": "gemini_sleep_seconds",
        "provider": "gemini",
        "pricing_model": "gemini-3.1-flash-lite",
    },
    "llama3.1": {
        "backend": "ollama",
        "model_name": "llama3.1",
        "fn": call_ollama,
        "kwargs": {"model_id": "llama3.1", "model_name": "llama3.1"},
        "sleep_key": "ollama_sleep_seconds",
        "provider": "ollama",
        "pricing_model": "default",
    },
    "mistral": {
        "backend": "ollama",
        "model_name": "mistral",
        "fn": call_ollama,
        "kwargs": {"model_id": "mistral", "model_name": "mistral"},
        "sleep_key": "ollama_sleep_seconds",
        "provider": "ollama",
        "pricing_model": "default",
    },
}

# Dry-run token assumptions (conservative; from Phase 2 smoke + buffer)
DRY_IN_TOKENS = 350
DRY_OUT_TOKENS = 80
# Wall-clock seconds per successful call (incl. sleep + typical latency)
WALL_SEC = {
    "claude_haiku": 3.5,
    "gpt4o_mini": 2.0,
    "gemini_flash": 5.5,
    "llama3.1": 4.0,
    "mistral": 3.5,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_bank(path: Path) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"Empty or invalid bank: {path}")
    return items


def done_keys_from_jsonl(path: Path) -> set[tuple[str, int]]:
    """Return set of (question_id, trial_index) already written."""
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = rec.get("question_id")
            trial = rec.get("trial_index")
            if qid is not None and trial is not None:
                done.add((str(qid), int(trial)))
    return done


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def dry_run(models: list[str], n_items: int, trials: int, cfg: dict) -> None:
    print("=" * 72)
    print("DRY RUN — no API calls")
    print("=" * 72)
    print(f"bank items: {n_items}")
    print(f"trials_per_question: {trials}")
    print(f"max_output_tokens: {cfg.get('max_output_tokens')}")
    print(f"temperature: {cfg.get('temperature')}")
    print(f"config_version: {cfg.get('config_version')}")
    print()
    total_calls = 0
    total_cost = 0.0
    total_hours = 0.0
    print(f"{'model':<14} {'calls':>8} {'est_cost_usd':>14} {'est_hours':>10}")
    for mid in models:
        meta = MODEL_REGISTRY[mid]
        calls = n_items * trials
        total_calls += calls
        cost = estimate_cost_usd(
            meta["provider"],
            meta["pricing_model"],
            DRY_IN_TOKENS * calls,
            DRY_OUT_TOKENS * calls,
        )
        hours = (calls * WALL_SEC.get(mid, 3.0)) / 3600.0
        total_cost += cost
        total_hours += hours
        print(f"{mid:<14} {calls:>8} {cost:>14.4f} {hours:>10.2f}")
    print("-" * 72)
    print(f"{'TOTAL':<14} {total_calls:>8} {total_cost:>14.4f} {total_hours:>10.2f}")
    print()
    print("Assumptions: ~350 in / ~80 out tokens per call (conservative).")
    print("Parse retries may add ~5–15% extra calls/cost on hard items.")
    print("If TOTAL looks sane (~$1–$30 for API trio), proceed without --dry-run.")


def build_record(
    item: dict,
    *,
    model_id: str,
    meta: dict,
    trial_index: int,
    resp: ClientResponse | None,
    error: str | None,
    cfg: dict,
) -> dict:
    answer = resp.parsed_answer if resp else None
    conf = resp.parsed_confidence if resp else None
    parse_ok = bool(resp.parse_ok) if resp else False
    pred_idx = letter_to_index(answer)
    ans_idx = int(item["answer_index"])
    is_correct = bool(pred_idx == ans_idx) if pred_idx is not None else False
    return {
        "config_version": str(cfg.get("config_version", "v2")),
        "max_output_tokens": int(cfg.get("max_output_tokens", 1024)),
        "temperature": float(cfg.get("temperature", 0.3)),
        "bank_source": item.get("bank_source"),
        "question_id": item["question_id"],
        "subject": item.get("subject"),
        "domain": item.get("domain"),
        "difficulty": item.get("difficulty"),
        "model_id": model_id,
        "backend": meta["backend"],
        "model_name": meta["model_name"],
        "trial_index": trial_index,
        "collection_timestamp": utc_now(),
        "raw_response": resp.raw_text if resp else None,
        "predicted_letter": answer,
        "correct_letter": item.get("answer_letter"),
        "is_correct": is_correct,
        "confidence": conf,
        "parse_ok": parse_ok,
        "internal_prob_answer": resp.internal_prob_answer if resp else None,
        "internal_prob_alternatives": resp.internal_prob_alternatives if resp else None,
        "internal_prob_note": resp.internal_prob_note if resp else None,
        "call_cost_usd": resp.call_cost_usd if resp else None,
        "input_tokens": resp.input_tokens if resp else None,
        "output_tokens": resp.output_tokens if resp else None,
        "error": error,
    }


def collect_model(
    model_id: str,
    items: list[dict],
    trials: int,
    cfg: dict,
    out_dir: Path,
    run_stamp: str,
) -> Path:
    meta = MODEL_REGISTRY[model_id]
    out_path = out_dir / f"{model_id}_{run_stamp}.jsonl"
    # Also support resume into the latest existing file for this model if --resume-latest
    done = done_keys_from_jsonl(out_path)
    sleep_s = float(cfg.get(meta["sleep_key"], 0.5))
    parse_attempts = int(cfg.get("parse_max_attempts", 3))
    fn: Callable = meta["fn"]

    planned = len(items) * trials
    remaining = planned - len(done)
    print(f"\n=== {model_id}: writing → {out_path}")
    print(f"    planned={planned} already_done={len(done)} remaining={remaining}")

    pbar = tqdm(total=remaining, desc=model_id)
    n_done_this = 0
    for item in items:
        choices = [item["choice_A"], item["choice_B"], item["choice_C"], item["choice_D"]]
        for trial in range(trials):
            key = (str(item["question_id"]), trial)
            if key in done:
                continue
            resp: ClientResponse | None = None
            err: str | None = None
            try:
                resp = call_with_parse_retry(
                    fn,
                    item["question"],
                    choices,
                    max_attempts=parse_attempts,
                    **meta["kwargs"],
                )
            except CostLimitExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"

            rec = build_record(
                item,
                model_id=model_id,
                meta=meta,
                trial_index=trial,
                resp=resp,
                error=err,
                cfg=cfg,
            )
            append_jsonl(out_path, rec)
            done.add(key)
            n_done_this += 1
            pbar.update(1)

            if n_done_this % 100 == 0:
                print(
                    f"\n[{model_id}] checkpoint: {n_done_this} new calls | "
                    f"run_cost=${get_run_cost():.4f}"
                )

            time.sleep(sleep_s)

    pbar.close()
    return out_path


def integrity_check(path: Path, expected: int) -> dict:
    n = 0
    parse_ok = 0
    bad_cfg = 0
    bad_tok = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            n += 1
            if rec.get("parse_ok"):
                parse_ok += 1
            if rec.get("config_version") != "v2":
                bad_cfg += 1
            if int(rec.get("max_output_tokens") or 0) != 1024:
                bad_tok += 1
    return {
        "path": str(path),
        "n": n,
        "expected": expected,
        "parse_ok_rate": (parse_ok / n) if n else 0.0,
        "bad_config_version": bad_cfg,
        "bad_max_tokens": bad_tok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="v2 collection (540×5)")
    parser.add_argument(
        "--config",
        default="config/v2/experiment_v2.yaml",
        help="Path to experiment_v2.yaml",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model ids, e.g. claude_haiku gpt4o_mini gemini_flash",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-run-cost",
        type=float,
        default=40.0,
        help="Halt if estimated run cost exceeds this (default $40 for Phase 3)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/v2/raw",
        help="Directory for versioned jsonl outputs",
    )
    parser.add_argument(
        "--run-stamp",
        default=None,
        help="Reuse a stamp to resume into the same jsonl files (e.g. 20260716T180000Z)",
    )
    parser.add_argument("--limit-questions", type=int, default=None)
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = load_yaml(cfg_path)
    bank_path = ROOT / cfg["full_bank_source"]
    items = load_bank(bank_path)
    if args.limit_questions:
        items = items[: args.limit_questions]

    trials = int(cfg["trials_per_question"])
    for mid in args.models:
        if mid not in MODEL_REGISTRY:
            raise SystemExit(f"Unknown model id: {mid}. Choose from {list(MODEL_REGISTRY)}")

    if args.dry_run:
        dry_run(args.models, len(items), trials, cfg)
        return

    # Real collection
    print("=" * 72)
    print("PHASE 3 COLLECTION — LIVE")
    print(f"MAX_RUN_COST_USD for this process: ${args.max_run_cost:.2f}")
    print(f"Bank: {bank_path} ({len(items)} items)")
    print(f"Trials: {trials}")
    print(f"Models: {args.models}")
    print(f"max_output_tokens: {cfg.get('max_output_tokens')}")
    print("=" * 72)

    reset_run_cost(max_run_cost_usd=float(args.max_run_cost))
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = args.run_stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"run_stamp: {run_stamp}  (re-use with --run-stamp to resume)")

    outputs: list[Path] = []
    try:
        for mid in args.models:
            path = collect_model(mid, items, trials, cfg, out_dir, run_stamp)
            outputs.append(path)
    except CostLimitExceeded as e:
        print(f"\nHALTED by cost circuit breaker: {e}")
        print(f"Run cost so far: ${get_run_cost():.4f}")
        print(f"Resume with: --run-stamp {run_stamp}")
        raise SystemExit(4) from e

    expected = len(items) * trials
    print("\n" + "=" * 72)
    print("POST-COLLECTION INTEGRITY")
    print("=" * 72)
    print(f"{'model':<14} {'n':>6} {'expected':>8} {'parse_ok':>10} {'bad_cfg':>8} {'bad_tok':>8}")
    for path in outputs:
        mid = next((m for m in args.models if path.name.startswith(m + "_")), path.stem)
        info = integrity_check(path, expected)
        print(
            f"{mid:<14} {info['n']:>6} {info['expected']:>8} "
            f"{info['parse_ok_rate']:>10.3f} {info['bad_config_version']:>8} "
            f"{info['bad_max_tokens']:>8}"
        )
    print(f"\nTotal run cost (tracked): ${get_run_cost():.4f}")


if __name__ == "__main__":
    main()
