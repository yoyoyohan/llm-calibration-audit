"""
Per-call cost accounting with a hard run-cost circuit breaker.

Pricing sources (verified 2026-07-16):
- OpenAI gpt-4o-mini: https://developers.openai.com/api/docs/models/gpt-4o-mini
  $0.15 / 1M input, $0.60 / 1M output
- Gemini 2.5 Flash paid tier: https://ai.google.dev/gemini-api/docs/pricing
  $0.30 / 1M input (text), $2.50 / 1M output (incl. thinking)
  NOTE: free tier is $0; we always bill against paid-tier rates as a
  conservative overestimate so free-tier runs cannot under-report risk.
- Anthropic Claude Haiku: tracked for completeness using a conservative
  overestimate ($1.00 / 1M in, $5.00 / 1M out) if exact snapshot pricing
  cannot be confirmed at call time — still far below GPT-4o rates.
- Ollama: $0 (local).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from paths import ROOT

COST_LOG_PATH = ROOT / "data" / "v2" / "cost_log.json"
DEFAULT_MAX_RUN_COST_USD = 5.00

# USD per 1M tokens — conservative / verified where noted
PRICING_PER_1M = {
    "openai": {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "default": {"input": 0.15, "output": 0.60},
    },
    "gemini": {
        # Paid-tier gemini-3.1-flash-lite (conservative overestimate vs free tier)
        "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-flash-latest": {"input": 0.30, "output": 2.50},
        "default": {"input": 0.25, "output": 1.50},
    },
    "anthropic": {
        # Conservative overestimate for Haiku-class models
        "default": {"input": 1.00, "output": 5.00},
    },
    "ollama": {
        "default": {"input": 0.0, "output": 0.0},
    },
}


class CostLimitExceeded(RuntimeError):
    """Raised when cumulative estimated cost for the current run exceeds the cap."""


_lock = threading.Lock()
_run_cost_usd = 0.0
_max_run_cost_usd = float(os.getenv("MAX_RUN_COST_USD", DEFAULT_MAX_RUN_COST_USD))


def reset_run_cost(max_run_cost_usd: Optional[float] = None) -> None:
    """Reset the in-memory run accumulator (call at the start of a smoke/collection run)."""
    global _run_cost_usd, _max_run_cost_usd
    with _lock:
        _run_cost_usd = 0.0
        if max_run_cost_usd is not None:
            _max_run_cost_usd = float(max_run_cost_usd)


def get_run_cost() -> float:
    with _lock:
        return _run_cost_usd


def get_max_run_cost() -> float:
    with _lock:
        return _max_run_cost_usd


def _rates(provider: str, model: str) -> dict[str, float]:
    table = PRICING_PER_1M.get(provider.lower(), {"default": {"input": 1.0, "output": 5.0}})
    return table.get(model, table.get("default", {"input": 1.0, "output": 5.0}))


def estimate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    r = _rates(provider, model)
    return (input_tokens / 1_000_000.0) * r["input"] + (output_tokens / 1_000_000.0) * r["output"]


def _load_log(path: Path) -> dict:
    if not path.exists():
        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "totals_usd": {"openai": 0.0, "gemini": 0.0, "anthropic": 0.0, "ollama": 0.0},
            "calls": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def record_call(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    note: str = "",
    cost_usd: Optional[float] = None,
) -> float:
    """
    Record one API call. Returns the USD cost attributed to this call.
    Raises CostLimitExceeded if this run's cumulative cost would exceed the cap.
    """
    global _run_cost_usd
    provider = provider.lower()
    in_tok = int(input_tokens or 0)
    out_tok = int(output_tokens or 0)
    call_cost = float(cost_usd) if cost_usd is not None else estimate_cost_usd(
        provider, model, in_tok, out_tok
    )

    with _lock:
        projected = _run_cost_usd + call_cost
        if projected > _max_run_cost_usd:
            raise CostLimitExceeded(
                f"Run cost would reach ${projected:.4f}, exceeding MAX_RUN_COST_USD="
                f"${_max_run_cost_usd:.2f}. Halting before further spend."
            )
        _run_cost_usd = projected

        COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log = _load_log(COST_LOG_PATH)
        log.setdefault("totals_usd", {})
        log["totals_usd"][provider] = float(log["totals_usd"].get(provider, 0.0)) + call_cost
        log.setdefault("calls", []).append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "provider": provider,
                "model": model,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cost_usd": call_cost,
                "run_cost_usd": _run_cost_usd,
                "note": note,
            }
        )
        log["updated_at"] = datetime.now(timezone.utc).isoformat()
        COST_LOG_PATH.write_text(json.dumps(log, indent=2), encoding="utf-8")

    return call_cost
