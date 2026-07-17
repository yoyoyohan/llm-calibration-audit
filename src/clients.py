"""
LLM backends for v2: Anthropic Claude, OpenAI, Gemini, Ollama.

Phase 2 unified interface returns ClientResponse for every provider.
Legacy string helpers (call_anthropic / raw generate) remain for v1 collect.py.
"""
from __future__ import annotations

import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import requests
from dotenv import load_dotenv

from cost_tracker import record_call
from parse import parse_response
from paths import ROOT, load_experiment, load_experiment_v2
from prompts import SYSTEM_STYLE_NOTE, build_confidence_prompt

load_dotenv(ROOT / ".env")

CONFIG_VERSION = "v2"


def _clean_api_key(raw: str | None) -> str:
    """Strip whitespace and common copy-paste invisibles (ZWSP, BOM, etc.)."""
    if not raw:
        return ""
    # Remove BOM / zero-width / non-breaking spaces that break latin-1 headers
    for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\xa0"):
        raw = raw.replace(ch, "")
    return raw.strip()


REQUIRED_RESPONSE_FIELDS = (
    "raw_text",
    "parsed_answer",
    "parsed_confidence",
    "parse_ok",
    "internal_prob_answer",
    "internal_prob_alternatives",
    "model_id",
    "backend",
    "config_version",
)


class LLMClientError(RuntimeError):
    pass


@dataclass
class ClientResponse:
    raw_text: str
    parsed_answer: Optional[str]
    parsed_confidence: Optional[int]
    parse_ok: bool
    internal_prob_answer: Optional[float]
    internal_prob_alternatives: Optional[dict[str, float]]
    model_id: str
    backend: str
    config_version: str
    internal_prob_note: Optional[str] = None
    call_cost_usd: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _v2_defaults() -> tuple[float, int]:
    """Temperature + max_output_tokens from experiment_v2.yaml (fallback to v1)."""
    try:
        cfg = load_experiment_v2()
    except Exception:  # noqa: BLE001
        cfg = load_experiment()
    temp = float(cfg.get("temperature", 0.3))
    max_tok = int(cfg.get("max_output_tokens", 512))
    return temp, max_tok


def _finalize(
    raw_text: str,
    *,
    model_id: str,
    backend: str,
    internal_prob_answer: Optional[float],
    internal_prob_alternatives: Optional[dict[str, float]],
    internal_prob_note: Optional[str] = None,
    call_cost_usd: Optional[float] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
) -> ClientResponse:
    answer, conf, parse_ok = parse_response(raw_text)
    return ClientResponse(
        raw_text=raw_text or "",
        parsed_answer=answer,
        parsed_confidence=conf,
        parse_ok=parse_ok,
        internal_prob_answer=internal_prob_answer,
        internal_prob_alternatives=internal_prob_alternatives,
        model_id=model_id,
        backend=backend,
        config_version=CONFIG_VERSION,
        internal_prob_note=internal_prob_note,
        call_cost_usd=call_cost_usd,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _retry_sleep(attempt: int) -> None:
    # Exponential backoff starting at 1s: 1, 2, 4, 8, 16
    time.sleep(min(2 ** attempt, 32))


def _should_retry_status(code: int) -> bool:
    return code == 429 or 500 <= code <= 599


# ---------------------------------------------------------------------------
# Logprob helpers
# ---------------------------------------------------------------------------

def _token_is_letter(tok: str, letter: str) -> bool:
    t = (tok or "").strip()
    return t.upper() == letter.upper()


def extract_answer_letter_logprob(
    tokens: Sequence[dict[str, Any]],
    answer_letter: Optional[str],
    *,
    token_key: str = "token",
    logprob_key: str = "logprob",
    top_key: str = "top_logprobs",
) -> tuple[Optional[float], Optional[dict[str, float]], Optional[str]]:
    """
    Find the logprob of the answer letter token in a sequence of token logprob dicts.

    Returns (prob, alternatives_dict, note). Warns if ambiguous.
    """
    if not answer_letter or not tokens:
        return None, None, "no answer letter or empty logprobs"

    matches: list[int] = []
    for i, t in enumerate(tokens):
        tok = str(t.get(token_key, ""))
        if _token_is_letter(tok, answer_letter):
            matches.append(i)
        # Also accept tokens like " A" or "A\n"
        elif tok.strip().upper().startswith(answer_letter.upper()) and len(tok.strip()) <= 2:
            if tok.strip().upper()[0] == answer_letter.upper():
                matches.append(i)

    if not matches:
        # Heuristic: look for a token that is exactly one of A/B/C/D near "ANSWER"
        for i, t in enumerate(tokens):
            tok = str(t.get(token_key, "")).strip().upper()
            if tok in {"A", "B", "C", "D"} and tok == answer_letter.upper():
                matches.append(i)

    if not matches:
        msg = (
            f"logprob extraction ambiguous/failed: no token matching answer "
            f"letter '{answer_letter}' among {len(tokens)} tokens"
        )
        warnings.warn(msg, stacklevel=2)
        return None, None, msg

    if len(matches) > 1:
        msg = (
            f"logprob extraction ambiguous: {len(matches)} tokens match "
            f"'{answer_letter}' at indices {matches}; using first match"
        )
        warnings.warn(msg, stacklevel=2)
    else:
        msg = None

    idx = matches[0]
    entry = tokens[idx]
    lp = entry.get(logprob_key)
    if lp is None:
        return None, None, "matched token missing logprob"

    prob = float(math.exp(float(lp)))
    alts: dict[str, float] = {}
    top = entry.get(top_key) or []
    for alt in top:
        if isinstance(alt, dict):
            at = str(alt.get(token_key, alt.get("token", "")))
            alp = alt.get(logprob_key, alt.get("logprob"))
            if at and alp is not None:
                alts[at] = float(math.exp(float(alp)))
    return prob, (alts or None), msg


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------

def call_claude(
    question: str,
    choices: Sequence[str],
    model_id: str = "claude_haiku",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    prompt_suffix: str = "",
) -> ClientResponse:
    """
    Unified Claude client. max_output_tokens defaults to experiment_v2.yaml.
    Anthropic's public API does not expose token-level log probabilities as of this
    implementation; Claude is excluded from internal-probability comparison analyses
    by design, not by data availability failure.
    """
    default_temp, default_max = _v2_defaults()
    temp = default_temp if temperature is None else temperature
    max_tok = default_max if max_tokens is None else int(max_tokens)
    prompt = build_confidence_prompt(question, choices) + (prompt_suffix or "")
    model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    raw, usage = _anthropic_raw(prompt, model_name=model, temperature=temp, max_tokens=max_tok)
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    cost = record_call("anthropic", model, in_tok, out_tok, note=f"model_id={model_id}")
    return _finalize(
        raw,
        model_id=model_id,
        backend="anthropic",
        # Anthropic's public API does not expose token-level log probabilities as of
        # this implementation; Claude is excluded from internal-probability comparison
        # analyses by design, not by data availability failure.
        internal_prob_answer=None,
        internal_prob_alternatives=None,
        internal_prob_note=(
            "Anthropic's public API does not expose token-level log probabilities "
            "as of this implementation; Claude is excluded from internal-probability "
            "comparison analyses by design, not by data availability failure."
        ),
        call_cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def _anthropic_raw(
    prompt: str,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    api_key = _clean_api_key(os.getenv("ANTHROPIC_API_KEY"))
    if not api_key or api_key.startswith("your_"):
        raise LLMClientError("ANTHROPIC_API_KEY missing in .env")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    # No system prompt — must match v1 Anthropic behavior exactly.
    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    last_err: Exception | None = None
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if _should_retry_status(resp.status_code):
                last_err = LLMClientError(f"{resp.status_code}: {resp.text[:300]}")
                _retry_sleep(attempt)
                continue
            if 400 <= resp.status_code < 500:
                raise LLMClientError(f"Anthropic {resp.status_code}: {resp.text[:400]}")
            resp.raise_for_status()
            data = resp.json()
            parts = data.get("content") or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text"]
            text = "\n".join(t for t in texts if t).strip()
            if not text:
                raise LLMClientError(f"Anthropic empty content: {data}")
            usage = data.get("usage") or {}
            return text, {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        except LLMClientError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            _retry_sleep(attempt)
    raise LLMClientError(f"Anthropic call failed after retries: {last_err}")


def call_anthropic(prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None) -> str:
    """Legacy string interface used by v1 collect.py."""
    default_temp, default_max = _v2_defaults()
    # Preserve v1 collect behavior: use experiment.yaml when present for legacy path
    cfg = load_experiment()
    temp = cfg["temperature"] if temperature is None else temperature
    # Phase 2: prefer v2 max tokens even for legacy path when collecting under v2
    max_tok = int(cfg.get("max_output_tokens", default_max))
    # If caller is on v2 branch tooling, experiment.yaml may still say 256; allow env override
    if os.getenv("USE_V2_MAX_TOKENS", "1") == "1":
        max_tok = default_max
    model = model_name or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    text, usage = _anthropic_raw(prompt, model_name=model, temperature=float(temp), max_tokens=max_tok)
    record_call(
        "anthropic",
        model,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        note="legacy_call_anthropic",
    )
    return text


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def call_openai(
    question: str,
    choices: Sequence[str],
    model_id: str = "gpt4o_mini",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    prompt_suffix: str = "",
) -> ClientResponse:
    default_temp, default_max = _v2_defaults()
    temp = default_temp if temperature is None else temperature
    max_tok = default_max if max_tokens is None else int(max_tokens)
    prompt = build_confidence_prompt(question, choices) + (prompt_suffix or "")
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    api_key = _clean_api_key(os.getenv("OPENAI_API_KEY"))
    if not api_key or api_key.startswith("your_"):
        raise LLMClientError("OPENAI_API_KEY missing in .env")
    if not api_key.isascii():
        raise LLMClientError(
            "OPENAI_API_KEY contains non-ASCII characters (often invisible "
            "copy-paste junk). Re-paste the key into .env as plain text."
        )

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temp,
        "max_tokens": max_tok,
        "messages": [
            {"role": "system", "content": SYSTEM_STYLE_NOTE},
            {"role": "user", "content": prompt},
        ],
        "logprobs": True,
        "top_logprobs": 5,
    }

    last_err: Exception | None = None
    data = None
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if _should_retry_status(resp.status_code):
                last_err = LLMClientError(f"{resp.status_code}: {resp.text[:300]}")
                _retry_sleep(attempt)
                continue
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise LLMClientError(f"OpenAI {resp.status_code}: {resp.text[:400]}")
            resp.raise_for_status()
            data = resp.json()
            break
        except LLMClientError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            _retry_sleep(attempt)
    if data is None:
        raise LLMClientError(f"OpenAI call failed after retries: {last_err}")

    choice0 = (data.get("choices") or [{}])[0]
    raw = (choice0.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    in_tok = int(usage.get("prompt_tokens") or 0)
    out_tok = int(usage.get("completion_tokens") or 0)
    cost = record_call("openai", model, in_tok, out_tok, note=f"model_id={model_id}")

    answer, _, _ = parse_response(raw)
    logprob_content = ((choice0.get("logprobs") or {}).get("content")) or []
    # Normalize to list of dicts with token/logprob/top_logprobs
    normalized = []
    for t in logprob_content:
        if not isinstance(t, dict):
            continue
        top = t.get("top_logprobs") or []
        normalized.append(
            {
                "token": t.get("token", ""),
                "logprob": t.get("logprob"),
                "top_logprobs": [
                    {"token": x.get("token", ""), "logprob": x.get("logprob")}
                    for x in top
                    if isinstance(x, dict)
                ],
            }
        )
    prob, alts, note = extract_answer_letter_logprob(normalized, answer)

    return _finalize(
        raw,
        model_id=model_id,
        backend="openai",
        internal_prob_answer=prob,
        internal_prob_alternatives=alts,
        internal_prob_note=note,
        call_cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _extract_gemini_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        raise LLMClientError(f"Gemini returned no candidates: {data}")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("thought") is True:
            continue
        t = part.get("text")
        if t:
            texts.append(t)
    if not texts:
        raise LLMClientError(f"Gemini response missing text parts: {data}")
    return "\n".join(texts).strip()


def call_gemini(
    question: str | None = None,
    choices: Sequence[str] | None = None,
    model_id: str = "gemini_flash",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    prompt_suffix: str = "",
    *,
    prompt: Optional[str] = None,
) -> ClientResponse | str:
    """
    Unified Gemini client. Prefer question+choices → ClientResponse.
    Legacy: prompt=... returns raw string (v1 collect compatibility).
    """
    # Legacy string path
    if prompt is not None and question is None:
        return _gemini_raw_string(prompt, model_name=model_name, temperature=temperature)

    if question is None or choices is None:
        raise LLMClientError("call_gemini requires question+choices (or legacy prompt=)")

    default_temp, default_max = _v2_defaults()
    temp = default_temp if temperature is None else temperature
    max_tok = default_max if max_tokens is None else int(max_tokens)
    built = build_confidence_prompt(question, choices) + (prompt_suffix or "")
    # Prefer gemini-3.1-flash-lite — gemini-2.5-flash blocked for new AI Studio accounts.
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

    raw, usage, logprob_info = _gemini_generate(
        built, model_name=model, temperature=temp, max_tokens=max_tok
    )
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    cost = record_call("gemini", model, in_tok, out_tok, note=f"model_id={model_id}")

    answer, _, _ = parse_response(raw)
    prob = logprob_info.get("internal_prob_answer")
    alts = logprob_info.get("internal_prob_alternatives")
    note = logprob_info.get("internal_prob_note")

    # If we only got aggregate avgLogprobs, do NOT treat as per-token answer prob
    if logprob_info.get("only_avg"):
        prob = None
        alts = None
        note = (
            "not available at per-token granularity for this model/API version "
            f"(avgLogprobs={logprob_info.get('avg_logprobs')!r})"
        )

    return _finalize(
        raw,
        model_id=model_id,
        backend="gemini",
        internal_prob_answer=prob,
        internal_prob_alternatives=alts,
        internal_prob_note=note,
        call_cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def _gemini_generate(
    prompt: str,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict, dict]:
    api_key = _clean_api_key(os.getenv("GEMINI_API_KEY"))
    if not api_key or api_key.startswith("your_"):
        raise LLMClientError("GEMINI_API_KEY missing in .env")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    gen_cfg: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        # Request token-level logprobs when supported
        "responseLogprobs": True,
        "logprobs": 5,
    }
    # Reduce thinking-token spend when supported
    gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
        "systemInstruction": {"parts": [{"text": SYSTEM_STYLE_NOTE}]},
    }

    last_err: Exception | None = None
    data = None
    for attempt in range(5):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
            if resp.status_code == 400 and "thinking" in resp.text.lower() and "thinkingConfig" in gen_cfg:
                gen_cfg.pop("thinkingConfig", None)
                payload["generationConfig"] = gen_cfg
                last_err = LLMClientError(f"400 retry without thinkingConfig: {resp.text[:200]}")
                continue
            # Some models reject responseLogprobs — strip and retry once
            if resp.status_code == 400 and "logprob" in resp.text.lower():
                gen_cfg.pop("responseLogprobs", None)
                gen_cfg.pop("logprobs", None)
                payload["generationConfig"] = gen_cfg
                last_err = LLMClientError(f"400 retry without logprobs: {resp.text[:200]}")
                continue
            if _should_retry_status(resp.status_code):
                last_err = LLMClientError(f"{resp.status_code}: {resp.text[:300]}")
                _retry_sleep(attempt)
                continue
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise LLMClientError(f"Gemini {resp.status_code}: {resp.text[:400]}")
            resp.raise_for_status()
            data = resp.json()
            break
        except LLMClientError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            _retry_sleep(attempt)
    if data is None:
        raise LLMClientError(f"Gemini call failed after retries: {last_err}")

    text = _extract_gemini_text(data)
    um = data.get("usageMetadata") or {}
    usage = {
        "input_tokens": um.get("promptTokenCount", 0),
        "output_tokens": um.get("candidatesTokenCount", 0),
    }

    logprob_info: dict[str, Any] = {
        "internal_prob_answer": None,
        "internal_prob_alternatives": None,
        "internal_prob_note": None,
        "only_avg": False,
        "avg_logprobs": None,
    }
    cand0 = (data.get("candidates") or [{}])[0]
    avg = cand0.get("avgLogprobs")
    if avg is not None:
        logprob_info["avg_logprobs"] = avg
        logprob_info["only_avg"] = True

    # Per-token logprobsResult (when present)
    lp_result = cand0.get("logprobsResult") or data.get("logprobsResult")
    if lp_result:
        chosen = lp_result.get("chosenCandidates") or lp_result.get("topCandidates") or []
        # Flatten to token list
        tokens = []
        if chosen and isinstance(chosen[0], dict) and "candidates" in chosen[0]:
            # topCandidates style: list of {candidates: [{token, logProbability}, ...]}
            for pos in chosen:
                cands = pos.get("candidates") or []
                if not cands:
                    continue
                top = cands[0]
                tokens.append(
                    {
                        "token": top.get("token", ""),
                        "logprob": top.get("logProbability"),
                        "top_logprobs": [
                            {"token": c.get("token", ""), "logprob": c.get("logProbability")}
                            for c in cands
                        ],
                    }
                )
        elif chosen:
            for c in chosen:
                tokens.append(
                    {
                        "token": c.get("token", ""),
                        "logprob": c.get("logProbability", c.get("logprob")),
                        "top_logprobs": [],
                    }
                )
        if tokens:
            answer, _, _ = parse_response(text)
            prob, alts, note = extract_answer_letter_logprob(tokens, answer)
            logprob_info.update(
                {
                    "internal_prob_answer": prob,
                    "internal_prob_alternatives": alts,
                    "internal_prob_note": note,
                    "only_avg": False,
                }
            )
    elif avg is not None:
        logprob_info["internal_prob_note"] = (
            "not available at per-token granularity for this model/API version"
        )
    else:
        logprob_info["internal_prob_note"] = (
            "not available at per-token granularity for this model/API version"
        )

    return text, usage, logprob_info


def _gemini_raw_string(
    prompt: str,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Legacy Gemini path used by call_model / v1 collect."""
    default_temp, default_max = _v2_defaults()
    cfg = load_experiment()
    temp = cfg["temperature"] if temperature is None else temperature
    max_tok = int(cfg.get("max_output_tokens", default_max))
    if os.getenv("USE_V2_MAX_TOKENS", "1") == "1":
        max_tok = default_max
    model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    text, usage, _ = _gemini_generate(
        prompt, model_name=model, temperature=float(temp), max_tokens=max_tok
    )
    record_call(
        "gemini",
        model,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        note="legacy_call_gemini",
    )
    return text


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

def call_ollama(
    question: str,
    choices: Sequence[str] | str | None = None,
    model_id: str = "llama3.1",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    prompt_suffix: str = "",
    *,
    prompt: Optional[str] = None,
) -> ClientResponse | str:
    """
    Unified Ollama client. Prefer question+choices → ClientResponse.

    Legacy v1 signature still supported:
      call_ollama(prompt: str, model_name: str, temperature=...) -> str
    detected when the second positional arg is a model-name string.
    """
    # Legacy: call_ollama(prompt, model_name: str)
    if isinstance(choices, str) and prompt is None:
        return _ollama_raw_string(question, model_name=choices, temperature=temperature)

    if prompt is not None:
        name = model_name or (choices if isinstance(choices, str) else None)
        if not name:
            raise LLMClientError("legacy call_ollama requires model_name")
        return _ollama_raw_string(prompt, model_name=name, temperature=temperature)

    if not isinstance(choices, (list, tuple)):
        raise LLMClientError("call_ollama requires question+choices (list of 4)")

    default_temp, default_max = _v2_defaults()
    temp = default_temp if temperature is None else temperature
    max_tok = default_max if max_tokens is None else int(max_tokens)
    built = build_confidence_prompt(question, choices) + (prompt_suffix or "")
    model = model_name or {
        "llama3.1": "llama3.1",
        "mistral": "mistral",
    }.get(model_id, model_id)

    raw, usage, logprob_info = _ollama_generate(
        built, model_name=model, temperature=temp, max_tokens=max_tok
    )
    in_tok = int(usage.get("input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    cost = record_call("ollama", model, in_tok, out_tok, note=f"model_id={model_id}")

    return _finalize(
        raw,
        model_id=model_id,
        backend="ollama",
        internal_prob_answer=logprob_info.get("internal_prob_answer"),
        internal_prob_alternatives=logprob_info.get("internal_prob_alternatives"),
        internal_prob_note=logprob_info.get("internal_prob_note"),
        call_cost_usd=cost,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )


def _ollama_generate(
    prompt: str,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict, dict]:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/generate"
    payload: dict[str, Any] = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 5,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    last_err: Exception | None = None
    data = None
    for attempt in range(5):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            # If logprobs rejected, retry without
            if resp.status_code >= 400 and "logprob" in (resp.text or "").lower():
                payload.pop("logprobs", None)
                payload.pop("top_logprobs", None)
                last_err = LLMClientError(f"retry without logprobs: {resp.text[:200]}")
                continue
            if _should_retry_status(resp.status_code):
                last_err = LLMClientError(f"{resp.status_code}: {resp.text[:300]}")
                _retry_sleep(attempt)
                continue
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise LLMClientError(f"Ollama {resp.status_code}: {resp.text[:400]}")
            resp.raise_for_status()
            data = resp.json()
            break
        except LLMClientError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            _retry_sleep(attempt)
    if data is None:
        raise LLMClientError(
            f"Ollama call failed for model '{model_name}'. Is Ollama running? Last error: {last_err}"
        )

    if "response" not in data:
        raise LLMClientError(f"Unexpected Ollama payload: {data}")
    text = data["response"]
    usage = {
        "input_tokens": data.get("prompt_eval_count") or 0,
        "output_tokens": data.get("eval_count") or 0,
    }

    logprob_info: dict[str, Any] = {
        "internal_prob_answer": None,
        "internal_prob_alternatives": None,
        "internal_prob_note": None,
    }
    lp = data.get("logprobs")
    if not lp:
        logprob_info["internal_prob_note"] = (
            "Ollama logprobs requested but not returned (null/absent). "
            "internal_prob_answer set to None."
        )
        return text, usage, logprob_info

    # Expected: list of {token, logprob, top_logprobs: [...]}
    tokens = []
    for t in lp:
        if not isinstance(t, dict):
            continue
        top = t.get("top_logprobs") or []
        tokens.append(
            {
                "token": t.get("token", ""),
                "logprob": t.get("logprob"),
                "top_logprobs": [
                    {"token": x.get("token", ""), "logprob": x.get("logprob")}
                    for x in top
                    if isinstance(x, dict)
                ],
            }
        )
    answer, _, _ = parse_response(text)
    prob, alts, note = extract_answer_letter_logprob(tokens, answer)
    logprob_info.update(
        {
            "internal_prob_answer": prob,
            "internal_prob_alternatives": alts,
            "internal_prob_note": note,
        }
    )
    return text, usage, logprob_info


def _ollama_raw_string(
    prompt: str,
    model_name: str,
    temperature: Optional[float] = None,
) -> str:
    default_temp, default_max = _v2_defaults()
    cfg = load_experiment()
    temp = cfg["temperature"] if temperature is None else temperature
    max_tok = default_max if os.getenv("USE_V2_MAX_TOKENS", "1") == "1" else int(
        cfg.get("max_output_tokens", default_max)
    )
    text, usage, _ = _ollama_generate(
        prompt, model_name=model_name, temperature=float(temp), max_tokens=max_tok
    )
    record_call(
        "ollama",
        model_name,
        int(usage.get("input_tokens") or 0),
        int(usage.get("output_tokens") or 0),
        note="legacy_call_ollama",
    )
    return text


# ---------------------------------------------------------------------------
# Dispatcher (v1 collect compatibility)
# ---------------------------------------------------------------------------

def call_model(backend: str, model_name: str, prompt: str) -> str:
    if backend == "gemini":
        out = call_gemini(prompt=prompt, model_name=model_name)
        assert isinstance(out, str)
        return out
    if backend == "anthropic":
        return call_anthropic(prompt, model_name=model_name)
    if backend == "ollama":
        return _ollama_raw_string(prompt, model_name=model_name)
    if backend == "openai":
        # Minimal legacy bridge: not used by v1 collect
        raise LLMClientError("Use call_openai(question, choices, ...) for OpenAI")
    raise LLMClientError(f"Unknown backend: {backend}")
