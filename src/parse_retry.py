"""
Parse-success guarantee: retry API calls until ANSWER+CONFIDENCE parse, or exhaust attempts.

Primary attempt uses the locked v1/v2 prompt unchanged.
Retry attempts append a short repair footer (does not rewrite the locked template).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Sequence

from clients import ClientResponse

# Appended only on parse-failure retries — primary attempt stays byte-identical to locked prompt.
PARSE_REPAIR_SUFFIX = """

CRITICAL FORMAT REPAIR: Your previous reply was invalid or incomplete.
Reply with EXACTLY two lines and no other text (no reasoning, no refusal):
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]
You must choose one of A, B, C, or D even if unsure.
"""

DEFAULT_MAX_PARSE_ATTEMPTS = 3


def call_with_parse_retry(
    client_fn: Callable[..., ClientResponse],
    question: str,
    choices: Sequence[str],
    *,
    max_attempts: Optional[int] = None,
    **kwargs: Any,
) -> ClientResponse:
    """
    Call a unified client until parse_ok is True, or return the last response.

    Attempt 1: locked primary prompt only.
    Attempts 2+: same call with prompt_suffix=PARSE_REPAIR_SUFFIX.
    """
    n = int(max_attempts if max_attempts is not None else os.getenv("PARSE_MAX_ATTEMPTS", DEFAULT_MAX_PARSE_ATTEMPTS))
    n = max(1, n)
    last: Optional[ClientResponse] = None
    total_cost = 0.0
    total_in = 0
    total_out = 0

    for attempt in range(1, n + 1):
        call_kwargs = dict(kwargs)
        if attempt > 1:
            call_kwargs["prompt_suffix"] = PARSE_REPAIR_SUFFIX
        resp = client_fn(question, choices, **call_kwargs)
        if not isinstance(resp, ClientResponse):
            raise TypeError(f"Expected ClientResponse, got {type(resp)}")

        total_cost += float(resp.call_cost_usd or 0.0)
        total_in += int(resp.input_tokens or 0)
        total_out += int(resp.output_tokens or 0)
        # Annotate attempt metadata on optional fields (extras allowed by interface)
        resp.call_cost_usd = total_cost
        resp.input_tokens = total_in
        resp.output_tokens = total_out
        note_bits = []
        if resp.internal_prob_note:
            note_bits.append(resp.internal_prob_note)
        note_bits.append(f"parse_attempts={attempt}/{n}")
        resp.internal_prob_note = " | ".join(note_bits)
        last = resp
        if resp.parse_ok:
            return resp

    assert last is not None
    return last
