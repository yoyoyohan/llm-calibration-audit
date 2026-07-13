"""Parse model responses into answer letter + confidence."""
from __future__ import annotations

import re
from typing import Optional, Tuple


ANSWER_RE = re.compile(r"ANSWER:\s*([A-Da-d])", re.IGNORECASE)
CONF_RE = re.compile(r"CONFIDENCE:\s*(\d{1,3})", re.IGNORECASE)


def parse_response(response_text: Optional[str]) -> Tuple[Optional[str], Optional[int], bool]:
    """
    Returns (answer_letter, confidence_0_100, parse_ok).

    parse_ok is True only when BOTH fields are found and valid.
    """
    if not response_text or not str(response_text).strip():
        return None, None, False

    text = str(response_text).strip()
    answer_match = ANSWER_RE.search(text)
    conf_match = CONF_RE.search(text)

    answer = answer_match.group(1).upper() if answer_match else None
    confidence = int(conf_match.group(1)) if conf_match else None

    if confidence is not None:
        # Accept only [0, 100]; out-of-range => parse failure for confidence.
        if confidence < 0 or confidence > 100:
            confidence = None

    parse_ok = answer is not None and confidence is not None
    return answer, confidence, parse_ok


def letter_to_index(letter: Optional[str]) -> Optional[int]:
    if letter is None:
        return None
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(letter.upper())


def index_to_letter(idx: int) -> str:
    return ["A", "B", "C", "D"][idx]
