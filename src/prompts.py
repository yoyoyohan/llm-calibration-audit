"""Prompt construction for verbal confidence elicitation."""
from __future__ import annotations

from typing import Sequence

SYSTEM_STYLE_NOTE = (
    "You are answering a multiple-choice academic question. "
    "Follow the output format exactly."
)


def build_confidence_prompt(question: str, choices: Sequence[str]) -> str:
    """
    Build the exact prompt used in the experiment.

    choices must be length 4 in order A,B,C,D.
    """
    if len(choices) != 4:
        raise ValueError(f"Expected 4 choices, got {len(choices)}")

    letters = ["A", "B", "C", "D"]
    choices_block = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, choices))

    return f"""Answer the following multiple-choice question. Then rate your confidence.

Question: {question}

{choices_block}

Respond in EXACTLY this format with no other text:
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]

Example of a valid reply:
ANSWER: B
CONFIDENCE: 70
"""


# Kept for documentation / paper appendix — identical to build_confidence_prompt output template.
PROMPT_TEMPLATE_FOR_PAPER = """Answer the following multiple-choice question. Then rate your confidence.

Question: {question}

A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Respond in EXACTLY this format with no other text:
ANSWER: [A/B/C/D]
CONFIDENCE: [integer 0-100]
"""
