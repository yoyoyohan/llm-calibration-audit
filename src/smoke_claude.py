"""Quick Claude connectivity + parse check."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clients import call_anthropic
from parse import parse_response
from prompts import build_confidence_prompt


def main() -> None:
    prompt = build_confidence_prompt("What is 2 + 2?", ["3", "4", "5", "22"])
    print("Calling Claude Haiku...")
    raw = call_anthropic(prompt)
    print("RAW:", raw)
    print("PARSED:", parse_response(raw))


if __name__ == "__main__":
    main()
