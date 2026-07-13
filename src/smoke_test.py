"""
Quick connectivity smoke test (does not need question bank).

Usage:
  python src/smoke_test.py --ollama
  python src/smoke_test.py --gemini
  python src/smoke_test.py --both
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clients import call_gemini, call_ollama
from parse import parse_response
from prompts import build_confidence_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama", action="store_true")
    parser.add_argument("--gemini", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--ollama-model", default="llama3.1")
    args = parser.parse_args()

    if not (args.ollama or args.gemini or args.both):
        args.both = True

    prompt = build_confidence_prompt(
        "What is 2 + 2?",
        ["3", "4", "5", "22"],
    )
    print("PROMPT:\n", prompt)

    if args.ollama or args.both:
        print("\n--- Ollama ---")
        text = call_ollama(prompt, args.ollama_model)
        print("RAW:", text)
        print("PARSED:", parse_response(text))

    if args.gemini or args.both:
        print("\n--- Gemini ---")
        text = call_gemini(prompt)
        print("RAW:", text)
        print("PARSED:", parse_response(text))


if __name__ == "__main__":
    main()
