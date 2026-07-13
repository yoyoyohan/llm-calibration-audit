"""
Verify Python imports, Ollama, and Gemini before Day 1 collection.

Usage:
  python src/setup_verify.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import ROOT

print(f"Python: {sys.version}")
print(f"Repo: {ROOT}")

modules = [
    "requests",
    "pandas",
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "statsmodels",
    "sklearn",
    "datasets",
    "yaml",
    "dotenv",
    "tqdm",
]
for name in modules:
    try:
        __import__(name if name != "sklearn" else "sklearn")
        print(f"{name}: OK")
    except ImportError:
        print(f"{name}: MISSING — run: pip install -r requirements.txt")

print("\nTesting Ollama (llama3.1)...")
try:
    import requests

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1",
            "prompt": "Say the word WORKING and nothing else.",
            "stream": False,
        },
        timeout=120,
    )
    if r.status_code == 200:
        print(f"Ollama llama3.1: OK — {r.json().get('response', '')[:80]!r}")
    else:
        print(f"Ollama: ERROR status {r.status_code} {r.text[:200]}")
except Exception as e:  # noqa: BLE001
    print(f"Ollama: ERROR — {e}")
    print("Is the Ollama app running? Try: ollama pull llama3.1")

print("\nTesting Gemini...")
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
key = os.getenv("GEMINI_API_KEY", "")
if not key or key.startswith("your_gemini"):
    print("Gemini: NOT TESTED — put GEMINI_API_KEY in .env")
else:
    try:
        from clients import call_gemini

        text = call_gemini("Say the word WORKING and nothing else.")
        print(f"Gemini: OK — {text[:80]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"Gemini: ERROR — {e}")

print("\nSetup verification complete.")
