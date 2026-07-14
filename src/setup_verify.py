"""
Verify Python imports, Ollama, and Claude Haiku before collection.

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

print("\nTesting Claude Haiku...")
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
key = os.getenv("ANTHROPIC_API_KEY", "")
if not key or key.startswith("your_anthropic"):
    print("Claude: NOT TESTED — put ANTHROPIC_API_KEY in .env")
else:
    try:
        from clients import call_anthropic
        from paths import load_experiment

        models = load_experiment()["models"]
        claude = next((m for m in models if m.get("backend") == "anthropic"), None)
        model_name = claude["model_name"] if claude else None
        text = call_anthropic("Say the word WORKING and nothing else.", model_name=model_name)
        print(f"Claude ({model_name or 'default'}): OK — {text[:80]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"Claude: ERROR — {e}")

print("\nSetup verification complete.")
