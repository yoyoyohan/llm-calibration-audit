"""LLM backends: Gemini API + Ollama local."""
from __future__ import annotations

import os
import time
from typing import Optional

import requests
from dotenv import load_dotenv

from paths import ROOT, load_experiment

load_dotenv(ROOT / ".env")


class LLMClientError(RuntimeError):
    pass


def call_gemini(prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None) -> str:
    cfg = load_experiment()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_gemini"):
        raise LLMClientError(
            "GEMINI_API_KEY missing. Copy .env.example to .env and add your key from Google AI Studio."
        )

    model = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    temp = cfg["temperature"] if temperature is None else temperature
    max_tokens = cfg.get("max_output_tokens", 80)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temp,
            "maxOutputTokens": max_tokens,
        },
    }

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(8 * (attempt + 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise LLMClientError(f"Gemini call failed after retries: {last_err}")


def call_ollama(prompt: str, model_name: str, temperature: Optional[float] = None) -> str:
    cfg = load_experiment()
    temp = cfg["temperature"] if temperature is None else temperature
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = f"{host}/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temp},
    }

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=180)
            resp.raise_for_status()
            data = resp.json()
            if "response" not in data:
                raise LLMClientError(f"Unexpected Ollama payload: {data}")
            return data["response"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise LLMClientError(
        f"Ollama call failed for model '{model_name}'. Is Ollama running? Last error: {last_err}"
    )


def call_model(backend: str, model_name: str, prompt: str) -> str:
    if backend == "gemini":
        return call_gemini(prompt, model_name=model_name)
    if backend == "ollama":
        return call_ollama(prompt, model_name=model_name)
    raise LLMClientError(f"Unknown backend: {backend}")
