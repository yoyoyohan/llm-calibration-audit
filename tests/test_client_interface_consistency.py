"""
Assert every unified client returns the exact same ClientResponse field set.

Uses mocks — no live API calls, no cost.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clients import (  # noqa: E402
    REQUIRED_RESPONSE_FIELDS,
    ClientResponse,
    call_claude,
    call_gemini,
    call_ollama,
    call_openai,
)


def _assert_fields(resp: ClientResponse, test: unittest.TestCase) -> None:
    d = resp.to_dict()
    missing = [f for f in REQUIRED_RESPONSE_FIELDS if f not in d]
    # Allow extras (internal_prob_note, call_cost_usd, …) but require exact core set present
    test.assertEqual(missing, [], f"Missing required fields: {missing}")
    for f in REQUIRED_RESPONSE_FIELDS:
        test.assertIn(f, d)
    # Core field count check: every required field is present (extras OK)
    test.assertTrue(set(REQUIRED_RESPONSE_FIELDS).issubset(set(d.keys())))


class TestClientInterfaceConsistency(unittest.TestCase):
    question = "What is 2+2?"
    choices = ["3", "4", "5", "22"]

    def test_claude_fields(self) -> None:
        fake = (
            "ANSWER: B\nCONFIDENCE: 70",
            {"input_tokens": 10, "output_tokens": 5},
        )
        with patch("clients._anthropic_raw", return_value=fake), patch(
            "clients.record_call", return_value=0.0
        ):
            resp = call_claude(self.question, self.choices, model_id="claude_haiku")
        self.assertIsInstance(resp, ClientResponse)
        _assert_fields(resp, self)
        self.assertIsNone(resp.internal_prob_answer)
        self.assertEqual(resp.backend, "anthropic")
        self.assertEqual(resp.config_version, "v2")

    def test_openai_fields(self) -> None:
        fake_resp = {
            "choices": [
                {
                    "message": {"content": "ANSWER: B\nCONFIDENCE: 80"},
                    "logprobs": {
                        "content": [
                            {"token": "ANSWER", "logprob": -0.1, "top_logprobs": []},
                            {"token": ":", "logprob": -0.01, "top_logprobs": []},
                            {
                                "token": " B",
                                "logprob": -0.2,
                                "top_logprobs": [
                                    {"token": " B", "logprob": -0.2},
                                    {"token": " A", "logprob": -1.5},
                                ],
                            },
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        }

        class FakeHTTP:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return fake_resp

        with patch("clients.requests.post", return_value=FakeHTTP()), patch(
            "clients.os.getenv",
            side_effect=lambda k, default=None: {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_MODEL": "gpt-4o-mini",
            }.get(k, default),
        ), patch("clients.record_call", return_value=0.0001):
            resp = call_openai(self.question, self.choices, model_id="gpt4o_mini")
        self.assertIsInstance(resp, ClientResponse)
        _assert_fields(resp, self)
        self.assertEqual(resp.backend, "openai")

    def test_gemini_fields(self) -> None:
        fake = (
            "ANSWER: B\nCONFIDENCE: 75",
            {"input_tokens": 15, "output_tokens": 6},
            {
                "internal_prob_answer": None,
                "internal_prob_alternatives": None,
                "internal_prob_note": "not available at per-token granularity for this model/API version",
                "only_avg": True,
                "avg_logprobs": -0.3,
            },
        )
        with patch("clients._gemini_generate", return_value=fake), patch(
            "clients.record_call", return_value=0.0
        ):
            resp = call_gemini(self.question, self.choices, model_id="gemini_flash")
        self.assertIsInstance(resp, ClientResponse)
        _assert_fields(resp, self)
        self.assertEqual(resp.backend, "gemini")

    def test_ollama_fields(self) -> None:
        fake = (
            "ANSWER: B\nCONFIDENCE: 60",
            {"input_tokens": 12, "output_tokens": 7},
            {
                "internal_prob_answer": None,
                "internal_prob_alternatives": None,
                "internal_prob_note": "Ollama logprobs requested but not returned",
            },
        )
        with patch("clients._ollama_generate", return_value=fake), patch(
            "clients.record_call", return_value=0.0
        ):
            resp = call_ollama(self.question, self.choices, model_id="llama3.1")
        self.assertIsInstance(resp, ClientResponse)
        _assert_fields(resp, self)
        self.assertEqual(resp.backend, "ollama")

    def test_all_share_exact_required_field_set(self) -> None:
        """Every client returns dicts whose required keys are identical."""
        responses = []

        with patch(
            "clients._anthropic_raw",
            return_value=("ANSWER: A\nCONFIDENCE: 50", {"input_tokens": 1, "output_tokens": 1}),
        ), patch("clients.record_call", return_value=0.0):
            responses.append(call_claude(self.question, self.choices))

        class FakeHTTP:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {"content": "ANSWER: A\nCONFIDENCE: 50"},
                            "logprobs": {"content": []},
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }

        with patch("clients.requests.post", return_value=FakeHTTP()), patch(
            "clients.os.getenv",
            side_effect=lambda k, default=None: {
                "OPENAI_API_KEY": "sk-test",
            }.get(k, default),
        ), patch("clients.record_call", return_value=0.0):
            responses.append(call_openai(self.question, self.choices))

        with patch(
            "clients._gemini_generate",
            return_value=(
                "ANSWER: A\nCONFIDENCE: 50",
                {"input_tokens": 1, "output_tokens": 1},
                {
                    "internal_prob_answer": None,
                    "internal_prob_alternatives": None,
                    "internal_prob_note": "n/a",
                    "only_avg": False,
                },
            ),
        ), patch("clients.record_call", return_value=0.0):
            responses.append(call_gemini(self.question, self.choices))

        with patch(
            "clients._ollama_generate",
            return_value=(
                "ANSWER: A\nCONFIDENCE: 50",
                {"input_tokens": 1, "output_tokens": 1},
                {
                    "internal_prob_answer": None,
                    "internal_prob_alternatives": None,
                    "internal_prob_note": "n/a",
                },
            ),
        ), patch("clients.record_call", return_value=0.0):
            responses.append(call_ollama(self.question, self.choices))

        key_sets = [set(REQUIRED_RESPONSE_FIELDS) & set(r.to_dict().keys()) for r in responses]
        self.assertTrue(all(k == set(REQUIRED_RESPONSE_FIELDS) for k in key_sets))


if __name__ == "__main__":
    unittest.main()
