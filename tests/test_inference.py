"""Offline checks for explicit local sampling controls."""

from __future__ import annotations

import contextlib
import types
import unittest
from unittest.mock import patch

from training.inference import generate_from_messages


class _Encoded(dict):
    def to(self, device: object) -> "_Encoded":
        return self


class _Tokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def __call__(self, rendered, **kwargs):
        return _Encoded(input_ids=types.SimpleNamespace(shape=(1, 2)))

    def decode(self, tokens, **kwargs):
        return "completion"


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.kwargs = {}

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return [[10, 11, 12]]


class SamplingTests(unittest.TestCase):
    def test_sampling_explicitly_overrides_model_top_p_and_top_k_defaults(self) -> None:
        model = _Model()
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.dict("sys.modules", {"torch": fake_torch}):
            result = generate_from_messages(
                model,
                _Tokenizer(),
                [{"role": "user", "content": "diagnose"}],
                temperature=1.8,
                top_p=1.0,
                top_k=0,
                num_return_sequences=8,
            )

        self.assertEqual(result, ["completion"])
        self.assertEqual(model.kwargs["temperature"], 1.8)
        self.assertEqual(model.kwargs["top_p"], 1.0)
        self.assertEqual(model.kwargs["top_k"], 0)
        self.assertTrue(model.kwargs["do_sample"])

    def test_sampling_bounds_are_validated_before_model_use(self) -> None:
        for kwargs in ({"top_p": 0.0}, {"top_p": 1.1}, {"top_k": -1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    generate_from_messages(
                        _Model(),
                        _Tokenizer(),
                        [{"role": "user", "content": "diagnose"}],
                        **kwargs,
                    )


if __name__ == "__main__":
    unittest.main()
