from __future__ import annotations

import unittest

from crashdiag.agents import BlueAgent, parse_action


class BlueAgentParserTests(unittest.TestCase):
    def test_accepts_valid_action(self) -> None:
        self.assertEqual(
            parse_action('{"action":"clear_disk","parameters":{}}'),
            {"action": "clear_disk", "parameters": {}},
        )

    def test_recovers_json_from_code_fence(self) -> None:
        self.assertEqual(
            parse_action('```json\n{"action":"restart_app"}\n```'),
            {"action": "restart_app", "parameters": {}},
        )

    def test_invalid_output_falls_back_to_wait(self) -> None:
        fallback = {"action": "wait_and_observe", "parameters": {}}
        cyclic_parameters: dict[str, object] = {}
        cyclic_parameters["self"] = cyclic_parameters
        for invalid in (
            "not json",
            '{"action":"run_arbitrary_shell"}',
            '{"action":"restart_app","parameters":[]}',
            '{{"action":"restart_app"}',
            'I suggest {"action":"restart_app"}',
            {"action": "restart_app", "parameters": cyclic_parameters},
            "[" * 2000 + "0" + "]" * 2000,
            None,
        ):
            with self.subTest(invalid=invalid):
                self.assertEqual(parse_action(invalid), fallback)

    def test_constructor_rejects_invalid_inference_settings(self) -> None:
        for kwargs in (
            {"model": ""},
            {"model": "demo", "base_url": ""},
            {"model": "demo", "timeout": float("nan")},
            {"model": "demo", "temperature": float("inf")},
            {"model": "demo", "max_tokens": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BlueAgent(**kwargs)


if __name__ == "__main__":
    unittest.main()
