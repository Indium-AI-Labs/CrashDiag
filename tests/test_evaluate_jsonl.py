from __future__ import annotations

import unittest
from unittest.mock import patch

from training.evaluate_jsonl import evaluate_rows


class ExactEvaluationProgressTests(unittest.TestCase):
    def test_reports_live_cumulative_progress_for_every_row(self) -> None:
        rows = [
            {
                "fault_name": "oom_kill",
                "sample_seed": index,
                "scenario_schema_version": 1,
                "prompt": [{"role": "user", "content": f"row {index}"}],
            }
            for index in range(2)
        ]
        messages: list[str] = []

        def reward(completions, *, log_extra, **kwargs):
            del completions, kwargs
            log_extra("crashdiag_action", ["restart_app"])
            log_extra("crashdiag_resolved", [True])
            log_extra("crashdiag_backend_error", [False])
            log_extra("crashdiag_strict_json", [True])
            return [1.0]

        with patch("training.evaluate_jsonl.mechanical_reward", side_effect=reward):
            report = evaluate_rows(rows, lambda prompt: "completion", progress=messages.append)

        self.assertEqual(report["summary"]["resolved_episodes"], 2)
        self.assertEqual(len(messages), 3)
        self.assertIn("Starting evaluation: 2 rows", messages[0])
        self.assertIn("[1/2]", messages[1])
        self.assertIn("resolved=1", messages[1])
        self.assertIn("[2/2]", messages[2])
        self.assertIn("success_rate=100.0%", messages[2])
        self.assertIn("backend_errors=0", messages[2])


if __name__ == "__main__":
    unittest.main()
