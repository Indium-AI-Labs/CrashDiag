"""Behavior tests for the HUD-facing CrashDiag task adapter."""

from __future__ import annotations

import unittest

from crashdiag.hud_adapter import create_hud_episode
from training.hard_scenarios import hard_expert_workflow, hard_sample_seed


class HudAdapterTests(unittest.TestCase):
    def test_expert_workflow_gets_full_mechanical_reward(self) -> None:
        fault_name = "memory_pressure_high"
        episode = create_hud_episode(
            fault_name,
            hard_sample_seed(42, fault_name, 1),
            "noisy",
        )
        self.assertEqual(episode.workflow.resolved_subfault_count(episode.sandbox), 0)
        import json

        answer = json.dumps(hard_expert_workflow(fault_name), separators=(",", ":"))
        self.assertEqual(episode.grade(answer), 1.0)

    def test_non_strict_or_repeated_submission_fails_closed(self) -> None:
        fault_name = "oom_kill"
        episode = create_hud_episode(
            fault_name,
            hard_sample_seed(42, fault_name, 0),
            "redacted",
        )
        self.assertEqual(episode.grade("```json\n{}\n```"), 0.0)
        with self.assertRaisesRegex(RuntimeError, "already been graded"):
            episode.grade('{"actions":[{"action":"restart_app","parameters":{}}]}')


if __name__ == "__main__":
    unittest.main()
