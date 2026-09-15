from __future__ import annotations

import unittest

from training.run_14b import apply_14b_defaults, exact_resolution_rate


class Run14BTests(unittest.TestCase):
    def test_exact_resolution_rate_uses_resolved_over_total(self) -> None:
        rate = exact_resolution_rate(
            {"summary": {"resolved_episodes": 16, "total_episodes": 832}}
        )
        self.assertAlmostEqual(rate, 16 / 832)

    def test_baseline_gate_threshold_is_ten_percent(self) -> None:
        self.assertLess(
            exact_resolution_rate(
                {"summary": {"resolved_episodes": 83, "total_episodes": 832}}
            ),
            0.10,
        )
        self.assertGreaterEqual(
            exact_resolution_rate(
                {"summary": {"resolved_episodes": 84, "total_episodes": 832}}
            ),
            0.10,
        )

    def test_apply_14b_defaults_sets_qlora_and_docker_requirements(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            apply_14b_defaults()
            self.assertEqual(os.environ["CRASHDIAG_BASE_MODEL"], "Qwen/Qwen2.5-14B-Instruct")
            self.assertEqual(os.environ["CRASHDIAG_MODEL_SLUG"], "qwen2.5_14b")
            self.assertEqual(os.environ["CRASHDIAG_REQUIRE_SANDBOX_BACKEND"], "docker")
            self.assertEqual(os.environ["CRASHDIAG_GRPO_LOAD_IN_4BIT"], "1")
            self.assertEqual(os.environ["CRASHDIAG_GRPO_BATCH_SIZE"], "1")
            self.assertEqual(os.environ["CRASHDIAG_GRPO_GRAD_ACCUM"], "8")


if __name__ == "__main__":
    unittest.main()
