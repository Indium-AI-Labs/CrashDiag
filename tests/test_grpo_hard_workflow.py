"""Offline tests for calibration, exact evaluation, and promotion gates."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from training.artifacts import ArtifactError
from training.calibrate_grpo import (
    build_rollout_replay_generator,
    calibrate,
    main as calibrate_main,
    select_calibration_rows,
    summarize_temperature,
)
from training.common import FAULT_NAMES
from training.evaluate_jsonl import summarize_results
from training.grpo_gates import promotion_gate, smoke_gate
from training.hard_scenarios import HARD_SCENARIO_PROFILES, generate_hard_records


class CalibrationTests(unittest.TestCase):
    def test_signed_rollout_replay_ignores_recorded_rewards(self) -> None:
        rows = generate_hard_records(
            samples_per_fault=3, seed=17, start_variation=0, split="train"
        )
        selected = select_calibration_rows(rows, prompts_per_fault_profile=1)
        rollouts = []
        for prompt_index, row in enumerate(selected):
            for generation_index, completion in enumerate(("first", "second")):
                rollouts.append(
                    {
                        "temperature": 1.6,
                        "prompt_index": prompt_index,
                        "generation_index": generation_index,
                        "fault_name": row["fault_name"],
                        "scenario_profile": row["scenario_profile"],
                        "sample_seed": row["sample_seed"],
                        "completion": completion,
                        "reward": 999.0,
                    }
                )
        replay = build_rollout_replay_generator(
            rows,
            rollouts,
            {
                "num_generations": 2,
                "prompts_per_fault_profile": 1,
                "sampling": {
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_new_tokens": 48,
                },
            },
            temperatures=[1.6],
            num_generations=2,
            prompts_per_fault_profile=1,
            top_p=0.9,
            top_k=50,
            max_new_tokens=48,
        )

        self.assertEqual(replay(selected[0]["prompt"], 1.6, 2), ["first", "second"])

    def test_artifact_collision_returns_status_instead_of_raising_system_exit(self) -> None:
        with (
            patch(
                "training.calibrate_grpo.uploader_from_args",
                side_effect=ArtifactError("stage already complete"),
            ),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
        ):
            status = calibrate_main([])

        self.assertEqual(status, 2)
        self.assertIn("stage already complete", stderr.getvalue())

    def test_calibration_slice_is_exactly_stratified(self) -> None:
        rows = generate_hard_records(
            samples_per_fault=9, seed=12, start_variation=0, split="train"
        )
        selected = select_calibration_rows(rows, prompts_per_fault_profile=2)
        self.assertEqual(len(selected), 36)
        cells = {(row["fault_name"], row["scenario_profile"]) for row in selected}
        self.assertEqual(len(cells), len(FAULT_NAMES) * len(HARD_SCENARIO_PROFILES))

    def test_variance_gate_requires_mixed_groups_across_four_faults(self) -> None:
        rollouts = []
        prompt_index = 0
        for fault_index, fault in enumerate(FAULT_NAMES):
            for _ in range(2):
                for generation in range(8):
                    reward = 1.0 if fault_index < 4 and generation < 4 else 0.0
                    if fault_index >= 4:
                        reward = 1.0
                    rollouts.append(
                        {
                            "prompt_index": prompt_index,
                            "fault_name": fault,
                            "reward": reward,
                            "strict_json": True,
                            "backend_error": False,
                        }
                    )
                prompt_index += 1
        result = summarize_temperature(rollouts, expected_group_size=8)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["mixed_fault_families"]), 4)
        self.assertEqual(
            set(result["positive_reward_fault_families"]),
            set(FAULT_NAMES),
        )

    def test_variance_gate_rejects_a_fault_family_with_no_positive_rollout(self) -> None:
        rollouts = []
        prompt_index = 0
        for fault_index, fault in enumerate(FAULT_NAMES):
            for _ in range(2):
                for generation in range(8):
                    reward = 1.0 if generation < 4 else 0.0
                    if fault_index == len(FAULT_NAMES) - 1:
                        reward = 0.0
                    rollouts.append(
                        {
                            "prompt_index": prompt_index,
                            "fault_name": fault,
                            "reward": reward,
                            "strict_json": True,
                            "backend_error": False,
                        }
                    )
                prompt_index += 1

        result = summarize_temperature(rollouts, expected_group_size=8)

        self.assertFalse(result["passed"])
        self.assertFalse(result["gates"]["positive_reward_all_fault_families"])

    def test_calibration_scores_each_generation_concurrently(self) -> None:
        rows = generate_hard_records(
            samples_per_fault=6, seed=31, start_variation=0, split="train"
        )
        lock = threading.Lock()
        active = 0
        maximum_active = 0

        def fake_reward(completions, *, log_extra, **kwargs):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.002)
            with lock:
                active -= 1
            reward = 1.0 if completions[0] == "good" else 0.0
            log_extra("crashdiag_action", ["restart_app"])
            log_extra("crashdiag_resolved", [reward == 1.0])
            log_extra("crashdiag_backend_error", [False])
            log_extra("crashdiag_strict_json", [True])
            return [reward]

        messages: list[str] = []
        with patch("training.calibrate_grpo.mechanical_reward", side_effect=fake_reward):
            report, _ = calibrate(
                rows,
                lambda prompt, temperature, count: ["good", "bad"],
                temperatures=[0.9],
                num_generations=2,
                prompts_per_fault_profile=1,
                reward_workers=2,
                progress=messages.append,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(maximum_active, 2)
        self.assertTrue(any("18/18 prompt groups" in message for message in messages))


class EvaluationAndGateTests(unittest.TestCase):
    def test_summary_uses_recorded_mechanical_rewards(self) -> None:
        results = [
            {
                "fault_name": "oom_kill",
                "scenario_profile": "redacted",
                "reward": reward,
                "strict_json": True,
                "backend_error": False,
            }
            for reward in (1.0, 0.0)
        ]
        report = summarize_results(results)
        self.assertEqual(report["summary"]["success_rate"], 0.5)
        self.assertEqual(report["per_fault"]["oom_kill"]["resolved"], 1)

    def test_smoke_and_promotion_gates_pass_only_with_real_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_bytes = b"parent"
            adapter = root / "adapter_model.safetensors"
            adapter.write_bytes(b"updated")
            parent = root / "parent.json"
            parent.write_text(
                json.dumps({"adapter_sha256": hashlib.sha256(parent_bytes).hexdigest()})
            )
            state = root / "trainer_state.json"
            state.write_text(
                json.dumps(
                    {
                        "log_history": [
                            {
                                "step": 20,
                                "reward_std": 0.4,
                                "grad_norm": 1.2,
                                "crashdiag/success_rate": 0.5,
                                "crashdiag/backend_error_rate": 0.0,
                            }
                        ]
                    }
                )
            )
            self.assertTrue(smoke_gate(state, adapter, parent)["passed"])

            def report(path: Path, total: int, rate: float, per_fault: float) -> None:
                path.write_text(
                    json.dumps(
                        {
                            "summary": {
                                "total_episodes": total,
                                "success_rate": rate,
                                "backend_error_rate": 0.0,
                            },
                            "per_fault": {
                                name: {"success_rate": per_fault} for name in FAULT_NAMES
                            },
                        }
                    )
                )

            hard = root / "hard.json"
            regression = root / "regression.json"
            report(hard, 192, 0.8, 0.6)
            report(regression, 96, 0.98, 0.98)
            self.assertTrue(promotion_gate(hard, regression)["passed"])
            report(hard, 192, 0.8, 0.4)
            self.assertFalse(promotion_gate(hard, regression)["passed"])


if __name__ == "__main__":
    unittest.main()
