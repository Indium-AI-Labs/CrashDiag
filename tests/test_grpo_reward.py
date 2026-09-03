from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from crashdiag.faults.workflows import WORKFLOWS
from training.common import observation_messages
from training.grpo import (
    _validate_positive,
    build_parser,
    completion_text,
    configure_reward_backend,
    mechanical_reward,
)
from training.generate_dataset import generate_records, prepare_scenario, sample_seed
from training.hard_scenarios import (
    HARD_SCENARIO_PROFILES,
    build_v6_sample,
    hard_expert_workflow,
)


def _prompt(fault_name: str, scenario_seed: int) -> list[dict[str, str]]:
    _, sandbox, _ = prepare_scenario(fault_name, scenario_seed)
    return observation_messages(sandbox.observe())


def _workflow_completion(fault_name: str) -> str:
    return json.dumps(hard_expert_workflow(fault_name))


class MechanicalRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_reward_backend(sandbox_url=None)

    def test_correct_workflows_are_rewarded_from_sandbox_state(self) -> None:
        completions = [
            _workflow_completion("oom_kill"),
            _workflow_completion("disk_full"),
        ]

        seeds = [
            sample_seed(73, "oom_kill", 2),
            sample_seed(73, "disk_full", 7),
        ]
        rewards = mechanical_reward(
            completions,
            fault_name=["oom_kill", "disk_full"],
            sample_seed=seeds,
            prompts=[_prompt("oom_kill", seeds[0]), _prompt("disk_full", seeds[1])],
            subfault_count=[
                WORKFLOWS["oom_kill"].subfault_count,
                WORKFLOWS["disk_full"].subfault_count,
            ],
        )

        self.assertEqual(rewards, [1.0, 1.0])

    def test_generated_expert_workflows_reward_all_tasks(self) -> None:
        sft_rows, grpo_rows = generate_records(samples_per_fault=1, seed=91)
        rewards = mechanical_reward(
            [row["completion"] for row in sft_rows],
            fault_name=[row["fault_name"] for row in grpo_rows],
            sample_seed=[row["sample_seed"] for row in grpo_rows],
            prompts=[row["prompt"] for row in grpo_rows],
            scenario_schema_version=[row["scenario_schema_version"] for row in grpo_rows],
            scenario_profile=[row["scenario_profile"] for row in grpo_rows],
            subfault_count=[row["subfault_count"] for row in grpo_rows],
        )

        self.assertEqual(rewards, [1.0] * len(WORKFLOWS))

    def test_schema_v6_replays_every_task_and_profile(self) -> None:
        rows = []
        completions = []
        for profile_index, profile in enumerate(HARD_SCENARIO_PROFILES):
            for fault_index, fault_name in enumerate(WORKFLOWS):
                variation = profile_index + 3 * fault_index
                row = build_v6_sample(
                    fault_name,
                    base_seed=81,
                    variation_index=variation,
                    split="train",
                )
                self.assertEqual(row["scenario_profile"], profile)
                rows.append(row)
                completions.append(_workflow_completion(fault_name))

        rewards = mechanical_reward(
            completions,
            fault_name=[row["fault_name"] for row in rows],
            sample_seed=[row["sample_seed"] for row in rows],
            prompts=[row["prompt"] for row in rows],
            scenario_schema_version=[row["scenario_schema_version"] for row in rows],
            scenario_profile=[row["scenario_profile"] for row in rows],
            subfault_count=[row["subfault_count"] for row in rows],
        )

        self.assertEqual(rewards, [1.0] * len(rows))

    def test_dependency_repair_ignores_a_model_guessed_version(self) -> None:
        row = build_v6_sample(
            "dependency_mismatch",
            base_seed=81,
            variation_index=1,
            split="train",
        )
        completion = (
            '{"actions":[{"action":"fix_dependency","parameters":'
            '{"name":"web-framework","version":"0.0.1-model-guess"}},'
            '{"action":"restart_app","parameters":{}}]}'
        )

        rewards = mechanical_reward(
            [completion],
            fault_name=[row["fault_name"]],
            sample_seed=[row["sample_seed"]],
            prompts=[row["prompt"]],
            scenario_schema_version=[row["scenario_schema_version"]],
            scenario_profile=[row["scenario_profile"]],
            subfault_count=[row["subfault_count"]],
        )

        self.assertEqual(rewards, [1.0])

    def test_schema_v6_prompt_profile_and_version_fail_closed(self) -> None:
        row = build_v6_sample(
            "disk_full", base_seed=44, variation_index=2, split="eval"
        )
        completion = _workflow_completion("disk_full")
        common = {
            "completions": [completion],
            "fault_name": [row["fault_name"]],
            "sample_seed": [row["sample_seed"]],
            "prompts": [row["prompt"]],
            "subfault_count": [row["subfault_count"]],
        }
        self.assertEqual(
            mechanical_reward(
                **common,
                scenario_schema_version=[2],
                scenario_profile=["redacted"],
            ),
            [0.0],
        )
        self.assertEqual(
            mechanical_reward(
                **common,
                scenario_schema_version=[99],
                scenario_profile=[row["scenario_profile"]],
            ),
            [0.0],
        )

    def test_wrong_or_malformed_actions_receive_zero(self) -> None:
        seeds = [
            sample_seed(42, "disk_full", 0),
            sample_seed(42, "bad_env_var", 0),
        ]
        rewards = mechanical_reward(
            [
                '{"actions":[{"action":"restart_app","parameters":{}}]}',
                "this is not JSON",
            ],
            fault_name=["disk_full", "bad_env_var"],
            sample_seed=seeds,
            prompts=[_prompt("disk_full", seeds[0]), _prompt("bad_env_var", seeds[1])],
            subfault_count=[1, 1],
        )

        self.assertEqual(rewards, [0.0, 0.0])

    def test_partial_credit_rewards_subfault_progress(self) -> None:
        seed = sample_seed(42, "cache_corruption", 0)
        extra: dict[str, list[object]] = {}
        rewards = mechanical_reward(
            ['{"actions":[{"action":"clear_cache","parameters":{}}]}'],
            fault_name=["cache_corruption"],
            sample_seed=[seed],
            prompts=[_prompt("cache_corruption", seed)],
            subfault_count=[WORKFLOWS["cache_corruption"].subfault_count],
            log_extra=lambda name, values: extra.__setitem__(name, list(values)),
        )
        self.assertEqual(rewards, [0.5])
        self.assertEqual(extra["crashdiag_backend_error"], [False])
        self.assertEqual(extra["crashdiag_subfault_progress"], [0.5])

    def test_seed_and_matching_prompt_are_mandatory(self) -> None:
        seed = sample_seed(42, "bad_env_var", 3)
        completion = ['{"actions":[{"action":"rollback_env_var","parameters":{}}]}']

        with self.assertRaisesRegex(ValueError, "sample_seed is required"):
            mechanical_reward(completion, fault_name=["bad_env_var"], prompts=[[]])
        with self.assertRaisesRegex(ValueError, "prompts are required"):
            mechanical_reward(
                completion,
                fault_name=["bad_env_var"],
                sample_seed=[seed],
            )
        self.assertEqual(
            mechanical_reward(
                completion,
                fault_name=["bad_env_var"],
                sample_seed=[seed],
                prompts=[[{"role": "user", "content": "tampered"}]],
                subfault_count=[1],
            ),
            [0.0],
        )

    def test_completion_text_supports_conversational_shape(self) -> None:
        self.assertEqual(
            completion_text([{"role": "assistant", "content": "answer"}]),
            "answer",
        )

    def test_distributed_batch_validation_matches_effective_batches(self) -> None:
        args = build_parser().parse_args(
            [
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "1",
                "--num-generations",
                "4",
            ]
        )
        with patch.dict("os.environ", {"WORLD_SIZE": "2"}):
            _validate_positive(args)

        invalid = build_parser().parse_args(["--num-generations", "1"])
        with self.assertRaisesRegex(SystemExit, "at least 2"):
            _validate_positive(invalid)


if __name__ == "__main__":
    unittest.main()
