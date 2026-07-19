from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from training.common import FAULT_NAMES, observation_messages
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
    build_hard_grpo_sample,
    hard_expert_action,
)


def _prompt(fault_name: str, scenario_seed: int) -> list[dict[str, str]]:
    _, sandbox, _ = prepare_scenario(fault_name, scenario_seed)
    return observation_messages(sandbox.observe())


class MechanicalRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        configure_reward_backend(sandbox_url=None)

    def test_correct_actions_are_rewarded_from_sandbox_state(self) -> None:
        completions = [
            [{"role": "assistant", "content": '{"action":"restart_app","parameters":{}}'}],
            [{"role": "assistant", "content": '{"action":"clear_disk","parameters":{}}'}],
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
        )

        self.assertEqual(rewards, [1.0, 1.0])

    def test_generated_expert_actions_reward_all_six_faults(self) -> None:
        sft_rows, grpo_rows = generate_records(samples_per_fault=1, seed=91)
        rewards = mechanical_reward(
            [row["completion"] for row in sft_rows],
            fault_name=[row["fault_name"] for row in grpo_rows],
            sample_seed=[row["sample_seed"] for row in grpo_rows],
            prompts=[row["prompt"] for row in grpo_rows],
        )

        self.assertEqual(rewards, [1.0] * 6)

    def test_schema_v2_replays_every_fault_and_profile(self) -> None:
        rows = []
        completions = []
        for profile_index, profile in enumerate(HARD_SCENARIO_PROFILES):
            for fault_index, fault_name in enumerate(FAULT_NAMES):
                variation = profile_index + 3 * fault_index
                row = build_hard_grpo_sample(
                    fault_name,
                    base_seed=81,
                    variation_index=variation,
                    split="train",
                )
                self.assertEqual(row["scenario_profile"], profile)
                rows.append(row)
                completions.append(json.dumps(hard_expert_action(fault_name)))

        rewards = mechanical_reward(
            completions,
            fault_name=[row["fault_name"] for row in rows],
            sample_seed=[row["sample_seed"] for row in rows],
            prompts=[row["prompt"] for row in rows],
            scenario_schema_version=[row["scenario_schema_version"] for row in rows],
            scenario_profile=[row["scenario_profile"] for row in rows],
        )

        self.assertEqual(rewards, [1.0] * len(rows))

    def test_schema_v2_prompt_profile_and_version_fail_closed(self) -> None:
        row = build_hard_grpo_sample(
            "disk_full", base_seed=44, variation_index=2, split="eval"
        )
        completion = json.dumps(hard_expert_action("disk_full"))
        common = {
            "completions": [completion],
            "fault_name": [row["fault_name"]],
            "sample_seed": [row["sample_seed"]],
            "prompts": [row["prompt"]],
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
                '{"action":"restart_app","parameters":{}}',
                "this is not JSON",
            ],
            fault_name=["disk_full", "bad_env_var"],
            sample_seed=seeds,
            prompts=[_prompt("disk_full", seeds[0]), _prompt("bad_env_var", seeds[1])],
        )

        self.assertEqual(rewards, [0.0, 0.0])

    def test_seed_and_matching_prompt_are_mandatory(self) -> None:
        seed = sample_seed(42, "bad_env_var", 3)
        completion = ['{"action":"rollback_env_var","parameters":{}}']

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
