"""Schema-v2 hard-scenario determinism and mechanical-solvability tests."""

from __future__ import annotations

import json
import unittest
from collections import Counter

from training.common import FAULT_NAMES
from training.hard_scenarios import (
    HARD_CURRICULUM_VERSION,
    HARD_SCENARIO_PROFILES,
    build_hard_grpo_sample,
    generate_hard_records,
    hard_expert_action,
    hard_observation_messages,
    hard_sample_seed,
    prepare_hard_scenario,
)


class HardScenarioTests(unittest.TestCase):
    def test_every_profile_and_fault_is_solvable_with_one_bounded_action(self) -> None:
        for profile_index, profile in enumerate(HARD_SCENARIO_PROFILES):
            for fault_name in FAULT_NAMES:
                with self.subTest(profile=profile, fault=fault_name):
                    seed = hard_sample_seed(42, fault_name, profile_index)
                    fault, sandbox, _ = prepare_hard_scenario(
                        fault_name,
                        seed,
                        profile,
                    )
                    prompt = hard_observation_messages(sandbox.observe())
                    self.assertIn(
                        "parameters value must therefore be exactly {}",
                        prompt[0]["content"],
                    )
                    text = prompt[1]["content"]
                    for forbidden in (
                        '"failures"',
                        '"checks"',
                        '"expected"',
                        '"required"',
                        '"app_port"',
                        '"healthy_below_percent"',
                    ):
                        self.assertNotIn(forbidden, text)
                    sandbox.wait_and_observe()
                    self.assertFalse(fault.is_resolved(sandbox))
                    action = hard_expert_action(fault_name)
                    sandbox.execute_action(action["action"], action["parameters"])
                    self.assertTrue(fault.is_resolved(sandbox))
                    self.assertTrue(sandbox.health_check()["healthy"])

    def test_generation_is_answer_free_deterministic_balanced_and_disjoint(self) -> None:
        first = generate_hard_records(
            samples_per_fault=6,
            seed=42,
            start_variation=0,
            split="train",
        )
        second = generate_hard_records(
            samples_per_fault=6,
            seed=42,
            start_variation=0,
            split="train",
        )
        evaluation = generate_hard_records(
            samples_per_fault=3,
            seed=42,
            start_variation=6,
            split="eval",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        self.assertEqual(Counter(row["fault_name"] for row in first), {name: 6 for name in FAULT_NAMES})
        self.assertEqual(
            Counter(row["scenario_profile"] for row in first),
            {profile: 12 for profile in HARD_SCENARIO_PROFILES},
        )
        train_seeds = {row["sample_seed"] for row in first}
        eval_seeds = {row["sample_seed"] for row in evaluation}
        self.assertTrue(train_seeds.isdisjoint(eval_seeds))
        for row in first + evaluation:
            serialized = json.dumps(row, sort_keys=True)
            self.assertNotIn('"completion"', serialized)
            self.assertNotIn('"answer"', serialized)
            self.assertNotIn('"target"', serialized)
            self.assertEqual(row["scenario_schema_version"], 2)
            self.assertEqual(row["curriculum_version"], HARD_CURRICULUM_VERSION)
            self.assertEqual(
                row["metadata"]["curriculum_version"],
                HARD_CURRICULUM_VERSION,
            )
            self.assertTrue(row["metadata"]["mechanically_validated"])

    def test_shifted_profile_changes_hidden_configuration_without_leaking_it(self) -> None:
        observations = []
        for variation in range(6):
            seed = hard_sample_seed(9, "port_proxy_misconfig", variation)
            _, sandbox, _ = prepare_hard_scenario(
                "port_proxy_misconfig",
                seed,
                "shifted_noisy",
            )
            raw = sandbox.observe()
            observations.append(raw["network"]["app_port"])
            prompt = hard_observation_messages(raw)
            self.assertNotIn(str(raw["network"]["app_port"]), prompt[1]["content"])
        self.assertGreater(len(set(observations)), 1)

    def test_build_sample_profile_follows_variation(self) -> None:
        rows = [
            build_hard_grpo_sample(
                "oom_kill",
                base_seed=42,
                variation_index=index,
                split="train",
            )
            for index in range(3)
        ]
        self.assertEqual(
            [row["scenario_profile"] for row in rows],
            list(HARD_SCENARIO_PROFILES),
        )


if __name__ == "__main__":
    unittest.main()
