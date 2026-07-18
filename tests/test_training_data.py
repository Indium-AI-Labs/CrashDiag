"""Tests for dependency-free dataset generation and training helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from crashdiag.agents import ACTION_SPACE, parse_action
from training.common import (
    FAULT_NAMES,
    SYSTEM_PROMPT,
    completion_text,
    fault_for_name,
    resolve_precision,
)
from training.generate_dataset import generate_datasets, generate_records
from training.sft import build_parser


class TrainingCommonTests(unittest.TestCase):
    def test_fault_for_name_returns_fresh_faults(self) -> None:
        for name in FAULT_NAMES:
            first = fault_for_name(name)
            second = fault_for_name(name)
            self.assertEqual(first.name, name)
            self.assertIsNot(first, second)

    def test_completion_text_handles_standard_and_conversational_values(self) -> None:
        self.assertEqual(completion_text("plain"), "plain")
        self.assertEqual(
            completion_text([{"role": "assistant", "content": "action"}]),
            "action",
        )
        self.assertEqual(
            completion_text([{"type": "text", "text": "one"}, {"text": " two"}]),
            "one two",
        )
        self.assertEqual(completion_text({"unexpected": True}), "")

    def test_precision_resolution_has_no_torch_import_requirement(self) -> None:
        class CpuCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class FakeTorch:
            cuda = CpuCuda()

        self.assertEqual(resolve_precision(FakeTorch, "auto"), (False, False))
        self.assertEqual(resolve_precision(FakeTorch, "bf16"), (True, False))
        self.assertEqual(resolve_precision(FakeTorch, "fp16"), (False, True))
        with self.assertRaises(ValueError):
            resolve_precision(FakeTorch, "int8")


class DatasetGenerationTests(unittest.TestCase):
    def test_records_are_stratified_validated_and_answer_free(self) -> None:
        sft_rows, grpo_rows = generate_records(samples_per_fault=2, seed=73)
        expected_count = len(FAULT_NAMES) * 2
        self.assertEqual(len(sft_rows), expected_count)
        self.assertEqual(len(grpo_rows), expected_count)
        self.assertEqual(
            Counter(row["fault_name"] for row in sft_rows),
            Counter({name: 2 for name in FAULT_NAMES}),
        )

        for sft, grpo in zip(sft_rows, grpo_rows, strict=True):
            self.assertEqual(sft["prompt"], grpo["prompt"])
            self.assertEqual(sft["sample_seed"], grpo["sample_seed"])
            self.assertEqual(sft["prompt"][0], {"role": "system", "content": SYSTEM_PROMPT})
            self.assertTrue(sft["metadata"]["mechanically_validated"])
            self.assertNotIn("completion", grpo)
            self.assertNotIn("answer", grpo)
            self.assertNotIn("expert_action", grpo)
            self.assertEqual(
                set(grpo),
                {
                    "fault_name",
                    "difficulty",
                    "sample_seed",
                    "variation_index",
                    "prompt",
                    "metadata",
                },
            )
            target = parse_action(completion_text(sft["completion"]))
            self.assertIn(target["action"], ACTION_SPACE)
            self.assertNotEqual(target["action"], "wait_and_observe")

    def test_generation_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_paths = [root / f"first-{name}.jsonl" for name in range(4)]
            second_paths = [root / f"second-{name}.jsonl" for name in range(4)]
            counts = generate_datasets(
                *first_paths,
                train_samples_per_fault=1,
                eval_samples_per_fault=1,
                seed=19,
            )
            generate_datasets(
                *second_paths,
                train_samples_per_fault=1,
                eval_samples_per_fault=1,
                seed=19,
            )
            self.assertEqual(counts, {"train": len(FAULT_NAMES), "eval": len(FAULT_NAMES)})
            for first, second in zip(first_paths, second_paths, strict=True):
                self.assertEqual(first.read_bytes(), second.read_bytes())
            for grpo_path in (first_paths[2], first_paths[3]):
                for line in grpo_path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertNotIn("completion", row)

            train_seeds = {
                json.loads(line)["sample_seed"]
                for line in first_paths[0].read_text(encoding="utf-8").splitlines()
            }
            eval_seeds = {
                json.loads(line)["sample_seed"]
                for line in first_paths[1].read_text(encoding="utf-8").splitlines()
            }
            self.assertTrue(train_seeds.isdisjoint(eval_seeds))

    def test_invalid_generation_arguments_fail_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "same.jsonl"
            with self.assertRaises(ValueError):
                generate_datasets(
                    target,
                    target,
                    Path(directory) / "grpo-train.jsonl",
                    Path(directory) / "grpo-eval.jsonl",
                    train_samples_per_fault=1,
                    eval_samples_per_fault=1,
                )
            self.assertFalse(target.exists())
        with self.assertRaises(ValueError):
            generate_records(samples_per_fault=0)


class SftCliTests(unittest.TestCase):
    def test_sft_module_and_parser_do_not_require_ml_dependencies(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.precision, "auto")
        self.assertEqual(args.lora_target_modules, "all-linear")
        self.assertEqual(args.dataset, Path("data/sft_train.jsonl"))
        self.assertEqual(args.eval_dataset, Path("data/sft_eval.jsonl"))
        self.assertEqual(args.output_dir, Path("outputs/sft"))
        split_args = build_parser().parse_args(["--no-eval-dataset"])
        self.assertIsNone(split_args.eval_dataset)


if __name__ == "__main__":
    unittest.main()
