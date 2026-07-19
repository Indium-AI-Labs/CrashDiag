"""Tests for dependency-free dataset generation and training helpers."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from crashdiag.agents import ACTION_SPACE, parse_action
from training.common import (
    FAULT_NAMES,
    SYSTEM_PROMPT,
    completion_text,
    fault_for_name,
    resolve_precision,
)
from training.artifacts import ArtifactError
from training.generate_dataset import generate_datasets, generate_records, main
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

    def test_cli_automatically_uploads_a_versioned_private_dataset_stage(self) -> None:
        class FakeUploader:
            def __init__(self, output: io.StringIO) -> None:
                self.output = output
                self.calls: list[tuple[object, ...]] = []
                self.output_before_start = ""

            def start_run(self, metadata: object) -> None:
                self.output_before_start = self.output.getvalue()
                self.calls.append(("start_run", metadata))

            def start_stage(self, stage: str, metadata: object) -> None:
                self.calls.append(("start_stage", stage, metadata))

            def upload_files(
                self, files: object, stage: str, *, metadata: object
            ) -> None:
                self.calls.append(("upload_files", tuple(files), stage, metadata))

            @staticmethod
            def remote_uri(stage: str | None = None) -> str:
                suffix = f"/{stage}" if stage else ""
                return "hf://buckets/devaanshpa/CrashDiag/runs/generated" + suffix

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / f"dataset-{index}.jsonl" for index in range(4)]
            missing_env = root / "missing.env"
            stdout = io.StringIO()
            fake = FakeUploader(stdout)
            captured_args: list[object] = []

            def fake_uploader_from_args(args: object) -> FakeUploader:
                captured_args.append(args)
                return fake

            argv = [
                "--env-file",
                str(missing_env),
                "--sft-train-output",
                str(outputs[0]),
                "--sft-eval-output",
                str(outputs[1]),
                "--grpo-train-output",
                str(outputs[2]),
                "--grpo-eval-output",
                str(outputs[3]),
                "--train-samples-per-fault",
                "1",
                "--eval-samples-per-fault",
                "1",
                "--seed",
                "19",
            ]
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "training.generate_dataset.runtime_metadata",
                    return_value={"git_commit": "a" * 40},
                ),
                patch(
                    "training.generate_dataset.uploader_from_args",
                    side_effect=fake_uploader_from_args,
                ),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(argv), 0)

            self.assertEqual(len(captured_args), 1)
            args = captured_args[0]
            self.assertEqual(args.artifact_bucket, "devaanshpa/CrashDiag")
            self.assertRegex(
                args.run_id,
                r"^\d{8}T\d{6}Z-dataset-[0-9a-f]{12}$",
            )
            self.assertIn(f"RUN_ID={args.run_id}", fake.output_before_start)
            self.assertIn(f"SOURCE_COMMIT={'a' * 40}", fake.output_before_start)
            self.assertEqual(
                [call[0] for call in fake.calls],
                ["start_run", "start_stage", "upload_files"],
            )
            self.assertEqual(fake.calls[0][1]["source_commit"], "a" * 40)
            self.assertEqual(fake.calls[1][1], "datasets")
            self.assertEqual(fake.calls[1][2]["source_commit"], "a" * 40)
            self.assertEqual(fake.calls[1][2]["seed"], 19)
            self.assertEqual(fake.calls[2][1], tuple(outputs))
            self.assertEqual(fake.calls[2][2], "datasets")
            self.assertEqual(fake.calls[2][3]["train_rows"], len(FAULT_NAMES))
            self.assertEqual(fake.calls[2][3]["eval_rows"], len(FAULT_NAMES))
            self.assertTrue(fake.calls[2][3]["mechanically_validated"])
            self.assertFalse(fake.calls[2][3]["grpo_targets_included"])
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertIn(
                "artifacts: hf://buckets/devaanshpa/CrashDiag/runs/generated/datasets",
                stdout.getvalue(),
            )

    def test_required_upload_failure_happens_before_dataset_writes(self) -> None:
        class FailingUploader:
            @staticmethod
            def start_run(metadata: object) -> None:
                raise ArtifactError("private bucket unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "sft-train.jsonl"
            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "training.generate_dataset.runtime_metadata",
                    return_value={"git_commit": "b" * 40},
                ),
                patch(
                    "training.generate_dataset.uploader_from_args",
                    return_value=FailingUploader(),
                ),
                redirect_stdout(stdout),
            ):
                with self.assertRaisesRegex(
                    SystemExit, "private bucket unavailable"
                ):
                    main(
                        [
                            "--env-file",
                            str(root / "missing.env"),
                            "--sft-train-output",
                            str(target),
                        ]
                    )
            self.assertFalse(target.exists())
            self.assertRegex(stdout.getvalue(), r"RUN_ID=.*-dataset-[0-9a-f]{12}")
            self.assertIn(f"SOURCE_COMMIT={'b' * 40}", stdout.getvalue())

    def test_automatic_upload_requires_hf_token_before_dataset_writes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / f"required-{index}.jsonl" for index in range(4)]
            stdout = io.StringIO()
            argv = [
                "--env-file",
                str(root / "missing.env"),
                "--sft-train-output",
                str(outputs[0]),
                "--sft-eval-output",
                str(outputs[1]),
                "--grpo-train-output",
                str(outputs[2]),
                "--grpo-eval-output",
                str(outputs[3]),
            ]
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "training.generate_dataset.runtime_metadata",
                    return_value={"git_commit": "c" * 40},
                ),
                redirect_stdout(stdout),
            ):
                with self.assertRaisesRegex(SystemExit, "HF_TOKEN"):
                    main(argv)
            self.assertTrue(all(not path.exists() for path in outputs))
            self.assertIn("RUN_ID=", stdout.getvalue())
            self.assertNotIn("hf_", stdout.getvalue())

    def test_local_only_generation_requires_an_explicit_opt_out(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = [root / f"local-{index}.jsonl" for index in range(4)]
            argv = [
                "--env-file",
                str(root / "missing.env"),
                "--artifact-upload-policy",
                "disabled",
                "--sft-train-output",
                str(outputs[0]),
                "--sft-eval-output",
                str(outputs[1]),
                "--grpo-train-output",
                str(outputs[2]),
                "--grpo-eval-output",
                str(outputs[3]),
                "--train-samples-per-fault",
                "1",
                "--eval-samples-per-fault",
                "1",
            ]
            stdout = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "training.generate_dataset.runtime_metadata",
                    return_value={"git_commit": "unknown"},
                ),
                redirect_stdout(stdout),
            ):
                self.assertEqual(main(argv), 0)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertIn("artifact upload: disabled explicitly", stdout.getvalue())


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
