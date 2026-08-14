"""Tests for the answer-free v5 multi-action GRPO dataset handoff."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from crashdiag.faults.workflows import WORKFLOWS
from training.artifacts import ArtifactError
from training.generate_grpo_hard import read_parent_reference
from training.hard_scenarios import HARD_SCENARIO_PROFILES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HardDatasetTests(unittest.TestCase):
    def test_dataset_shape_is_balanced_and_disjoint(self) -> None:
        self.assertEqual(len(WORKFLOWS), 52)
        from training.generate_dataset import generate_datasets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "sft_train.jsonl"
            evaluation = root / "sft_eval.jsonl"
            grpo_train = root / "grpo_train.jsonl"
            grpo_eval = root / "grpo_eval.jsonl"
            summary_path = root / "summary.json"
            counts = generate_datasets(
                train,
                evaluation,
                grpo_train,
                grpo_eval,
                summary_path,
                train_samples_per_fault=1,
                eval_samples_per_fault=1,
                seed=19,
            )
            self.assertEqual(counts, {"train": 52, "eval": 52})
            train_rows = [json.loads(line) for line in train.read_text().splitlines()]
            self.assertEqual(
                Counter(row["fault_name"] for row in train_rows),
                Counter({name: 1 for name in WORKFLOWS}),
            )
            self.assertTrue(
                {row["sample_seed"] for row in train_rows}.isdisjoint(
                    {row["sample_seed"] for row in [json.loads(line) for line in evaluation.read_text().splitlines()]}
                )
            )
            for row in train_rows:
                self.assertIn("actions", row["completion"][0]["content"])
                self.assertNotIn("expert_action", row)

    def test_parent_reference_uses_signed_manifest_weight_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "adapter_config.json"
            config.write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-3B-Instruct"}),
                encoding="utf-8",
            )
            run_id = "parent-sft"
            manifest = {
                "schema_version": 1,
                "run_id": run_id,
                "stage": "sft",
                "runtime": {"git_commit": "a" * 40},
                "files": [
                    {
                        "path": "adapter_config.json",
                        "bytes": config.stat().st_size,
                        "sha256": _sha(config),
                    },
                    {
                        "path": "adapter_model.safetensors",
                        "bytes": 1234,
                        "sha256": "b" * 64,
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "_SUCCESS.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "run_id": run_id,
                        "stage": "sft",
                        "manifest_sha256": _sha(manifest_path),
                    }
                ),
                encoding="utf-8",
            )

            reference = read_parent_reference(root, run_id)
            self.assertEqual(reference["adapter_sha256"], "b" * 64)
            self.assertEqual(reference["adapter_bytes"], 1234)
            self.assertEqual(reference["source_commit"], "a" * 40)

            manifest["files"][0]["sha256"] = "c" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            success = json.loads((root / "_SUCCESS.json").read_text())
            success["manifest_sha256"] = _sha(manifest_path)
            (root / "_SUCCESS.json").write_text(json.dumps(success))
            with self.assertRaisesRegex(ArtifactError, "adapter_config"):
                read_parent_reference(root, run_id)


if __name__ == "__main__":
    unittest.main()
