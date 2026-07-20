"""Tests for the answer-free schema-v2 GRPO dataset handoff."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from training.artifacts import ArtifactError
from training.common import FAULT_NAMES
from training.generate_grpo_hard import generate_hard_datasets, read_parent_reference
from training.hard_scenarios import HARD_SCENARIO_PROFILES


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HardDatasetTests(unittest.TestCase):
    def test_default_shape_is_768_train_and_192_eval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            summary_path = root / "summary.json"
            summary = generate_hard_datasets(train, evaluation, summary_path)

            self.assertEqual(summary["train"]["rows"], 768)
            self.assertEqual(summary["eval"]["rows"], 192)
            self.assertEqual(summary["curriculum_version"], 2)
            self.assertEqual(summary["action_contract"], "parameter_free_repairs")
            train_rows = [json.loads(line) for line in train.read_text().splitlines()]
            eval_rows = [json.loads(line) for line in evaluation.read_text().splitlines()]
            self.assertEqual(
                Counter(row["fault_name"] for row in train_rows),
                Counter({name: 128 for name in FAULT_NAMES}),
            )
            self.assertEqual(
                Counter(row["fault_name"] for row in eval_rows),
                Counter({name: 32 for name in FAULT_NAMES}),
            )
            self.assertEqual(
                {row["scenario_profile"] for row in train_rows},
                set(HARD_SCENARIO_PROFILES),
            )
            self.assertTrue(
                {row["sample_seed"] for row in train_rows}.isdisjoint(
                    {row["sample_seed"] for row in eval_rows}
                )
            )
            for row in train_rows + eval_rows:
                self.assertNotIn("completion", row)
                self.assertNotIn("answer", row)
                self.assertNotIn("expert_action", row)

    def test_parent_reference_uses_signed_manifest_weight_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "adapter_config.json"
            config.write_text(
                json.dumps({"base_model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct"}),
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
