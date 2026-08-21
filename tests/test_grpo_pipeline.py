"""Security and wiring tests for the end-to-end GRPO pipeline."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from training import grpo_pipeline


class _DatasetUploader:
    def __init__(self, _config: object) -> None:
        pass

    def download_stage(self, stage: str, destination: Path) -> None:
        if stage != "datasets":
            raise AssertionError(f"unexpected stage: {stage}")
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "grpo_train.jsonl").write_text("{}\n", encoding="utf-8")
        (destination / "grpo_eval.jsonl").write_text("{}\n", encoding="utf-8")


class GrpoPipelineSecurityTests(unittest.TestCase):
    def test_main_keeps_sandbox_token_out_of_child_argv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env_file = root / "env.txt"
            env_file.write_text("# values supplied by the process environment\n", encoding="utf-8")
            commands: list[list[str]] = []
            environment = {
                "HF_TOKEN": "hf-test-placeholder",
                "CRASHDIAG_DATASET_RUN_ID": "dataset-test",
                "CRASHDIAG_HF_BUCKET_ID": "example/CrashDiag",
                "CRASHDIAG_SANDBOX_URL": "https://sandbox.example.com",
                "CRASHDIAG_SANDBOX_TOKEN": "sandbox-test-placeholder",
                "CRASHDIAG_GRPO_RUN_ID": "grpo-test",
                "CRASHDIAG_GRPO_EVAL_RUN_ID": "grpo-eval-test",
            }
            with (
                patch.dict(os.environ, environment, clear=True),
                patch.object(grpo_pipeline, "REPO_ROOT", root),
                patch.object(grpo_pipeline, "ENV_FILE", env_file),
                patch.object(sys, "path", list(sys.path)),
                patch("training.artifacts.ArtifactUploader", _DatasetUploader),
                patch.object(
                    grpo_pipeline,
                    "run",
                    side_effect=lambda command: commands.append(list(command)),
                ),
            ):
                self.assertEqual(grpo_pipeline.main(), 0)

            self.assertEqual(len(commands), 2)
            for command in commands:
                self.assertNotIn("--sandbox-token", command)
                self.assertNotIn("sandbox-test-placeholder", command)

    def test_command_logging_redacts_sensitive_flag_values(self) -> None:
        command = ["worker", "--sandbox-token", "do-not-print", "--max-steps", "1"]
        output = io.StringIO()
        with (
            patch.object(subprocess, "run") as run_process,
            redirect_stdout(output),
        ):
            grpo_pipeline.run(command)

        run_process.assert_called_once_with(command, cwd=grpo_pipeline.REPO_ROOT, check=True)
        self.assertNotIn("do-not-print", output.getvalue())
        self.assertIn("<redacted>", output.getvalue())


if __name__ == "__main__":
    unittest.main()
