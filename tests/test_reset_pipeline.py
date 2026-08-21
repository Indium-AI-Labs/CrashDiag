"""Safety tests for destructive pipeline reset helpers."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from scripts import fresh_start
from scripts import reset_pipeline


class ResetPipelineSafetyTests(unittest.TestCase):
    def test_fresh_start_requires_explicit_confirmation(self) -> None:
        with (
            patch.object(sys, "argv", ["fresh_start.py"]),
            patch.object(fresh_start, "_load_env") as load_env,
            self.assertRaisesRegex(SystemExit, "--yes"),
        ):
            fresh_start.main()

        load_env.assert_not_called()

    def test_reset_requires_explicit_confirmation(self) -> None:
        with (
            patch.object(sys, "argv", ["reset_pipeline.py", "--skip-bucket"]),
            patch.object(reset_pipeline, "load_env_file"),
            patch.object(reset_pipeline, "reset_local") as reset_local,
            patch.object(reset_pipeline, "reset_bucket") as reset_bucket,
            self.assertRaisesRegex(SystemExit, "--yes"),
        ):
            reset_pipeline.main()

        reset_local.assert_not_called()
        reset_bucket.assert_not_called()

    def test_remote_reset_requires_explicit_bucket_and_uses_it(self) -> None:
        argv = [
            "reset_pipeline.py",
            "--yes",
            "--bucket-id",
            "example/CrashDiag",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {"HF_TOKEN": "hf-test-placeholder"}, clear=True),
            patch.object(reset_pipeline, "load_env_file"),
            patch.object(reset_pipeline, "reset_local") as reset_local,
            patch.object(reset_pipeline, "reset_bucket") as reset_bucket,
        ):
            self.assertEqual(reset_pipeline.main(), 0)

        reset_local.assert_called_once_with()
        reset_bucket.assert_called_once_with("example/CrashDiag", "hf-test-placeholder")

    def test_confirmed_local_reset_does_not_require_hf_credentials(self) -> None:
        with (
            patch.object(
                sys,
                "argv",
                ["reset_pipeline.py", "--yes", "--skip-bucket"],
            ),
            patch.dict(os.environ, {}, clear=True),
            patch.object(reset_pipeline, "load_env_file") as load_env,
            patch.object(reset_pipeline, "reset_local") as reset_local,
            patch.object(reset_pipeline, "reset_bucket") as reset_bucket,
        ):
            self.assertEqual(reset_pipeline.main(), 0)

        load_env.assert_not_called()
        reset_local.assert_called_once_with()
        reset_bucket.assert_not_called()


if __name__ == "__main__":
    unittest.main()
