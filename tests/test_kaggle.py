"""Tests for the secret-preserving Kaggle launcher."""

from __future__ import annotations

import io
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from training.kaggle import build_parser, launch


class _SecretsClient:
    def get_secret(self, name: str) -> str:
        return {
            "HF_TOKEN": "hf_private_launcher_value",
            "CRASHDIAG_SANDBOX_TOKEN": "sandbox_private_launcher_value",
        }[name]


class KaggleLauncherTests(unittest.TestCase):
    def test_secrets_reach_child_environment_but_not_output_or_cli(self) -> None:
        module = types.ModuleType("kaggle_secrets")
        module.UserSecretsClient = _SecretsClient
        args = build_parser().parse_args(
            [
                "--sandbox-url",
                "https://sandbox.example.com",
                "--run-id",
                "run-123",
                "--",
                "python",
                "-m",
                "training.generate_dataset",
            ]
        )
        completed = types.SimpleNamespace(returncode=0)
        output = io.StringIO()
        with (
            patch.dict("sys.modules", {"kaggle_secrets": module}),
            patch("training.kaggle.subprocess.run", return_value=completed) as run,
            redirect_stdout(output),
        ):
            self.assertEqual(launch(args), 0)

        command = run.call_args.args[0]
        environment = run.call_args.kwargs["env"]
        self.assertEqual(command, ["python", "-m", "training.generate_dataset"])
        self.assertEqual(environment["HF_TOKEN"], "hf_private_launcher_value")
        self.assertEqual(
            environment["CRASHDIAG_SANDBOX_TOKEN"],
            "sandbox_private_launcher_value",
        )
        self.assertEqual(environment["CRASHDIAG_RUN_ID"], "run-123")
        self.assertNotIn("hf_private_launcher_value", output.getvalue())
        self.assertNotIn("sandbox_private_launcher_value", output.getvalue())
        self.assertNotIn("hf_private_launcher_value", repr(command))

    def test_https_sandbox_is_required_before_reading_secrets(self) -> None:
        args = build_parser().parse_args(["--sandbox-url", "http://sandbox.test"])
        with self.assertRaisesRegex(SystemExit, "HTTPS"):
            launch(args)


if __name__ == "__main__":
    unittest.main()
