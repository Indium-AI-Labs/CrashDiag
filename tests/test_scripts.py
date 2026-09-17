"""Static safety checks for operational shell entry points."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class OperationalScriptTests(unittest.TestCase):
    def test_grpo_log_captures_early_failures_and_is_ignored(self) -> None:
        script = (ROOT / "scripts" / "grpo.sh").read_text(encoding="utf-8")
        redirect = 'exec > >(tee -a "${LOG_FILE}") 2>&1'
        self.assertIn(redirect, script)
        self.assertLess(script.index(redirect), script.index('if [[ ! -f "${ENV_FILE}" ]]'))
        self.assertNotIn('training.grpo_pipeline 2>&1 | tee', script)

        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/grpo.log", ignored)

    def test_fresh_start_forwards_explicit_reset_confirmation_and_bucket(self) -> None:
        source = (ROOT / "scripts" / "fresh_start.py").read_text(encoding="utf-8")
        self.assertIn('"--bucket-id",', source)
        self.assertIn("bucket_id,", source)
        self.assertIn('"--yes",', source)


if __name__ == "__main__":
    unittest.main()
