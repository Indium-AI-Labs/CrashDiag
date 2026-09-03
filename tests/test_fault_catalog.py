"""Mechanical coverage for the 52-task workflow catalog."""

from __future__ import annotations

import unittest

from crashdiag.faults.workflows import WORKFLOWS
from crashdiag.sandbox_apps.mock import MockSandbox
from training.generate_dataset import expert_workflow


class FaultCatalogTests(unittest.TestCase):
    def test_every_workflow_has_a_mechanically_validated_expert_repair(self) -> None:
        self.assertEqual(len(WORKFLOWS), 52)
        for name, workflow in WORKFLOWS.items():
            with self.subTest(workflow=name):
                sandbox = MockSandbox()
                workflow.inject(sandbox)
                self.assertFalse(sandbox.health_check()["healthy"])
                target = expert_workflow(name)
                for action in target["actions"]:
                    sandbox.execute_action(action["action"], action["parameters"])
                self.assertTrue(workflow.is_resolved(sandbox))
                self.assertTrue(sandbox.health_check()["healthy"])


if __name__ == "__main__":
    unittest.main()
