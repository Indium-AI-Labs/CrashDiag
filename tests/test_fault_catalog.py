"""Mechanical coverage for the reset's 18-task fault catalog."""

from __future__ import annotations

import unittest

from crashdiag.faults.modules import ALL_FAULTS
from crashdiag.sandbox_apps.mock import MockSandbox
from training.generate_dataset import expert_action


class FaultCatalogTests(unittest.TestCase):
    def test_every_fault_has_a_mechanically_validated_expert_repair(self) -> None:
        self.assertEqual(len(ALL_FAULTS), 18)
        for fault in ALL_FAULTS:
            with self.subTest(fault=fault.name):
                sandbox = MockSandbox()
                fault.inject(sandbox)
                self.assertFalse(sandbox.health_check()["healthy"])
                action = expert_action(fault.name, sandbox, __import__("random").Random(42))
                sandbox.execute_action(action["action"], action["parameters"])
                self.assertTrue(fault.is_resolved(sandbox))
                self.assertTrue(sandbox.health_check()["healthy"])
