from __future__ import annotations

import unittest

from crashdiag.faults.modules import (
    ALL_FAULTS,
    BadEnvVar,
    BrokenDBConnection,
    DependencyMismatch,
    DiskFull,
    OOMKill,
    PortProxyMisconfig,
)
from crashdiag.sandbox_apps.mock import MockSandbox


class MockFaultTests(unittest.TestCase):
    CASES = (
        (OOMKill, "restart_app"),
        (BadEnvVar, "rollback_env_var"),
        (BrokenDBConnection, "rollback_env_var"),
        (DependencyMismatch, "fix_dependency"),
        (DiskFull, "clear_disk"),
        (PortProxyMisconfig, "fix_port_config"),
    )

    def test_all_faults_inject_and_resolve_mechanically(self) -> None:
        self.assertEqual(len(ALL_FAULTS), 6)

        for fault_type, recovery_action in self.CASES:
            with self.subTest(fault=fault_type.__name__):
                sandbox = MockSandbox()
                fault = fault_type()

                self.assertTrue(sandbox.health_check()["healthy"])
                fault.inject(sandbox)
                self.assertFalse(fault.is_resolved(sandbox))
                self.assertFalse(sandbox.health_check()["healthy"])

                sandbox.execute_action("wait_and_observe")
                self.assertFalse(fault.is_resolved(sandbox))

                sandbox.execute_action(recovery_action)

                self.assertTrue(fault.is_resolved(sandbox))
                self.assertTrue(sandbox.health_check()["healthy"])

    def test_fault_resolution_also_requires_application_health(self) -> None:
        sandbox = MockSandbox()
        env_fault = BadEnvVar()
        disk_fault = DiskFull()
        env_fault.inject(sandbox)
        disk_fault.inject(sandbox)

        sandbox.execute_action("rollback_env_var", {"name": "APP_ENV"})

        self.assertFalse(env_fault.is_resolved(sandbox))
        self.assertFalse(sandbox.health_check()["healthy"])

    def test_action_dispatch_rejects_actions_outside_the_fixed_space(self) -> None:
        with self.assertRaises(ValueError):
            MockSandbox().execute_action("run_shell", {"command": "whoami"})


if __name__ == "__main__":
    unittest.main()
