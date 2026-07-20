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
    def test_scenario_baseline_configuration_is_mechanical_and_repairable(self) -> None:
        sandbox = MockSandbox()
        sandbox.set_expected_env_var("APP_ENV", "canary")
        sandbox.set_required_dependency_version("web-framework", "2.1.1")
        sandbox.set_app_port(8443)
        sandbox.set_disk_health_threshold(80.0)

        observation = sandbox.observe()
        self.assertEqual(observation["environment"]["expected"]["APP_ENV"], "canary")
        self.assertEqual(observation["environment"]["variables"]["APP_ENV"], "canary")
        self.assertEqual(
            observation["dependencies"]["required"]["web-framework"], "2.1.1"
        )
        self.assertEqual(
            observation["dependencies"]["installed"]["web-framework"], "2.1.1"
        )
        self.assertEqual(observation["network"], {"app_port": 8443, "proxy_target_port": 8443})
        self.assertEqual(observation["disk"]["healthy_below_percent"], 80.0)
        self.assertTrue(sandbox.health_check()["healthy"])

        sandbox.set_dependency_version("web-framework", "0.0.1")
        sandbox.fix_dependency()
        sandbox.set_proxy_target_port(3000)
        sandbox.fix_port_config()
        self.assertTrue(sandbox.health_check()["healthy"])

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

    def test_dependency_repair_uses_declared_version_not_requested_version(self) -> None:
        sandbox = MockSandbox()
        sandbox.set_dependency_version("web-framework", "9.9.9")

        sandbox.fix_dependency("web-framework", "0.0.1-model-guess")

        observation = sandbox.observe()
        self.assertEqual(
            observation["dependencies"]["installed"]["web-framework"],
            observation["dependencies"]["required"]["web-framework"],
        )


if __name__ == "__main__":
    unittest.main()
