"""Mechanical tests for the Docker victim runtime and optional Compose backend."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from crashdiag.faults.workflows import WORKFLOWS
from crashdiag.sandbox_apps.docker import docker_available, host_repo_root
from crashdiag.sandbox_apps.target.runtime import VictimRuntime
from training.generate_dataset import expert_workflow
from training.hard_scenarios import (
    HARD_SCENARIO_PROFILES,
    hard_expert_workflow,
    hard_sample_seed,
    prepare_v6_scenario,
)


def _docker_tests_enabled() -> bool:
    flag = os.environ.get("CRASHDIAG_DOCKER_TESTS", "").strip().lower()
    return flag in {"1", "true", "yes"} and docker_available()


class DockerSandboxConfigTests(unittest.TestCase):
    def test_host_repo_root_prefers_engine_visible_override(self) -> None:
        with patch.dict(
            os.environ, {"CRASHDIAG_REPO_ROOT": "/host/CrashDiag"}, clear=False
        ):
            self.assertEqual(host_repo_root(), "/host/CrashDiag")


class VictimRuntimeTests(unittest.TestCase):
    def test_every_workflow_resolves_from_real_process_and_file_state(self) -> None:
        self.assertEqual(len(WORKFLOWS), 52)
        for name, workflow in WORKFLOWS.items():
            with self.subTest(workflow=name):
                with VictimRuntime() as sandbox:
                    self.assertTrue(sandbox.health_check()["healthy"])
                    workflow.inject(sandbox)
                    self.assertFalse(sandbox.health_check()["healthy"])
                    sandbox.execute_action("wait_and_observe")
                    self.assertFalse(workflow.is_resolved(sandbox))
                    for action in expert_workflow(name)["actions"]:
                        sandbox.execute_action(action["action"], action["parameters"])
                    self.assertTrue(workflow.is_resolved(sandbox))
                    self.assertTrue(sandbox.health_check()["healthy"])
                    self.assertTrue(sandbox.observe()["process"]["running"])

    def test_wrong_action_does_not_resolve_oom_workflow(self) -> None:
        workflow = WORKFLOWS["oom_kill"]
        with VictimRuntime() as sandbox:
            workflow.inject(sandbox)
            sandbox.execute_action("wait_and_observe")
            self.assertFalse(workflow.is_resolved(sandbox))
            self.assertFalse(sandbox.health_check()["healthy"])

    def test_v6_profiles_have_zero_pre_policy_resolved_subfaults(self) -> None:
        for profile_index, profile in enumerate(HARD_SCENARIO_PROFILES):
            for fault_name in WORKFLOWS:
                with self.subTest(profile=profile, workflow=fault_name):
                    sandbox = VictimRuntime()
                    try:
                        workflow, prepared, _ = prepare_v6_scenario(
                            fault_name,
                            hard_sample_seed(42, fault_name, profile_index),
                            profile,
                            sandbox=sandbox,
                        )
                        self.assertEqual(workflow.resolved_subfault_count(prepared), 0)
                        self.assertIs(prepared, sandbox)
                        expert = hard_expert_workflow(fault_name)
                        for action in expert["actions"]:
                            prepared.execute_action(action["action"], action["parameters"])
                        self.assertTrue(workflow.is_resolved(prepared))
                        self.assertTrue(prepared.health_check()["healthy"])
                    finally:
                        sandbox.close()


@unittest.skipUnless(
    _docker_tests_enabled(),
    "set CRASHDIAG_DOCKER_TESTS=1 with a working Docker Engine to run Compose tests",
)
class DockerSandboxTests(unittest.TestCase):
    def test_compose_session_injects_repairs_and_tears_down(self) -> None:
        from crashdiag.sandbox_apps.docker import DockerSandbox

        workflow = WORKFLOWS["bad_env_var"]
        with DockerSandbox() as sandbox:
            session_id = sandbox.session_id
            workflow.inject(sandbox)
            self.assertEqual(workflow.resolved_subfault_count(sandbox), 0)
            self.assertFalse(sandbox.health_check()["healthy"])
            sandbox.execute_action("rollback_env_var")
            self.assertTrue(workflow.is_resolved(sandbox))
            self.assertTrue(sandbox.health_check()["healthy"])
            self.assertTrue(sandbox.observe()["process"]["running"])
        leftover = os.popen(
            f'docker ps -aq --filter label=crashdiag.session={session_id}'
        ).read().strip()
        self.assertEqual(leftover, "")


if __name__ == "__main__":
    unittest.main()
