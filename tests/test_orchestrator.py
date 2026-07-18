from __future__ import annotations

import json
import unittest
from typing import Any

from crashdiag.faults.modules import ALL_FAULTS
from crashdiag.orchestrator import Orchestrator, Trajectory
from crashdiag.sandbox_apps.mock import MockSandbox
from crashdiag.verifier import RewardConfig


class _FaultAwareScriptedAgent:
    ACTION_BY_FAILURE = {
        "process": "restart_app",
        "environment": "rollback_env_var",
        "database": "rollback_env_var",
        "dependencies": "fix_dependency",
        "disk": "clear_disk",
        "port_proxy": "fix_port_config",
    }

    def choose_action(
        self,
        observation: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del history
        failure = observation["health"]["failures"][0]
        return {"action": self.ACTION_BY_FAILURE[failure], "parameters": {}}


class _EmptyObservationSandbox:
    def __init__(self) -> None:
        self.observe_calls = 0

    def observe(self) -> dict[str, Any]:
        self.observe_calls += 1
        return {}


class _ImmediatelyResolvedFault:
    name = "already_resolved"
    difficulty = "easy"

    def inject(self, instance: Any) -> None:
        del instance

    def is_resolved(self, instance: Any) -> bool:
        del instance
        return True


class _ChangingProbeFault(_ImmediatelyResolvedFault):
    name = "changing_probe"

    def __init__(self) -> None:
        self.check_count = 0

    def is_resolved(self, instance: Any) -> bool:
        del instance
        self.check_count += 1
        return self.check_count == 2


class _NoOpAgent:
    def choose_action(self, observation: Any, history: Any = None) -> dict[str, Any]:
        del observation, history
        return {"action": "wait_and_observe", "parameters": {}}


class _EmptyActionSandbox(_EmptyObservationSandbox):
    def execute_action(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]:
        del action, parameters
        return {}


class _BrokenStateRewardVerifier:
    config = RewardConfig(unresolved_reward=-2.0)

    def reward_for_resolution(self, resolved: bool, instance: Any) -> float:
        del resolved, instance
        raise RuntimeError("verifier unavailable")


class _BrokenStringValue:
    def __str__(self) -> str:
        raise RuntimeError("cannot stringify")


class OrchestratorTests(unittest.TestCase):
    def test_each_fault_completes_a_real_episode(self) -> None:
        for fault in ALL_FAULTS:
            with self.subTest(fault=fault.name):
                sandbox = MockSandbox()
                trajectory = Orchestrator(
                    sandbox,
                    _FaultAwareScriptedAgent(),
                    max_steps=2,
                ).run_episode(fault)

                self.assertTrue(trajectory.resolved)
                self.assertEqual(trajectory.reward, 1.0)
                self.assertEqual(len(trajectory.steps), 1)
                self.assertTrue(trajectory.final_observation["healthy"])
                self.assertEqual(json.loads(trajectory.to_json()), trajectory.to_dict())

    def test_batch_runs_all_faults_in_order(self) -> None:
        trajectories = Orchestrator(
            MockSandbox(),
            _FaultAwareScriptedAgent(),
            max_steps=2,
        ).run_batch(ALL_FAULTS)

        self.assertEqual(
            [trajectory.fault_name for trajectory in trajectories],
            [fault.name for fault in ALL_FAULTS],
        )
        self.assertTrue(all(trajectory.resolved for trajectory in trajectories))

    def test_terminal_reward_reuses_the_terminal_resolution_check(self) -> None:
        fault = _ChangingProbeFault()
        trajectory = Orchestrator(
            _EmptyActionSandbox(),
            _NoOpAgent(),
            max_steps=1,
        ).run_episode(fault)

        self.assertEqual(fault.check_count, 2)
        self.assertTrue(trajectory.resolved)
        self.assertEqual(trajectory.reward, 1.0)

    def test_empty_final_observation_is_not_read_twice(self) -> None:
        sandbox = _EmptyObservationSandbox()
        trajectory = Orchestrator(sandbox, _NoOpAgent()).run_episode(
            _ImmediatelyResolvedFault()
        )

        self.assertEqual(trajectory.final_observation, {})
        self.assertEqual(sandbox.observe_calls, 1)

    def test_reward_error_uses_configured_unresolved_reward(self) -> None:
        trajectory = Orchestrator(
            _EmptyObservationSandbox(),
            _NoOpAgent(),
            verifier=_BrokenStateRewardVerifier(),
        ).run_episode(_ImmediatelyResolvedFault())

        self.assertTrue(trajectory.resolved)
        self.assertEqual(trajectory.reward, -2.0)
        self.assertIn("reward verification failed", trajectory.error or "")

    def test_trajectory_serializes_values_with_broken_string_conversion(self) -> None:
        trajectory = Trajectory(
            fault_name="serialization_test",
            initial_observation=_BrokenStringValue(),
        )

        payload = json.loads(trajectory.to_json())
        self.assertEqual(
            payload["initial_observation"],
            "<unserializable _BrokenStringValue>",
        )


if __name__ == "__main__":
    unittest.main()
