"""Dependency-free end-to-end smoke test for CrashDiag's mock backend."""

from __future__ import annotations

from typing import Any

from crashdiag.faults.modules import BadEnvVar
from crashdiag.orchestrator import Orchestrator
from crashdiag.sandbox_apps.mock import MockSandbox


class ScriptedBlueAgent:
    """A deterministic policy used to prove the loop without a model server."""

    def choose_action(
        self,
        observation: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        del observation, history
        return {
            "action": "rollback_env_var",
            "parameters": {"name": "APP_ENV"},
        }


def main() -> None:
    sandbox = MockSandbox()
    fault = BadEnvVar()
    orchestrator = Orchestrator(
        sandbox=sandbox,
        agent=ScriptedBlueAgent(),
        max_steps=3,
    )

    trajectory = orchestrator.run_episode(fault)

    assert trajectory.resolved, "BadEnvVar was not mechanically resolved"
    assert trajectory.reward == 1.0, "resolved episode did not receive sparse reward"
    assert fault.is_resolved(sandbox), "sandbox state still contains the injected fault"
    assert sandbox.health_check()["healthy"], "application did not return to health"

    print("CrashDiag smoke test")
    print(f"fault={trajectory.fault_name}")
    print(f"resolved={trajectory.resolved}")
    print(f"reward={trajectory.reward}")
    print(f"steps={len(trajectory.steps)}")
    print("PASS: BadEnvVar resolved mechanically")


if __name__ == "__main__":
    main()
