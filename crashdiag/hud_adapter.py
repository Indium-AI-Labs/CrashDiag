"""Dependency-light adapter from CrashDiag episodes to HUD task semantics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from crashdiag.agents import ACTION_SPACE, parse_workflow
from crashdiag.faults.workflows import WorkflowFault
from crashdiag.sandbox_apps.mock import MockSandbox
from training.hard_scenarios import (
    HARD_SCENARIO_PROFILES,
    hard_observation_workflow_messages,
    prepare_v6_scenario,
)


def _strict_workflow(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return False
    if not isinstance(decoded, dict) or set(decoded) != {"actions"}:
        return False
    actions = decoded["actions"]
    if not isinstance(actions, list) or not 1 <= len(actions) <= 8:
        return False
    return all(
        isinstance(entry, dict)
        and set(entry) == {"action", "parameters"}
        and entry["action"] in ACTION_SPACE
        and entry["parameters"] == {}
        for entry in actions
    )


@dataclass(slots=True)
class HudEpisode:
    """One isolated task instance with stateful mechanical grading."""

    workflow: WorkflowFault
    sandbox: MockSandbox
    prompt: str
    _graded: bool = False

    def grade(self, answer: Any) -> float:
        """Execute one strict workflow exactly once and return partial reward."""

        if self._graded:
            raise RuntimeError("HUD episode has already been graded")
        self._graded = True
        if not _strict_workflow(answer):
            return 0.0
        parsed = parse_workflow(answer)
        for entry in parsed["actions"]:
            self.sandbox.execute_action(entry["action"], entry["parameters"])
        return self.workflow.resolved_subfault_count(self.sandbox) / len(
            self.workflow.sub_faults
        )


def create_hud_episode(
    fault_name: str,
    sample_seed: int,
    scenario_profile: str,
) -> HudEpisode:
    """Create a schema-v6 episode and render its policy-facing prompt."""

    if scenario_profile not in HARD_SCENARIO_PROFILES:
        raise ValueError(f"unknown scenario profile: {scenario_profile!r}")
    workflow, sandbox, _ = prepare_v6_scenario(
        fault_name,
        sample_seed,
        scenario_profile,
    )
    messages = hard_observation_workflow_messages(
        sandbox.observe(),
        workflow_name=fault_name,
    )
    prompt = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in messages
    )
    return HudEpisode(workflow=workflow, sandbox=sandbox, prompt=prompt)


__all__ = ["HudEpisode", "create_hud_episode"]
