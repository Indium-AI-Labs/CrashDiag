"""CrashDiag schema-v6 environment for the HUD platform."""

from hud.environment import Environment

from crashdiag.hud_adapter import create_hud_episode


env = Environment(name="crashdiag", version="0.4.0")


@env.template(
    id="diagnose",
    description=(
        "Diagnose one deterministic infrastructure incident from incomplete "
        "telemetry and return a strict ordered JSON remediation workflow. "
        "Reward is computed from mechanical post-action state, with partial "
        "credit for resolved sub-faults and no LLM judge."
    ),
)
async def diagnose(
    fault_name: str,
    sample_seed: int,
    scenario_profile: str,
):
    """Diagnose one hidden fault and mechanically grade the repair workflow."""

    episode = create_hud_episode(fault_name, sample_seed, scenario_profile)
    answer = yield episode.prompt
    yield episode.grade(answer)
