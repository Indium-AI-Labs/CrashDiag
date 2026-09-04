"""Public deterministic CrashDiag taskset for HUD."""

from crashdiag.faults.workflows import WORKFLOWS
from env import diagnose, env
from training.generate_dataset import EVAL_START_VARIATION
from training.hard_scenarios import hard_sample_seed, profile_for_variation


TASKS_PER_FAULT = 16

tasks = [
    diagnose(
        fault_name=fault_name,
        sample_seed=hard_sample_seed(42, fault_name, variation_index),
        scenario_profile=profile_for_variation(variation_index),
    )
    for variation_index in range(
        EVAL_START_VARIATION,
        EVAL_START_VARIATION + TASKS_PER_FAULT,
    )
    for fault_name in WORKFLOWS
]

__all__ = ["env", "tasks"]
