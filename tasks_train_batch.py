"""Bounded upload batches for the canonical HUD training taskset."""

from __future__ import annotations

import os

from crashdiag.faults.workflows import WORKFLOWS
from env import diagnose, env
from training.hard_scenarios import hard_sample_seed, profile_for_variation


BATCH_SIZE = 32
BATCH_COUNT = 4
batch_index = int(os.environ.get("CRASHDIAG_HUD_TRAIN_BATCH", "0"))
if not 0 <= batch_index < BATCH_COUNT:
    raise ValueError(f"CRASHDIAG_HUD_TRAIN_BATCH must be in [0, {BATCH_COUNT})")

start_variation = batch_index * BATCH_SIZE
tasks = [
    diagnose(
        fault_name=fault_name,
        sample_seed=hard_sample_seed(42, fault_name, variation_index),
        scenario_profile=profile_for_variation(variation_index),
    )
    for variation_index in range(start_variation, start_variation + BATCH_SIZE)
    for fault_name in WORKFLOWS
]

__all__ = ["env", "tasks"]
