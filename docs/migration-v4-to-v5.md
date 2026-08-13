# Migration from hard-v4 to v5

## What changed

| Dimension | hard-v4 | v5 |
|---|---|---|
| Tasks | 18 single-action faults | 52 multi-action workflows |
| Action output | `{"action": ..., "parameters": {}}` | `{"actions": [{...}, {...}]}` |
| Reward | sparse 0/1 final resolution | partial credit `resolved_subfaults / total_subfaults` |
| Model | 5-model Qwen2.5 sweep | single `Qwen/Qwen2.5-3B-Instruct` |
| Actions | 13 | 27 |
| Train size | 432 | 1,040,000 |
| Eval size | 144 | 104,000 |
| Curriculum version | 4 | 5 |
| Dataset files | separate `grpo_hard_*` | consolidated SFT + GRPO + summary |

## Why

- **Multi-action workflows** reflect real repairs: a failure often has blast-radius
  sub-faults that must each be fixed, in order.
- **52 tasks** give the policy a wider, harder diagnostic surface so evaluation
  retains headroom instead of saturating.
- **Partial credit** gives a dense, mechanically-grounded training signal without
  compromising the "no LLM grader" property.
- **Single 3B model** follows the "smallest model that reaches target accuracy"
  preference and removes the multi-model sweep complexity.
- **1M rows** gives the larger curriculum enough coverage per task.

## What was removed

- The multi-model notebook folders and top-level `eval_all_*`, `sft_all`,
  `grpo_all` notebooks.
- The separate hard-v4 `grpo_hard_train/eval/summary` generation path in the main
  workflow (kept as a legacy shim only where needed for imports).
- Per-model run IDs in `.env.example.*`.
- The old `qwen3_14b` references in README/docstrings that predated the Qwen2.5 sweep.

## Compatibility

- The `crashdiag/agents.py` single-action `parse_action` remains for the legacy
  fault-loop evaluator and tests, but the training reward path uses the new
  `parse_workflow`.
- The `Orchestrator` continues to work with agents returning either a single action
  or an `actions` list.
