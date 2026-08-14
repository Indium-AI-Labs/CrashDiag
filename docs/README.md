# CrashDiag Documentation

This directory is the source of truth for the v5 curriculum. The code registries in
`crashdiag/` and `training/` are generated to match these documents.

## Index

- [tasks.md](tasks.md) — the 52 mechanically-verifiable tasks, each resolved by an ordered multi-action workflow.
- [actions.md](actions.md) — the 27-action space, the sandbox state each action mutates, and its mechanical verification.
- [curriculum-v5.md](curriculum-v5.md) — v5 schema, workflow JSON contract, partial-credit reward, dataset sizes, version constants.
- [dataset-generation.md](dataset-generation.md) — how to generate the v5 datasets, including the balanced 260k-row train set.
- [notebook-workflow.md](notebook-workflow.md) — the single `qwen2.5_3b` pipeline, env vars, stage handoff, and Kaggle secrets.
- [migration-v4-to-v5.md](migration-v4-to-v5.md) — what changed from the hard-v4 curriculum and why.

## Quick facts

| Property | v5 value |
|---|---|
| Tasks | 52 |
| Actions | 27 (plus `wait_and_observe` fallback) |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Train rows | 260,000 (5,000 per task) |
| Eval rows | 1,300 (25 per task) |
| Workflow | one JSON reply with an ordered `actions` array |
| Reward | partial credit (`resolved_subfaults / total_subfaults`) |
| Curriculum version | `5` |
