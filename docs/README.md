# CrashDiag documentation

These documents describe the v5 mechanically verified environment and its
reproducible training/evaluation workflow. Runtime registries in `crashdiag/`
and `training/` are the executable source of truth.

## Index

- [Tasks](tasks.md) — the 52 mechanically verifiable fault families and their
  ordered repair workflows.
- [Actions](actions.md) — the 27-action space, sandbox mutations, and mechanical
  checks.
- [Curriculum v5](curriculum-v5.md) — schema, strict workflow contract,
  partial-credit reward, and version constants.
- [Dataset generation](dataset-generation.md) — deterministic generation,
  validation, splits, and scaling.
- [Notebook workflow](notebook-workflow.md) — the direct Qwen2.5-3B GRPO and
  evaluation pipeline.
- [Migration from v4](migration-v4-to-v5.md) — curriculum changes and rationale.
- [Hugging Face model card](huggingface-model-card.md) — released adapter,
  evaluation, usage, provenance, and limitations.

## Quick facts

| Property | v5 value |
|---|---|
| Fault families | 52 |
| Actions | 27 plus `wait_and_observe` |
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Released adapter | [`Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO`](https://huggingface.co/Indium-AI-Labs/CrashDiag-Qwen2.5-3B-GRPO) |
| Retained experiment | 6,656 train / 832 held-out rows |
| Generator defaults | 260,000 train / 1,300 held-out rows |
| Workflow | one JSON object with an ordered `actions` array |
| Reward | `resolved_subfaults / total_subfaults`; exact success tracked separately |
| Curriculum version | 5 |
