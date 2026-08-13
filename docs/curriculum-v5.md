# CrashDiag v5 Curriculum

The v5 curriculum replaces the single-action v4 curriculum with a multi-action
workflow contract over 52 tasks and a single `Qwen/Qwen2.5-3B-Instruct` policy.

## Version constants

| Constant | Value |
|---|---|
| `HARD_CURRICULUM_VERSION` | `5` |
| `HARD_SCENARIO_SCHEMA_VERSION` | `5` |
| `CRASHDIAG_CURRICULUM` default | `v5` |

## Workflow JSON contract

The policy returns exactly one JSON object:

```json
{
  "actions": [
    {"action": "restart_app", "parameters": {}},
    {"action": "clear_cache", "parameters": {}}
  ]
}
```

Rules:

- `actions` is a non-empty, ordered list of at most 8 entries.
- Each entry is `{"action": "<name>", "parameters": {}}`.
- `parameters` is always `{}` for every repair action.
- Malformed or unknown output falls back to
  `{"actions": [{"action": "wait_and_observe", "parameters": {}}]}`.
- Strict JSON is defined as a top-level object whose only key is `actions`, where each
  element is `{action, parameters}` and the action is in the allowlist.

## Partial-credit reward

For a dataset row with `subfault_count = N`:

1. Reconstruct the exact scenario from `fault_name`, `sample_seed`, and
   `scenario_profile` (anti-tamper prompt check unchanged from v4).
2. Parse the `actions` list and execute each action in order.
3. Count how many of the `N` injected sub-faults are mechanically resolved:
   `resolved_count = fault.resolved_subfault_count(sandbox)`.
4. Reward = `resolved_count / N` in `[0, 1]`.
5. Terminal success = `resolved_count == N`.

The reward is mechanically pure: it never calls a model or reads prose. It only
inspects sandbox state.

## Dataset files

Single consolidated v5 dataset stage:

| file | contents |
|---|---|
| `data/sft_train.jsonl` | SFT train rows with `completion` = `{"actions": [...]}` |
| `data/sft_eval.jsonl` | SFT eval rows |
| `data/grpo_train.jsonl` | answer-free GRPO train prompts |
| `data/grpo_eval.jsonl` | answer-free GRPO eval prompts |
| `data/grpo_summary.json` | schema/curriculum version, row counts, distributions |

The old separate `grpo_hard_train.jsonl` / `grpo_hard_eval.jsonl` hard-v4 files are
removed from the main generation path.

## Row schema

Common fields:

- `fault_name`
- `difficulty`
- `sample_seed`
- `variation_index`
- `subfault_count`
- `scenario_schema_version` (`5`)
- `curriculum_version` (`5`)
- `scenario_profile`
- `prompt` (system + user)
- `metadata` (schema/curriculum versions, split, variation, profile, `mechanically_validated`)

SFT rows add:

- `completion` = `[{"role": "assistant", "content": "{\"actions\":[...]}"}]`

GRPO rows are answer-free and carry no `completion`.

## Dataset sizes

| split | rows per task | total rows |
|---|---|---|
| train | 20,000 | 1,040,000 |
| eval | 2,000 | 104,000 |

Train and eval variations are disjoint by construction (eval `variation_index` starts
at the train count).
