# Migration from v5 to v6

Schema v6 invalidates all v5 generated datasets and benchmark scores. In v5,
the noisy-profile history generator usually called `restart_app()` as an
"unsuccessful remediation." Restarting is an actual repair action for
`oom_kill`. Twelve composite workflows include that sub-fault, so part of those
scenarios could be resolved before the policy received its observation.

## Fix and enforcement

- Decoy remediation is selected by excluding every repair action declared by
  the active workflow, including all composite sub-fault actions.
- Generation asserts zero resolved active sub-faults immediately after fault
  injection and again after adding noisy history.
- Local tests cover every combination of 52 workflows and three profiles.
- HTTP scenario preparation reports an integrity attestation and uses the new
  `/v1/sessions/{session_id}/scenarios/v6` endpoint.
- Training, evaluation notebooks, and the shell pipeline reject dataset rows
  whose scenario or curriculum version is not 6.
- The pipeline checks `/healthz` for schema-v6 support before allocating a GPU.

## Required clean rerun

Do not resume a v5 checkpoint or reuse a v5 dataset run ID. Deploy the current
sandbox, generate a fresh v6 dataset with a new run ID, evaluate the untouched
base model on the v6 held-out split, train from the base model, and evaluate the
new adapter on exactly that same split and decoding configuration. Historical
v5 artifacts may be retained for audit, but they must be labelled invalidated.
