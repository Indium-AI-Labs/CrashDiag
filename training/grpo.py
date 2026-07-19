"""GRPO training with rewards computed from executable sandbox state.

The reward function in this module never calls a model.  For every generated
completion it creates a fresh local or remote sandbox, injects the dataset's
fault, parses one bounded action, executes it, and asks ``CrashDiagVerifier``
to score the resulting state.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from crashdiag.agents import ACTION_SPACE, parse_action
from crashdiag.sandbox_apps.mock import MockSandbox, SandboxBackend
from crashdiag.verifier import CrashDiagVerifier

from .artifacts import (
    ArtifactError,
    add_artifact_arguments,
    make_checkpoint_upload_callback,
    preload_env,
    process_is_world_zero,
    uploader_from_args,
)
from .common import FAULT_NAMES, completion_text, observation_messages, resolve_precision
from .generate_dataset import prepare_scenario
from .hard_scenarios import (
    HARD_SCENARIO_PROFILES,
    HARD_SCENARIO_SCHEMA_VERSION,
    hard_observation_messages,
    prepare_hard_scenario,
)
from .reporting import ReportBundle, generate_trainer_report

_SANDBOX_URL = os.environ.get("CRASHDIAG_SANDBOX_URL", "").strip()
_SANDBOX_TOKEN = os.environ.get("CRASHDIAG_API_TOKEN") or os.environ.get(
    "CRASHDIAG_SANDBOX_TOKEN"
)
_SANDBOX_TIMEOUT = 15.0


def configure_reward_backend(
    *,
    sandbox_url: str | None = None,
    api_token: str | None = None,
    timeout: float = 15.0,
) -> None:
    """Configure the backend used by subsequent reward-function calls."""

    if timeout <= 0:
        raise ValueError("sandbox timeout must be positive")
    global _SANDBOX_URL, _SANDBOX_TOKEN, _SANDBOX_TIMEOUT
    _SANDBOX_URL = (sandbox_url or "").strip()
    _SANDBOX_TOKEN = api_token
    _SANDBOX_TIMEOUT = float(timeout)


def _new_sandbox() -> SandboxBackend:
    if not _SANDBOX_URL:
        return MockSandbox()
    from crashdiag.sandbox_apps.http import HttpSandbox

    return HttpSandbox(
        _SANDBOX_URL,
        api_token=_SANDBOX_TOKEN,
        timeout=_SANDBOX_TIMEOUT,
    )


def _close_sandbox(sandbox: SandboxBackend) -> None:
    close = getattr(sandbox, "close", None)
    if callable(close):
        for _ in range(2):
            try:
                close()
                return
            except Exception:
                # HttpSandbox retains its session ID after a transient cleanup
                # failure, so one immediate retry avoids needless TTL leaks.
                continue


def _broadcast_column(
    value: Any,
    count: int,
    name: str,
    *,
    scalar_types: tuple[type, ...],
) -> list[Any]:
    if isinstance(value, scalar_types) and not isinstance(value, bool):
        values = [value]
    else:
        values = list(value)
    if len(values) == 1 and count > 1:
        values *= count
    if len(values) != count:
        raise ValueError(
            f"{name} and completions must have equal lengths "
            f"({len(values)} != {count})"
        )
    return values


def _strict_action_json(value: Any) -> bool:
    try:
        decoded = json.loads(completion_text(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(decoded, dict)
        and set(decoded).issubset({"action", "parameters"})
        and decoded.get("action") in ACTION_SPACE
        and isinstance(decoded.get("parameters", {}), dict)
    )


def mechanical_reward(
    completions: list[Any],
    fault_name: list[str] | tuple[str, ...] | str,
    sample_seed: list[int] | tuple[int, ...] | int | None = None,
    prompts: list[Any] | tuple[Any, ...] | None = None,
    scenario_schema_version: list[int] | tuple[int, ...] | int | None = None,
    scenario_profile: list[str] | tuple[str, ...] | str | None = None,
    log_extra: Any | None = None,
    log_metric: Any | None = None,
    **_: Any,
) -> list[float]:
    """Return sparse rewards after executing each generated action.

    ``fault_name`` is a top-level column in the generated GRPO JSONL dataset,
    so TRL supplies the correct fault for every sampled prompt and every member
    of its generation group.
    """

    count = len(completions)
    names = _broadcast_column(
        fault_name, count, "fault_name", scalar_types=(str,)
    )
    if sample_seed is None:
        raise ValueError("sample_seed is required for exact scenario replay")
    seeds = _broadcast_column(
        sample_seed, count, "sample_seed", scalar_types=(int,)
    )
    if prompts is None:
        raise ValueError("prompts are required to verify exact scenario replay")
    prompt_values = _broadcast_column(
        prompts, count, "prompts", scalar_types=()
    )
    versions = (
        [1] * count
        if scenario_schema_version is None
        else _broadcast_column(
            scenario_schema_version,
            count,
            "scenario_schema_version",
            scalar_types=(int,),
        )
    )
    profiles = (
        [None] * count
        if scenario_profile is None
        else _broadcast_column(
            scenario_profile,
            count,
            "scenario_profile",
            scalar_types=(str,),
        )
    )

    rewards: list[float] = []
    parsed_actions: list[str] = []
    resolved_values: list[bool] = []
    backend_errors: list[bool] = []
    strict_json_values: list[bool] = []
    verifier = CrashDiagVerifier()

    for completion, name, scenario_seed, supplied_prompt, version, profile in zip(
        completions, names, seeds, prompt_values, versions, profiles, strict=True
    ):
        sandbox: SandboxBackend | None = None
        action_name = "wait_and_observe"
        resolved = False
        backend_error = False
        strict_json = _strict_action_json(completion)
        try:
            if isinstance(scenario_seed, bool) or not isinstance(
                scenario_seed, int
            ):
                raise TypeError("sample_seed values must be integers")
            if isinstance(version, bool) or not isinstance(version, int):
                raise TypeError("scenario_schema_version values must be integers")
            sandbox = _new_sandbox()
            if version == 1:
                fault, _, _ = prepare_scenario(
                    str(name),
                    scenario_seed,
                    sandbox=sandbox,
                )
                expected_prompt = observation_messages(sandbox.observe())
            elif version == HARD_SCENARIO_SCHEMA_VERSION:
                if not isinstance(profile, str):
                    raise TypeError("schema-v2 scenarios require scenario_profile")
                fault, _, _ = prepare_hard_scenario(
                    str(name),
                    scenario_seed,
                    profile,
                    sandbox=sandbox,
                )
                expected_prompt = hard_observation_messages(sandbox.observe())
            else:
                raise ValueError(f"unsupported scenario schema version: {version}")
            expected_json = json.dumps(
                expected_prompt,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            supplied_json = json.dumps(
                supplied_prompt,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            if supplied_json != expected_json:
                raise ValueError("prompt does not match the reconstructed scenario")
            parsed = parse_action(completion_text(completion))
            action_name = parsed["action"]
            sandbox.execute_action(action_name, parsed["parameters"])
            resolved = verifier.is_resolved(fault, sandbox)
            reward = verifier.reward_for_resolution(resolved, sandbox)
        except Exception:
            # Infrastructure/parser/action failures are failed rollouts, never
            # evidence that the application was repaired.
            reward = 0.0
            backend_error = True
        finally:
            if sandbox is not None:
                _close_sandbox(sandbox)

        rewards.append(float(reward))
        parsed_actions.append(action_name)
        resolved_values.append(resolved)
        backend_errors.append(backend_error)
        strict_json_values.append(strict_json)

    if callable(log_extra):
        log_extra("crashdiag_action", parsed_actions)
        log_extra("crashdiag_resolved", resolved_values)
        log_extra("crashdiag_backend_error", backend_errors)
        log_extra("crashdiag_strict_json", strict_json_values)
    if callable(log_metric) and rewards:
        log_metric("crashdiag/success_rate", sum(rewards) / len(rewards))
        log_metric(
            "crashdiag/backend_error_rate",
            sum(backend_errors) / len(backend_errors),
        )
        log_metric(
            "crashdiag/strict_json_rate",
            sum(strict_json_values) / len(strict_json_values),
        )
    return rewards


def validate_grpo_dataset(dataset: Any, label: str = "GRPO dataset") -> None:
    """Fail closed on unknown faults or non-replayable scenario schemas."""

    required_columns = {"prompt", "fault_name", "sample_seed"}
    columns = set(dataset.column_names)
    missing = required_columns.difference(columns)
    if missing:
        raise SystemExit(f"{label} is missing columns: {sorted(missing)}")
    unknown_faults = sorted(
        {str(name) for name in dataset["fault_name"] if str(name) not in FAULT_NAMES}
    )
    if unknown_faults:
        raise SystemExit(f"{label} contains unknown faults: {unknown_faults}")
    versions = (
        [1] * len(dataset)
        if "scenario_schema_version" not in columns
        else list(dataset["scenario_schema_version"])
    )
    unknown_versions = sorted(
        {
            value
            for value in versions
            if isinstance(value, bool)
            or not isinstance(value, int)
            or value not in {1, HARD_SCENARIO_SCHEMA_VERSION}
        },
        key=str,
    )
    if unknown_versions:
        raise SystemExit(
            f"{label} contains unsupported schema versions: {unknown_versions}"
        )
    if HARD_SCENARIO_SCHEMA_VERSION in versions:
        if "scenario_profile" not in columns:
            raise SystemExit(f"{label} schema-v2 rows require scenario_profile")
        profiles = list(dataset["scenario_profile"])
        invalid_profiles = sorted(
            {
                str(profile)
                for version, profile in zip(versions, profiles, strict=True)
                if version == HARD_SCENARIO_SCHEMA_VERSION
                and profile not in HARD_SCENARIO_PROFILES
            }
        )
        if invalid_profiles:
            raise SystemExit(
                f"{label} contains unsupported scenario profiles: {invalid_profiles}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="outputs/sft")
    parser.add_argument("--train-file", default="data/grpo_train.jsonl")
    parser.add_argument("--eval-file", default="data/grpo_eval.jsonl")
    parser.add_argument("--output-dir", default="outputs/grpo")
    parser.add_argument(
        "--sandbox-url",
        default=os.environ.get("CRASHDIAG_SANDBOX_URL", "").strip(),
    )
    parser.add_argument(
        "--sandbox-token",
        default=(
            os.environ.get("CRASHDIAG_API_TOKEN")
            or os.environ.get("CRASHDIAG_SANDBOX_TOKEN")
        ),
    )
    parser.add_argument("--sandbox-timeout", type=float, default=15.0)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm-server-base-url", default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--artifact-stage",
        default="grpo",
        help="bucket stage name; use grpo-smoke for a preliminary job",
    )
    add_artifact_arguments(parser)
    return parser


def _validate_positive(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "temperature": args.temperature,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "sandbox_timeout": args.sandbox_timeout,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise SystemExit(f"these arguments must be positive: {', '.join(invalid)}")
    if args.num_generations < 2:
        raise SystemExit("--num-generations must be at least 2 for GRPO advantages")
    try:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    except ValueError:
        world_size = 1
    generation_batch = (
        args.batch_size * world_size * args.gradient_accumulation_steps
    )
    if generation_batch % args.num_generations != 0:
        raise SystemExit(
            "effective generation batch "
            "(--batch-size * WORLD_SIZE * --gradient-accumulation-steps) "
            "must be divisible by --num-generations"
        )
    evaluation_batch = args.batch_size * world_size
    if args.eval_file and evaluation_batch % args.num_generations != 0:
        raise SystemExit(
            "global evaluation batch (--batch-size * WORLD_SIZE) must be "
            "divisible by --num-generations"
        )


def main(argv: list[str] | None = None) -> None:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_positive(args)

    try:
        uploader = uploader_from_args(args)
        if uploader is not None and process_is_world_zero():
            uploader.start_stage(
                args.artifact_stage,
                {
                    "model": args.model,
                    "train_file": args.train_file,
                    "output_dir": args.output_dir,
                    "remote_sandbox": bool(args.sandbox_url),
                },
            )
    except ArtifactError as exc:
        parser.exit(2, f"GRPO artifact error: {exc}\n")

    train_path = Path(args.train_file)
    if not train_path.is_file():
        raise SystemExit(
            f"GRPO dataset not found: {train_path}. Run `python -m training.generate_dataset` first."
        )
    eval_path = Path(args.eval_file) if args.eval_file else None
    if eval_path is not None and not eval_path.is_file():
        raise SystemExit(f"evaluation dataset not found: {eval_path}")

    configure_reward_backend(
        sandbox_url=args.sandbox_url,
        api_token=args.sandbox_token,
        timeout=args.sandbox_timeout,
    )
    if args.sandbox_url:
        probe = _new_sandbox()
        try:
            probe.observe()
        finally:
            _close_sandbox(probe)

    try:
        import torch
        from datasets import load_dataset
        from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install them with "
            "`pip install -e .[train]`."
        ) from exc

    train_dataset = load_dataset("json", data_files=str(train_path), split="train")
    validate_grpo_dataset(train_dataset)
    eval_dataset = (
        load_dataset("json", data_files=str(eval_path), split="train")
        if eval_path is not None
        else None
    )
    if eval_dataset is not None:
        validate_grpo_dataset(eval_dataset, "GRPO evaluation dataset")

    bf16, fp16 = resolve_precision(torch, args.precision)
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    adapter_checkpoint = Path(args.model).is_dir() and (
        Path(args.model) / "adapter_config.json"
    ).is_file()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
    except OSError:
        if not adapter_checkpoint:
            raise
        adapter_config = PeftConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_config.base_model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if adapter_checkpoint:
        model: Any = AutoPeftModelForCausalLM.from_pretrained(
            args.model,
            is_trainable=True,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
        )
        peft_config = None
        model_init_kwargs = None
    else:
        model = args.model
        peft_config = (
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
            if args.lora
            else None
        )
        model_init_kwargs = {"dtype": dtype}

    eval_enabled = eval_dataset is not None
    config = GRPOConfig(
        output_dir=args.output_dir,
        model_init_kwargs=model_init_kwargs,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.beta,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        logging_first_step=True,
        log_completions=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=args.eval_steps if eval_enabled else None,
        report_to=args.report_to,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        warmup_ratio=0.03,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        chat_template_kwargs={"enable_thinking": False},
    )
    callback = make_checkpoint_upload_callback(uploader, args.artifact_stage)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=mechanical_reward,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[callback] if callback is not None else None,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    eval_metrics = trainer.evaluate() if eval_enabled else None
    report_bundle: ReportBundle | None = None
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        trainer.save_state()
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        if eval_metrics is not None:
            trainer.log_metrics("eval", eval_metrics)
            trainer.save_metrics("eval", eval_metrics)
        output_path = Path(args.output_dir)
        report_bundle = generate_trainer_report(
            output_path / "trainer_state.json",
            output_path / "reports",
            kind="grpo",
            title=f"CrashDiag {args.artifact_stage} training metrics",
        )
        print(f"GRPO report: {report_bundle.summary_path}")
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero() and uploader is not None:
        uploader.upload_directory(
            args.output_dir,
            args.artifact_stage,
            metadata={
                "model": args.model,
                "train_metrics": result.metrics,
                "eval_metrics": eval_metrics,
                "remote_sandbox": bool(args.sandbox_url),
                "report": report_bundle.summary if report_bundle is not None else None,
            },
        )
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
