"""Evaluate a trained CrashDiag policy against mechanically verified faults.

The evaluator gives the policy exactly one action per episode.  Success comes
only from each fault module's ``is_resolved`` check after that action has been
executed against a fresh sandbox; generated prose and model confidence never
participate in scoring.

This module deliberately has no import-time dependency on PyTorch,
Transformers, or PEFT.  Local model loading happens only in :func:`main`, while
the helpers remain usable in lightweight unit tests and on endpoint-only hosts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crashdiag.agents import DEFAULT_SYSTEM_PROMPT, BlueAgent, parse_action
from crashdiag.faults.modules import ALL_FAULTS
from crashdiag.orchestrator import Orchestrator, Trajectory
from crashdiag.sandbox_apps.mock import MockSandbox, SandboxBackend

from .artifacts import (
    ArtifactError,
    add_artifact_arguments,
    preload_env,
    uploader_from_args,
)
from .common import FAULT_NAMES
from .generate_dataset import prepare_scenario, sample_seed


SandboxFactory = Callable[[], SandboxBackend]


class _PreInjectedFault:
    """Let the generic orchestrator consume a deterministically prepared state."""

    def __init__(self, fault: Any) -> None:
        self._fault = fault
        self.name = str(fault.name)
        self.difficulty = str(fault.difficulty)

    def inject(self, instance: SandboxBackend) -> dict[str, bool]:
        return {"pre_injected": True}

    def is_resolved(self, instance: SandboxBackend) -> bool:
        return bool(self._fault.is_resolved(instance))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalTransformersAgent:
    """Small adapter around an already-loaded Transformers causal LM.

    Model and tokenizer construction intentionally live outside this class so
    importing the evaluator does not require ML packages.  A malformed output
    or inference failure becomes the allowlisted ``wait_and_observe`` action,
    matching :class:`crashdiag.agents.BlueAgent`'s defensive behavior.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or temperature < 0
        ):
            raise ValueError("temperature must be a non-negative number")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be a string")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = float(temperature)
        self.system_prompt = system_prompt

    def _messages(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None,
    ) -> list[dict[str, str]]:
        context: dict[str, Any] = {"observation": observation}
        if history:
            context["history"] = list(history)
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": json.dumps(context, sort_keys=True, default=str),
            },
        ]

    def _render_prompt(self, messages: list[dict[str, str]]) -> tuple[str, bool]:
        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            try:
                rendered = apply_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (AttributeError, TypeError, ValueError):
                # Base causal tokenizers often expose the method without having
                # a configured chat template.  A plain role-labelled prompt is
                # still sufficient for models trained on that representation.
                pass
            else:
                if isinstance(rendered, str):
                    return rendered, True

        rendered = "\n".join(
            f"{message['role'].capitalize()}: {message['content']}"
            for message in messages
        )
        return f"{rendered}\nAssistant:", False

    @staticmethod
    def _input_length(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) >= 1:
            return int(shape[-1])
        first = input_ids[0] if input_ids and isinstance(input_ids[0], (list, tuple)) else input_ids
        return len(first)

    def generate_content(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> str:
        """Generate one completion string without interpreting its correctness."""

        prompt, used_chat_template = self._render_prompt(
            self._messages(observation, history)
        )
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=not used_chat_template,
        )
        if not isinstance(encoded, Mapping):
            raise TypeError("tokenizer output must be a mapping")

        device = getattr(self.model, "device", None)
        if device is not None and str(device) != "meta":
            move_batch = getattr(encoded, "to", None)
            if callable(move_batch):
                encoded = move_batch(device)
            else:
                encoded = {
                    key: value.to(device) if callable(getattr(value, "to", None)) else value
                    for key, value in encoded.items()
                }

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id

        generated = self.model.generate(**dict(encoded), **generation_kwargs)
        sequences = getattr(generated, "sequences", generated)
        input_length = self._input_length(encoded["input_ids"])
        completion_tokens = sequences[0][input_length:]
        return str(
            self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        ).strip()

    def choose_action(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return one schema-validated action, or a conservative no-op."""

        try:
            return parse_action(self.generate_content(observation, history))
        except Exception:
            return parse_action(None)

    def act(
        self,
        observation: Mapping[str, Any] | Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.choose_action(observation, history)


def sandbox_factory(
    *,
    sandbox_url: str | None = None,
    api_token: str | None = None,
    timeout: float = 15.0,
) -> SandboxFactory:
    """Build a factory that creates one isolated sandbox per episode."""

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("sandbox timeout must be positive")
    normalized_url = (sandbox_url or "").strip()
    if not normalized_url:
        return MockSandbox

    # This standard-library client is cheap to import, but keeping it in the
    # remote branch also makes the local evaluator's dependency boundary clear.
    from crashdiag.sandbox_apps.http import HttpSandbox

    def create_remote() -> SandboxBackend:
        return HttpSandbox(
            normalized_url,
            api_token=api_token,
            timeout=float(timeout),
        )

    return create_remote


def _close_sandbox(sandbox: SandboxBackend) -> None:
    close = getattr(sandbox, "close", None)
    if callable(close):
        for _ in range(2):
            try:
                close()
                return
            except Exception:
                # Cleanup cannot alter an already measured result. HttpSandbox
                # keeps failed cleanup retryable; TTL remains the final backstop.
                continue


def summarize_trajectories(
    trajectories: Sequence[Trajectory],
    faults: Sequence[Any],
    *,
    episodes_per_fault: int,
) -> dict[str, Any]:
    """Create success-rate metrics from terminal mechanical resolution flags."""

    per_fault: dict[str, dict[str, Any]] = {}
    for fault in faults:
        name = str(getattr(fault, "name", fault.__class__.__name__))
        matching = [item for item in trajectories if item.fault_name == name]
        resolved = sum(item.resolved is True for item in matching)
        count = len(matching)
        per_fault[name] = {
            "difficulty": str(getattr(fault, "difficulty", "unknown")),
            "episodes": count,
            "resolved": resolved,
            "success_rate": resolved / count if count else 0.0,
        }

    resolved_total = sum(item.resolved is True for item in trajectories)
    total = len(trajectories)
    return {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "scoring": "mechanical_fault_resolution",
        "episodes_per_fault": episodes_per_fault,
        "summary": {
            "total_episodes": total,
            "resolved_episodes": resolved_total,
            "success_rate": resolved_total / total if total else 0.0,
        },
        "per_fault": per_fault,
        "trajectories": [trajectory.to_dict() for trajectory in trajectories],
    }


def run_evaluation(
    agent: Any,
    *,
    episodes_per_fault: int = 1,
    make_sandbox: SandboxFactory = MockSandbox,
    faults: Iterable[Any] = ALL_FAULTS,
    seed: int = 42,
) -> dict[str, Any]:
    """Run one-action episodes for every fault and return a JSON-safe report.

    A fresh sandbox is constructed for every repetition.  The orchestrator's
    one-step limit ensures exactly one generated action is executed, and its
    verifier derives ``resolved`` from live sandbox state.
    """

    if (
        isinstance(episodes_per_fault, bool)
        or not isinstance(episodes_per_fault, int)
        or episodes_per_fault <= 0
    ):
        raise ValueError("episodes_per_fault must be a positive integer")
    if not callable(make_sandbox):
        raise TypeError("make_sandbox must be callable")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    selected_faults = tuple(faults)
    if not selected_faults:
        raise ValueError("at least one fault is required")

    trajectories: list[Trajectory] = []
    for fault in selected_faults:
        fault_name = str(getattr(fault, "name", fault.__class__.__name__))
        for episode_index in range(episodes_per_fault):
            sandbox = make_sandbox()
            try:
                scenario_seed: int | None = None
                episode_fault = fault
                if fault_name in FAULT_NAMES:
                    variation_index = 1_000_000 + episode_index
                    scenario_seed = sample_seed(
                        seed,
                        fault_name,
                        variation_index,
                    )
                    prepared_fault, _, _ = prepare_scenario(
                        fault_name,
                        scenario_seed,
                        sandbox=sandbox,
                    )
                    episode_fault = _PreInjectedFault(prepared_fault)
                trajectory = Orchestrator(
                    sandbox=sandbox,
                    agent=agent,
                    max_steps=1,
                ).run_episode(episode_fault)
                trajectory.metadata.update(
                    {
                        "episode_index": episode_index,
                        "backend": type(sandbox).__name__,
                        "action_limit": 1,
                        "sample_seed": scenario_seed,
                        "scenario_prepared": scenario_seed is not None,
                    }
                )
                trajectories.append(trajectory)
            finally:
                _close_sandbox(sandbox)

    report = summarize_trajectories(
        trajectories,
        selected_faults,
        episodes_per_fault=episodes_per_fault,
    )
    report["evaluation_seed"] = seed
    return report


# A readable alias for callers that think in terms of the agent rather than the
# whole evaluation job.
evaluate_agent = run_evaluation


def save_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    """Write a strict JSON report, creating its parent directory if needed."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def format_report(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable view of the mechanical metrics."""

    summary = report["summary"]
    lines = [
        "CrashDiag mechanical evaluation",
        (
            f"overall: {summary['resolved_episodes']}/{summary['total_episodes']} "
            f"({float(summary['success_rate']):.1%})"
        ),
    ]
    for name, metrics in report["per_fault"].items():
        lines.append(
            f"{name}: {metrics['resolved']}/{metrics['episodes']} "
            f"({float(metrics['success_rate']):.1%})"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="outputs/grpo",
        help="local model/adapter path or model name served by --base-url",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API root; skips all local ML imports",
    )
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--sandbox-url", default=os.environ.get("CRASHDIAG_SANDBOX_URL"))
    parser.add_argument(
        "--sandbox-token",
        default=(
            os.environ.get("CRASHDIAG_SANDBOX_TOKEN")
            or os.environ.get("CRASHDIAG_API_TOKEN")
        ),
    )
    parser.add_argument(
        "--episodes-per-fault",
        "--episodes",
        dest="episodes_per_fault",
        type=int,
        default=1,
    )
    parser.add_argument("--output", default="outputs/evaluation.json")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--sandbox-timeout", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adapter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="force (or disable) loading --model as a PEFT adapter",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value; use 'none' to omit it",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    add_artifact_arguments(parser)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = {
        "episodes_per_fault": args.episodes_per_fault,
        "max_new_tokens": args.max_new_tokens,
        "timeout": args.timeout,
        "sandbox_timeout": args.sandbox_timeout,
    }
    invalid = [
        name
        for name, value in positive.items()
        if isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ]
    if invalid:
        raise SystemExit(f"these arguments must be positive: {', '.join(invalid)}")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")


def _dtype_for_precision(torch: Any, precision: str) -> Any:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    return "auto"


def main(argv: list[str] | None = None) -> None:
    preload_env(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    try:
        uploader = uploader_from_args(args)
        if uploader is not None:
            uploader.start_stage(
                "evaluation",
                {
                    "model": args.model,
                    "episodes_per_fault": args.episodes_per_fault,
                    "remote_sandbox": bool(args.sandbox_url),
                },
            )
    except ArtifactError as exc:
        parser.exit(2, f"evaluation artifact error: {exc}\n")

    if args.base_url:
        agent: Any = BlueAgent(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=args.timeout,
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
        )
    else:
        # Heavyweight imports and weight loading are deliberately confined to
        # main, so importing/testing this module needs only the standard library.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise SystemExit(
                "Local evaluation dependencies are missing. Install them with "
                "`pip install -e .[train]`, or pass --base-url."
            ) from exc

        model_path = Path(args.model)
        detected_adapter = model_path.is_dir() and (
            model_path / "adapter_config.json"
        ).is_file()
        is_adapter = detected_adapter if args.adapter is None else args.adapter
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": args.trust_remote_code,
            "dtype": _dtype_for_precision(torch, args.precision),
        }
        if args.device_map.lower() != "none":
            load_kwargs["device_map"] = args.device_map

        set_seed(args.seed)
        tokenizer_source = args.model
        if is_adapter:
            try:
                from peft import AutoPeftModelForCausalLM, PeftConfig
            except ImportError as exc:
                raise SystemExit(
                    "PEFT is required to evaluate an adapter. Install "
                    "`pip install -e .[train]`."
                ) from exc
            model = AutoPeftModelForCausalLM.from_pretrained(
                args.model, **load_kwargs
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=args.trust_remote_code,
                )
            except OSError:
                adapter_config = PeftConfig.from_pretrained(args.model)
                tokenizer_source = adapter_config.base_model_name_or_path
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_source,
                    trust_remote_code=args.trust_remote_code,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=args.trust_remote_code,
            )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise SystemExit(
                    "tokenizer must define either a pad token or an EOS token"
                )
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        agent = LocalTransformersAgent(
            model,
            tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    report = run_evaluation(
        agent,
        episodes_per_fault=args.episodes_per_fault,
        make_sandbox=sandbox_factory(
            sandbox_url=args.sandbox_url,
            api_token=args.sandbox_token,
            timeout=args.sandbox_timeout,
        ),
        seed=args.seed,
    )
    output_path = save_report(report, args.output)
    if uploader is not None:
        try:
            uploader.upload_files(
                [output_path],
                "evaluation",
                metadata={
                    "model": args.model,
                    "episodes_per_fault": args.episodes_per_fault,
                    "summary": report["summary"],
                    "scoring": "mechanical_fault_resolution",
                },
            )
        except ArtifactError as exc:
            parser.exit(2, f"evaluation artifact error: {exc}\n")
    print(format_report(report))
    print(f"report: {output_path}")
    if uploader is not None:
        print(f"artifacts: {uploader.remote_uri('evaluation')}")


if __name__ == "__main__":
    main()


__all__ = [
    "LocalTransformersAgent",
    "build_parser",
    "evaluate_agent",
    "format_report",
    "run_evaluation",
    "sandbox_factory",
    "save_report",
    "summarize_trajectories",
]
