"""Local inference helpers shared by calibration and exact-dataset evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import resolve_precision


def load_local_policy(
    model_path: str | Path,
    *,
    precision: str = "auto",
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """Load a base model or PEFT adapter without import-time ML dependencies."""

    try:
        import torch
        from peft import AutoPeftModelForCausalLM, PeftConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install CrashDiag training dependencies before inference") from exc

    source = str(model_path)
    adapter = Path(model_path).is_dir() and (Path(model_path) / "adapter_config.json").is_file()
    bf16, fp16 = resolve_precision(torch, precision)
    dtype = torch.bfloat16 if bf16 else torch.float16 if fp16 else torch.float32
    tokenizer_source = source
    if adapter:
        config = PeftConfig.from_pretrained(source)
        tokenizer_source = source if (Path(source) / "tokenizer_config.json").is_file() else config.base_model_name_or_path
        model = AutoPeftModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def generate_from_messages(
    model: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    num_return_sequences: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    max_new_tokens: int = 96,
) -> list[str]:
    """Generate raw completions for the exact serialized dataset prompt."""

    if num_return_sequences < 1:
        raise ValueError("num_return_sequences must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be finite and in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k cannot be negative")
    rendered = tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    device = getattr(model, "device", None)
    move = getattr(encoded, "to", None)
    if device is not None and callable(move):
        encoded = move(device)
    generation: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        generation["temperature"] = temperature
        # Pass these explicitly so a model's bundled GenerationConfig cannot
        # silently narrow the candidate token set during GRPO calibration.
        generation["top_p"] = top_p
        generation["top_k"] = top_k
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for local inference") from exc
    with torch.inference_mode():
        output = model.generate(**dict(encoded), **generation)
    sequences = getattr(output, "sequences", output)
    input_length = int(encoded["input_ids"].shape[-1])
    return [
        str(tokenizer.decode(sequence[input_length:], skip_special_tokens=True)).strip()
        for sequence in sequences
    ]


__all__ = ["generate_from_messages", "load_local_policy"]
