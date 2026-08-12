"""Memory-safe preparation for 4-bit QLoRA policies."""

from __future__ import annotations

from typing import Any


def cast_trainable_parameters_to_fp32(model: Any) -> int:
    """Cast only trainable FP16/BF16 parameters to FP32 for stable AMP.

    PEFT adapters can inherit Qwen3's BF16 storage dtype even when Accelerate
    is launched in FP16 mode. PyTorch's FP16 GradScaler cannot unscale BF16
    gradients. The frozen 14B base stays quantized; only the small adapter is
    converted, so this has negligible memory cost.
    """

    converted = 0
    for parameter in model.parameters():
        data = getattr(parameter, "data", None)
        dtype_name = str(getattr(data, "dtype", "")).lower()
        if (
            getattr(parameter, "requires_grad", False)
            and dtype_name in {"float16", "bfloat16", "torch.float16", "torch.bfloat16"}
        ):
            parameter.data = data.float()
            converted += 1
    return converted


def prepare_4bit_qlora_model(
    model: Any,
    *,
    gradient_checkpointing: bool = True,
) -> Any:
    """Freeze a quantized base model without upcasting its weights to FP32.

    ``peft.prepare_model_for_kbit_training`` is a good default for smaller
    models, but it intentionally casts every non-quantized parameter to FP32.
    On a 14B NF4 model that temporary/permanent copy exceeds Kaggle T4 memory.
    This helper provides the QLoRA essentials only: freeze the base, enable
    input gradients, and enable checkpointing.  LoRA adapters are added later
    by TRL/PEFT and remain trainable.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False
    if hasattr(model, "config"):
        model.config.use_cache = False
    if not gradient_checkpointing:
        return model

    enable_input_require_grads = getattr(model, "enable_input_require_grads", None)
    if callable(enable_input_require_grads):
        enable_input_require_grads()
    else:
        embeddings = model.get_input_embeddings()

        def require_output_gradients(_module: Any, _inputs: Any, output: Any) -> None:
            output.requires_grad_(True)

        embeddings.register_forward_hook(require_output_gradients)

    enable_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
    if callable(enable_checkpointing):
        try:
            enable_checkpointing(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            enable_checkpointing()
    return model


__all__ = ["cast_trainable_parameters_to_fp32", "prepare_4bit_qlora_model"]
