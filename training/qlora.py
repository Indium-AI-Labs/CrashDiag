"""Memory-safe preparation for 4-bit QLoRA policies."""

from __future__ import annotations

from typing import Any


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


__all__ = ["prepare_4bit_qlora_model"]
