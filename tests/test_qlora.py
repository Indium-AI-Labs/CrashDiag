from __future__ import annotations

import unittest

from training.qlora import cast_trainable_parameters_to_fp32, prepare_4bit_qlora_model


class _Parameter:
    def __init__(self, data, *, requires_grad: bool = True) -> None:
        self.data = data
        self.requires_grad = requires_grad


class _Data:
    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def float(self):
        return _Data("float32")


class _Config:
    use_cache = True


class _Model:
    def __init__(self) -> None:
        self.config = _Config()
        self.parameters_ = [_Parameter(_Data("float16")), _Parameter(_Data("float16"))]
        self.input_grads_enabled = False
        self.checkpointing_kwargs = None

    def parameters(self):
        return iter(self.parameters_)

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.checkpointing_kwargs = kwargs


class QLoRAPreparationTests(unittest.TestCase):
    def test_freezes_base_and_enables_checkpointing_without_dtype_mutation(self) -> None:
        model = _Model()

        prepared = prepare_4bit_qlora_model(model)

        self.assertIs(prepared, model)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters_))
        self.assertFalse(model.config.use_cache)
        self.assertTrue(model.input_grads_enabled)
        self.assertEqual(
            model.checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )

    def test_only_trainable_low_precision_parameters_are_cast_to_fp32(self) -> None:
        trainable_bf16 = _Parameter(_Data("bfloat16"))
        trainable_fp16 = _Parameter(_Data("float16"))
        frozen_bf16 = _Parameter(_Data("bfloat16"), requires_grad=False)
        trainable_fp32 = _Parameter(_Data("float32"))
        model = _Model()
        model.parameters_ = [trainable_bf16, trainable_fp16, frozen_bf16, trainable_fp32]

        count = cast_trainable_parameters_to_fp32(model)

        self.assertEqual(count, 2)
        self.assertEqual(trainable_bf16.data.dtype, "float32")
        self.assertEqual(trainable_fp16.data.dtype, "float32")
        self.assertEqual(frozen_bf16.data.dtype, "bfloat16")
        self.assertEqual(trainable_fp32.data.dtype, "float32")


if __name__ == "__main__":
    unittest.main()
