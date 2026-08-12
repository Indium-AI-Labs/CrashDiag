from __future__ import annotations

import unittest

from training.qlora import prepare_4bit_qlora_model


class _Parameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _Config:
    use_cache = True


class _Model:
    def __init__(self) -> None:
        self.config = _Config()
        self.parameters_ = [_Parameter(), _Parameter()]
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


if __name__ == "__main__":
    unittest.main()
