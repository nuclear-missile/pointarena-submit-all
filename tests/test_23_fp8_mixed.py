import torch

from src.train.fp8_mixed import (
    FrozenFP8Linear,
    apply_fp8_base_storage,
    apply_fp8_mixed_precision,
    cast_trainable_parameters,
    fp8_dynamic_quantize_ste,
)


def test_fp8_dynamic_quantize_ste_preserves_shape_dtype_and_gradient():
    x = torch.linspace(-3.0, 3.0, steps=12, dtype=torch.float32).reshape(3, 4).requires_grad_()

    y = fp8_dynamic_quantize_ste(x, block_size=4)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all()

    y.sum().backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_apply_fp8_mixed_precision_wraps_selected_base_linears_only():
    class FakeLoraLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4, bias=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})

        def forward(self, x):
            return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target = FakeLoraLayer()
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.lm_head(self.target(x))

    model = ToyModel()
    report = apply_fp8_mixed_precision(
        model,
        selected_target_names=["target"],
        enabled=True,
        mode="emulated",
        quantize_lora_adapters=False,
    )

    assert report["enabled"] is True
    assert report["wrapped_linears"] == 1
    assert report["native_fp8_matmul_supported"] is False
    assert hasattr(model.target.base_layer, "_fp8_original_forward")
    assert not hasattr(model.target.lora_A["default"], "_fp8_original_forward")
    assert not hasattr(model.lm_head, "_fp8_original_forward")

    out = model(torch.randn(2, 4))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_frozen_fp8_linear_stores_weight_as_fp8_and_computes_in_input_dtype():
    linear = torch.nn.Linear(5, 3, bias=True).to(dtype=torch.float16)
    frozen = FrozenFP8Linear.from_linear(linear, weight_dtype="e4m3fn", block_size=4)

    assert not any(True for _ in frozen.parameters())
    assert frozen.weight_fp8.dtype is torch.float8_e4m3fn
    assert frozen.weight_scale.dtype is torch.float32

    x = torch.randn(2, 5, dtype=torch.float16)
    out = frozen(x)

    assert out.shape == (2, 3)
    assert out.dtype is torch.float16
    assert torch.isfinite(out).all()


def test_apply_fp8_base_storage_replaces_only_limited_selected_base_linears():
    class FakeLoraLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4, bias=False)
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})

        def forward(self, x):
            return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.target_a = FakeLoraLayer()
            self.target_b = FakeLoraLayer()

        def forward(self, x):
            return self.target_b(self.target_a(x))

    model = ToyModel().to(dtype=torch.float16)
    report = apply_fp8_base_storage(
        model,
        selected_target_names=["target_a", "target_b"],
        enabled=True,
        weight_dtype="e4m3fn",
        block_size=4,
        max_modules=1,
    )

    assert report["enabled"] is True
    assert report["replaced_linears"] == 1
    assert isinstance(model.target_a.base_layer, FrozenFP8Linear)
    assert isinstance(model.target_b.base_layer, torch.nn.Linear)
    assert not isinstance(model.target_a.lora_A["default"], FrozenFP8Linear)

    out = model(torch.randn(2, 4, dtype=torch.float16))
    assert out.shape == (2, 4)
    assert out.dtype is torch.float16
    assert torch.isfinite(out).all()


def test_cast_trainable_parameters_converts_only_requires_grad_params():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.Linear(4, 2, bias=False),
    ).to(dtype=torch.bfloat16)
    model[0].weight.requires_grad_(False)

    report = cast_trainable_parameters(model, dtype=torch.float16)

    assert report["converted_params"] == 1
    assert model[0].weight.dtype is torch.bfloat16
    assert model[1].weight.dtype is torch.float16
