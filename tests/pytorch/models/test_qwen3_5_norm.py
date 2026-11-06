# Copyright (c) OpenMMLab. All rights reserved.
"""Check zero-centered checkpoint loading across the Qwen3.5 model family."""
import importlib

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig

from lmdeploy.pytorch.model_inputs import BuildModelContext, StepContextManager
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.nn import RMSNorm


@pytest.mark.parametrize('model_path', [
    'qwen3_5.Qwen3_5ForConditionalGeneration',
    'qwen3_5_moe.Qwen3_5MoeForConditionalGeneration',
    'meta_moe.MetaMoeForConditionalGeneration',
    'interns2_mobius.InternS2MobiusForConditionalGeneration',
    'qwen3_5_mtp.Qwen3_5MTPModel',
])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_zero_centered_norm_loading(model_path, dtype):
    module_name, class_name = model_path.split('.')
    model_class = getattr(importlib.import_module(f'lmdeploy.pytorch.models.{module_name}'), class_name)
    text_config = Qwen3_5TextConfig(
        hidden_size=128, intermediate_size=256, vocab_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
        head_dim=64, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2,
        layer_types=['linear_attention', 'full_attention'],
        num_experts=2, num_experts_per_tok=1, moe_intermediate_size=64,
        shared_expert_intermediate_size=64, num_meta_moe_blocks=1, num_blocks=1,
        mtp_num_hidden_layers=1,
    )
    config = Qwen3_5Config(text_config=text_config.to_dict())
    context = BuildModelContext(language_model_only=True)
    with build_model_context(context):
        model = model_class(config, StepContextManager(context), dtype=dtype, device='cpu')

    norms = {name: module for name, module in model.named_modules() if isinstance(module, RMSNorm)}
    assert norms
    weights = []
    expected = {}
    for name, norm in norms.items():
        assert norm.weight.dtype == torch.float32, name
        # This offset loses low bits when +1 is evaluated or stored in BF16/FP16.
        offset = torch.full(norm.weight.shape, 0.24609375, dtype=dtype)
        offset[0] = -1.0
        offset[-1] = 0.0
        expected[name] = offset.float() + 1
        checkpoint_name = f'{name}.weight'
        if module_name == 'qwen3_5_mtp':
            checkpoint_name = checkpoint_name.replace('model.', 'mtp.', 1)
        weights.append((checkpoint_name, offset))

    # Gated norms use direct scales; they must not inherit the +1 convention.
    gated = {name: module for name, module in model.named_modules()
             if name.endswith('linear_attn.norm')}
    for name, norm in gated.items():
        assert norm.weight.dtype == dtype
        weights.append((f'{name}.weight', torch.full_like(norm.weight, 0.25)))
    model.load_weights(weights)
    for name, norm in norms.items():
        torch.testing.assert_close(norm.weight, expected[name], rtol=0, atol=0, msg=name)
    for norm in gated.values():
        torch.testing.assert_close(norm.weight, torch.full_like(norm.weight, 0.25), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA RMSNorm arithmetic')
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('with_residual', [False, True])
@torch.inference_mode()
def test_fp32_scale_matches_reference(dtype, with_residual):
    from lmdeploy.pytorch.kernels.cuda import rms_norm

    torch.manual_seed(113)
    x = torch.randn(8, 128, dtype=dtype, device='cuda')
    residual = torch.randn_like(x) if with_residual else None
    offset = torch.full((128,), 0.24609375, dtype=dtype, device='cuda')
    scale = offset.float() + 1
    combined = x + residual if residual is not None else x
    normalized = combined.float() * torch.rsqrt(combined.float().square().mean(-1, keepdim=True) + 1e-6)
    expected = (normalized * scale).to(dtype)
    actual = rms_norm(x, scale, residual=residual)
    if residual is not None:
        actual, actual_residual = actual
        torch.testing.assert_close(actual_residual, combined, rtol=0, atol=0)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rounded = rms_norm(x, (offset + 1).to(dtype), residual=residual)
    if residual is not None:
        rounded = rounded[0]
    if dtype == torch.bfloat16:
        assert not torch.equal(rounded, expected)
