# Copyright (c) OpenMMLab. All rights reserved.
"""Check task-specific and ordinary Qwen3.5 norm loading."""
import importlib

import pytest
import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig

from lmdeploy.pytorch.config import DistConfig, ModelConfig, QuantizationConfig
from lmdeploy.pytorch.model_inputs import BuildModelContext, StepContextManager
from lmdeploy.pytorch.models.intern_decision import InternDecisionForConditionalGeneration
from lmdeploy.pytorch.models.patch import build_model_context, build_model_from_hf_config, build_patched_model
from lmdeploy.pytorch.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from lmdeploy.pytorch.nn import RMSNorm


def _tiny_config():
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
    return Qwen3_5Config(architectures=['Qwen3_5ForConditionalGeneration'], text_config=text_config.to_dict())


@pytest.mark.parametrize('model_path', [
    'intern_decision.InternDecisionForConditionalGeneration',
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
    is_decision = module_name == 'intern_decision'
    model_class = getattr(importlib.import_module(f'lmdeploy.pytorch.models.{module_name}'), class_name)
    config = _tiny_config()
    context = BuildModelContext(language_model_only=True)
    with build_model_context(context):
        model = model_class(config, StepContextManager(context), dtype=dtype, device='cpu')

    norms = {name: module for name, module in model.named_modules() if isinstance(module, RMSNorm)}
    assert norms
    weights = []
    expected = {}
    for name, norm in norms.items():
        assert norm.weight.dtype == (torch.float32 if is_decision else dtype), name
        # This offset loses low bits when +1 is evaluated or stored in BF16/FP16.
        offset = torch.full(norm.weight.shape, 0.24609375, dtype=dtype)
        offset[0] = -1.0
        offset[-1] = 0.0
        expected[name] = offset.float() + 1 if is_decision else offset + 1
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


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('checkpoint_dtype', [torch.bfloat16, torch.float16, torch.float32])
@torch.inference_mode()
def test_decision_mixed_checkpoint_and_repeated_shards(dtype, checkpoint_dtype):
    context = BuildModelContext()
    config = _tiny_config()
    config.vision_config.depth = 0
    with build_model_context(context):
        model = InternDecisionForConditionalGeneration(config, StepContextManager(context), dtype=dtype, device='cpu')
    params = dict(model.named_parameters())
    target_names = {
        'model.language_model.norm.weight',
        *(f'model.language_model.layers.{i}.{kind}.weight' for i in range(2)
          for kind in ('input_layernorm', 'post_attention_layernorm')),
        *(f'model.language_model.layers.1.self_attn.{kind}_norm.weight' for kind in ('q', 'k')),
    }
    assert model._fp32_norm_weights == target_names
    untouched = ['model.language_model.layers.0.linear_attn.norm.weight', 'model.visual.merger.norm.weight',
                 'model.language_model.embed_tokens.weight', 'lm_head.weight']
    for name in untouched:
        assert params[name].dtype == dtype

    # Reloading the same shard must not add +1 to already folded parameters.
    offset = torch.tensor(1 / 256, dtype=checkpoint_dtype)
    for _ in range(2):
        for name in sorted(target_names | set(untouched)):
            weight = torch.full(params[name].shape, offset, dtype=checkpoint_dtype)
            model.load_weights(iter([(name, weight)]))
            expected = offset.float() + 1 if name in target_names else offset.to(dtype)
            torch.testing.assert_close(params[name], torch.full_like(params[name], expected), rtol=0, atol=0)


@pytest.mark.parametrize('enabled', [False, True])
def test_task_marker_survives_model_config_and_pickle(tmp_path, enabled):
    import pickle

    config = _tiny_config()
    config.save_pretrained(tmp_path)
    overrides = {'_lmdeploy_task': 'intern-decision'} if enabled else None
    model_config = ModelConfig.from_pretrained(str(tmp_path), dtype='bfloat16', dist_config=DistConfig(),
                                                hf_overrides=overrides, device_type='cpu')
    model_config = pickle.loads(pickle.dumps(model_config))
    model = build_patched_model(model_config, device=torch.device('cpu'),
                                 build_model_ctx=BuildModelContext(language_model_only=True))
    expected_class = InternDecisionForConditionalGeneration if enabled else Qwen3_5ForConditionalGeneration
    assert type(model) is expected_class
    assert model.model.language_model.norm.weight.dtype == (torch.float32 if enabled else torch.bfloat16)
    # A following ordinary build must not inherit a process-global mapping.
    ordinary = build_model_from_hf_config(config, dtype=torch.bfloat16, device=torch.device('cpu'),
                                          build_model_ctx=BuildModelContext(language_model_only=True))
    assert type(ordinary) is Qwen3_5ForConditionalGeneration


@pytest.mark.parametrize('mapping', ['architecture', 'auto_map', 'custom_map'])
def test_task_rejects_actual_derived_model(mapping, monkeypatch):
    from lmdeploy.pytorch.models.module_map import CUSTOM_MODULE_MAP

    config = _tiny_config()
    config._lmdeploy_task = 'intern-decision'
    if mapping == 'architecture':
        config.architectures = ['Qwen3_5MoeForConditionalGeneration']
    elif mapping == 'auto_map':
        config.auto_map = {'AutoModelForCausalLM': 'custom.Qwen3_5MoeForConditionalGeneration'}
    else:
        monkeypatch.setitem(CUSTOM_MODULE_MAP, 'Qwen3_5ForConditionalGeneration',
                            'lmdeploy.pytorch.models.qwen3_5_moe.Qwen3_5MoeForConditionalGeneration')
    with pytest.raises(ValueError, match='dense Qwen3.5 model implementation'):
        build_model_from_hf_config(config, dtype=torch.bfloat16, device=torch.device('cpu'))


@pytest.mark.parametrize('source', ['config', 'build_context'])
def test_decision_rejects_quantization_before_constructing_layers(source, monkeypatch):
    config = _tiny_config()
    context = BuildModelContext()
    if source == 'config':
        config.text_config.quantization_config = {'quant_method': 'fp8'}
    else:
        context.quant_config = QuantizationConfig(quant_method='fp8')

    def unexpected_build(*args, **kwargs):
        pytest.fail('Quantized decision models must fail before building layers')

    monkeypatch.setattr(Qwen3_5ForConditionalGeneration, '__init__', unexpected_build)
    with build_model_context(context), pytest.raises(ValueError, match='unquantized dense Qwen3.5'):
        InternDecisionForConditionalGeneration(config, None, dtype=torch.bfloat16, device='cpu')


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
