# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Ignore-rule routing through the Llama dense builders.

A compressed-tensors group-128 ``pack-quantized`` checkpoint that carries
``ignore`` rules needs the canonical module prefix at construction time:
``get_quant_method`` evaluates the rules per real projection. Without the
prefix the builders raise before any weight is loaded — even when the
excluded module lies outside the module being built.

All cases run on tiny shapes; nothing downloads.
"""

import pytest
import torch
from transformers.models.llama import LlamaConfig

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.llama import LlamaAttention, LlamaMLP
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.nn.linear import BaseLinear

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def _ct_quant_config(ignore: list[str]) -> QuantizationConfig:
    """Group-128 pack-quantized compressed-tensors config with ignore rules."""
    hf_config = type('HF', (), {
        'quantization_config': {
            'quant_method': 'compressed-tensors',
            'format': 'pack-quantized',
            'quantization_status': 'compressed',
            'config_groups': {
                'group_0': {
                    'targets': ['Linear'],
                    'input_activations': None,
                    'output_activations': None,
                    'weights': {
                        'num_bits': 4,
                        'group_size': 128,
                        'strategy': 'group',
                        'symmetric': True,
                        'dynamic': False,
                        'type': 'int',
                    },
                },
            },
            'ignore': ignore,
        }
    })()
    return QuantizationConfig.from_config(hf_config)


def _tiny_llama_config(ignore: list[str]) -> LlamaConfig:
    config = LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        vocab_size=96,
        attention_bias=False,
    )
    config.quantization_config = _ct_quant_config(ignore).hf_quant_config
    return config


def _is_quantized(module: torch.nn.Module) -> bool:
    return not isinstance(module, BaseLinear)


@cuda_required
def test_ignore_outside_attention_allows_attention_and_mlp_construction():
    """ignore=['lm_head'] must not block attention/MLP construction."""
    config = _tiny_llama_config(ignore=['lm_head'])
    device = torch.device('cuda')
    with build_model_context(BuildModelContext(quant_config=_ct_quant_config(['lm_head']))):
        attn = LlamaAttention(config, device=device, is_tp=False, prefix='model.layers.0.self_attn')
        mlp = LlamaMLP(config, device=device, is_tp=False, prefix='model.layers.0.mlp')
    assert _is_quantized(attn.qkv_proj)
    assert _is_quantized(attn.o_proj)
    assert _is_quantized(mlp.gate_up_proj)
    assert _is_quantized(mlp.down_proj)


@cuda_required
def test_full_llama_model_with_ignore_list_constructs():
    """The whole Llama stack builds with a nonempty ignore list."""
    from lmdeploy.pytorch.model_inputs import StepContextManager
    from lmdeploy.pytorch.models.llama import LlamaForCausalLM

    ignore = ['lm_head']
    config = _tiny_llama_config(ignore=ignore)
    with build_model_context(BuildModelContext(quant_config=_ct_quant_config(ignore))):
        model = LlamaForCausalLM(config, StepContextManager(), dtype=torch.float16, device=torch.device('cuda'))
    assert _is_quantized(model.model.layers[0].self_attn.qkv_proj)
    assert isinstance(model.lm_head, BaseLinear)


@cuda_required
@pytest.mark.parametrize('rule,prefix,excluded', [
    ('model.layers.0.self_attn.o_proj', 'model.layers.0.self_attn', 'o_proj'),
    ('model.layers.0.mlp.down_proj', 'model.layers.0.mlp', 'down_proj'),
])
def test_excluded_projection_builds_unquantized(rule, prefix, excluded):
    """An ignore rule matching one projection excludes exactly that module."""
    config = _tiny_llama_config(ignore=[rule])
    device = torch.device('cuda')
    quant_config = _ct_quant_config([rule])
    with build_model_context(BuildModelContext(quant_config=quant_config)):
        attn = LlamaAttention(config, device=device, is_tp=False, prefix='model.layers.0.self_attn')
        mlp = LlamaMLP(config, device=device, is_tp=False, prefix='model.layers.0.mlp')
    assert isinstance(getattr(attn if excluded.startswith('o') else mlp, excluded), BaseLinear)
    siblings = {
        'qkv_proj': attn.qkv_proj,
        'o_proj': attn.o_proj,
        'gate_up_proj': mlp.gate_up_proj,
        'down_proj': mlp.down_proj,
    }
    for name, module in siblings.items():
        if name == excluded:
            continue
        assert _is_quantized(module), f'{name} must stay quantized'
