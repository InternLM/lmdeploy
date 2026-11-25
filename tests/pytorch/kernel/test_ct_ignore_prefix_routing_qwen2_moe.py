# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""Ignore-rule routing through the Qwen2-MoE block builders.

A 4-bit group-128 GPTQ checkpoint that carries ``modules_to_not_convert``
rules needs the canonical module prefix at construction time, both for the
fused routed experts and for the shared-expert projections. Without the
prefix ``get_quant_method`` raises before any expert is built — even when
the excluded module lies outside the block being built.

All cases run on tiny shapes; nothing downloads.
"""

import pytest
import torch
from transformers.models.qwen2_moe import Qwen2MoeConfig

from lmdeploy.pytorch.config import QuantizationConfig
from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from lmdeploy.pytorch.nn.linear import BaseLinear
from lmdeploy.pytorch.nn.moe.default import FusedMoE

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def _gptq_quant_config(ignore: list[str]) -> QuantizationConfig:
    """4-bit group-128 GPTQ config with modules_to_not_convert rules."""
    hf_config = type('HF', (), {
        'quantization_config': {
            'quant_method': 'gptq',
            'bits': 4,
            'group_size': 128,
            'modules_to_not_convert': ignore,
        }
    })()
    return QuantizationConfig.from_config(hf_config)


def _tiny_qwen2_moe_config(ignore: list[str]) -> Qwen2MoeConfig:
    config = Qwen2MoeConfig(
        hidden_size=128,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        vocab_size=96,
        decoder_sparse_step=1,
    )
    config.quantization_config = _gptq_quant_config(ignore).hf_quant_config
    return config


def _build_block(ignore: list[str]) -> Qwen2MoeSparseMoeBlock:
    config = _tiny_qwen2_moe_config(ignore)
    device = torch.device('cuda')
    with build_model_context(BuildModelContext(quant_config=_gptq_quant_config(ignore))):
        return Qwen2MoeSparseMoeBlock(config,
                                      layer_idx=0,
                                      dtype=torch.float16,
                                      device=device,
                                      prefix='model.layers.0.mlp')


@cuda_required
def test_gptq_ignore_outside_block_allows_construction():
    """ignore=['lm_head'] must not block MoE block construction."""
    block = _build_block(ignore=['lm_head'])
    assert not isinstance(block.experts, FusedMoE)
    assert not isinstance(block.shared_expert.gate_up_proj, BaseLinear)
    assert not isinstance(block.shared_expert.down_proj, BaseLinear)


@cuda_required
@pytest.mark.parametrize('rule', [
    'model.layers.0.mlp.experts.gate_up_proj',
    'model.layers.0.mlp.shared_expert.gate_up_proj',
])
def test_excluded_projection_rule_is_honored(rule):
    """An ignore rule matching one module excludes exactly that module."""
    block = _build_block(ignore=[rule])
    experts_quantized = rule != 'model.layers.0.mlp.experts.gate_up_proj'
    shared_quantized = rule != 'model.layers.0.mlp.shared_expert.gate_up_proj'
    assert isinstance(block.experts, FusedMoE) is not experts_quantized
    assert isinstance(block.shared_expert.gate_up_proj, BaseLinear) is not shared_quantized
    assert not isinstance(block.shared_expert.down_proj, BaseLinear)
