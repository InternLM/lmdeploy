# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise the additional direct gated-norm callers and architecture
aliases."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3NextConfig

from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.glm5_next import Glm5NextLinearAttention
from lmdeploy.pytorch.models.module_map import MODULE_MAP
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.models.qwen3_next import Qwen3NextGatedDeltaNet
from lmdeploy.pytorch.nn import RMSNormGated
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('family', ['qwen3_next', 'glm5_next'])
def test_direct_model_norm_construction_and_loading(dtype, family):
    context = BuildModelContext()
    with build_model_context(context):
        if family == 'qwen3_next':
            config = Qwen3NextConfig(hidden_size=128, num_attention_heads=2, num_key_value_heads=2,
                                    head_dim=64, linear_key_head_dim=32, linear_value_head_dim=32,
                                    linear_num_key_heads=2, linear_num_value_heads=2)
            layer = Qwen3NextGatedDeltaNet(config, layer_idx=0, dtype=dtype, device='cpu')
            name, activation = 'norm', 'silu'
        else:
            config = SimpleNamespace(hidden_size=128, rms_norm_eps=1e-6,
                                     linear_attn_config=dict(num_heads=4, head_dim=32, short_conv_kernel_size=4))
            layer = Glm5NextLinearAttention(config, layer_idx=0, dtype=dtype, device='cpu')
            name, activation = 'o_norm', 'sigmoid'
    norm = getattr(layer, name)
    assert isinstance(norm, RMSNormGated)
    assert norm.activation == activation
    assert norm.weight.dtype == dtype
    assert tuple(norm.weight.shape) == (32,)
    params = dict(layer.named_parameters())
    assert params[f'{name}.weight'] is norm.weight
    assert not any(key.startswith(f'{name}.impl.') for key in params)
    scale = torch.linspace(-1, 1, 32, dtype=dtype)
    load_weight(params[f'{name}.weight'], scale)
    torch.testing.assert_close(norm.weight, scale, rtol=0, atol=0)


@pytest.mark.parametrize('architecture', ['InternS2PreviewForConditionalGeneration', 'InternS2PreviewForCausalLM'])
def test_preview_alias_uses_covered_qwen35_moe(architecture):
    assert MODULE_MAP[architecture] == MODULE_MAP['Qwen3_5MoeForConditionalGeneration']
