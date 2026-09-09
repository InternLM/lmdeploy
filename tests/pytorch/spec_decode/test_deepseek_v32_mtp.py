# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.models.deepseek_v32_mtp import DeepseekV32MTPModel


@pytest.mark.parametrize('scale_first', [False, True])
def test_mtp_fused_indexer_loads_fp8_weight_and_scale_in_either_order(scale_first):
    # No decoder allocation is needed to exercise the real model's loader.
    config = SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=0)
    model = DeepseekV32MTPModel(config, ctx_mgr=None, dtype=torch.bfloat16, device='cpu')
    prefix = 'model.layers.61.mtp_block.self_attn.indexer'
    fused = torch.nn.Parameter(torch.empty(6, 8, dtype=torch.bfloat16), requires_grad=False)
    params = {f'{prefix}.wk_weights_proj.weight': fused}
    wk = torch.arange(32, dtype=torch.float32).reshape(4, 8).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 2.0], [1.0, 4.0]])
    gate = torch.arange(16, dtype=torch.bfloat16).reshape(2, 8)
    parts = [('wk.weight', wk), ('wk.weight_scale_inv', scale)]
    if scale_first:
        parts.reverse()
    # Separate calls model checkpoint shards arriving in either order.
    for suffix, tensor in [parts[0], ('weights_proj.weight', gate), parts[1]]:
        model._load_weight_attention(f'{prefix}.{suffix}', tensor, params, [])
    expected_wk = (wk.float() * scale.repeat_interleave(2, 0).repeat_interleave(4, 1)).bfloat16()
    torch.testing.assert_close(fused, torch.cat([expected_wk, gate]), atol=0, rtol=0)
    assert not model._load_buffers
