# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.config import DistConfig


@pytest.mark.parametrize('dp,tp,ep', [(1, 1, 1), (1, 2, 2), (2, 1, 2),
                                     (2, 4, 4), (2, 4, 1), (4, 1, 4)])
def test_v4_query_projection_matches_attention_head_sharding(monkeypatch, dp, tp, ep):
    from lmdeploy.pytorch.models import deepseek_v4 as model
    from lmdeploy.pytorch.nn import linear

    cfg = DistConfig(dp=dp, tp=tp, ep=ep)
    monkeypatch.setattr(model, 'get_tp_world_rank', lambda *_: (cfg.attn_tp, 0))
    monkeypatch.setattr(linear, 'get_dist_manager', lambda: SimpleNamespace(current_config=lambda: cfg))

    def build_linear(in_features, out_features, is_tp, **kwargs):
        # Exercise the real columnwise builder's distributed policy, replacing
        # only the device-specific implementation underneath it.
        local_out = out_features // cfg.attn_tp if is_tp else out_features
        return nn.Linear(in_features, local_out, bias=False)

    monkeypatch.setattr(linear, 'build_linear', build_linear)
    for name in ('RMSNorm', 'DeepseekV4BMM', 'build_o_proj', 'NativeV4Attention', 'ApplyRotaryEmb'):
        monkeypatch.setattr(model, name, lambda *args, **kwargs: nn.Identity())
    args = SimpleNamespace(n_heads=8, head_dim=4, rope_head_dim=2, o_groups=4, n_groups=4,
                           window_size=128, ring_storage_capacity=132, compress_ratios=[0],
                           norm_eps=1e-6, o_lora_rank=4, dim=16, q_lora_rank=8)
    attn = model.Attention(SimpleNamespace(attention_bias=False), 0, args, torch.float32, 'cpu')
    projected = attn.wq_b(torch.zeros(1, 3, args.q_lora_rank))
    assert projected.shape[-1] == attn.n_local_heads * args.head_dim
    assert projected.unflatten(-1, (attn.n_local_heads, args.head_dim)).shape == (
        1, 3, args.n_heads // cfg.attn_tp, args.head_dim)
