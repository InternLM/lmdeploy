# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.models import glm5_next
from lmdeploy.pytorch.nn.kpool import KPOOL_INDEXER_PARAMETER_NAMES, KPoolIndexer
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA')),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('owner', ['kpool', 'vision_merger'])
def test_layer_norm_call_sites_preserve_fp32_contract(monkeypatch, dtype, device, owner):
    if owner == 'kpool':
        module = KPoolIndexer(128, 2, 128, 8, 4, 4, dtype=dtype, device=device)
        assert set(dict(module.named_parameters())) == set(KPOOL_INDEXER_PARAMETER_NAMES)
        module.wk = nn.Identity()
        norm = module.k_norm
        forward = module.project_key
    else:
        module = glm5_next.Glm5NextVisionPatchMerger(
            SimpleNamespace(out_hidden_size=128, intermediate_size=128, swiglu_limit=10.0),
            dtype=dtype, device=device)
        # Isolate the real merger's norm -> output cast -> GELU call sequence.
        module.proj = nn.Identity()
        module.gate_up_proj = nn.Identity()
        module.down_proj = nn.Identity()
        monkeypatch.setattr(glm5_next, '_glm_swiglu_impl', lambda x, *args, **kwargs: x)
        norm = module.post_projection_norm
        forward = module

    assert type(norm) is nn.LayerNorm
    assert norm.eps == 1e-6
    assert norm.weight.dtype == norm.bias.dtype == torch.float32
    assert not norm.weight.requires_grad and not norm.bias.requires_grad
    torch.testing.assert_close(norm.weight, torch.ones_like(norm.weight))
    torch.testing.assert_close(norm.bias, torch.zeros_like(norm.bias))
    generator = torch.Generator().manual_seed(123)
    # Non-BF16-representable weights catch accidental parameter downcasting.
    load_weight(norm.weight, torch.randn(128, generator=generator))
    load_weight(norm.bias, torch.randn(128, generator=generator))
    for tokens in (1, 129, 8192):
        inputs = torch.randn(tokens, 128, generator=generator).to(device=device, dtype=dtype)
        expected = F.layer_norm(inputs.float(), (128,), norm.weight, norm.bias, norm.eps).to(dtype)
        if owner == 'vision_merger':
            expected = F.gelu(expected)
        actual = forward(inputs)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_glm_indexer_uses_default_layer_norm():
    config = SimpleNamespace(hidden_size=16, index_n_heads=2, index_head_dim=8,
                             index_topk=8, q_lora_rank=4, index_kpool=4)
    indexer = glm5_next.Glm5NextSparseAttention._build_indexer(
        None, config, layer_idx=0, dtype=torch.bfloat16, device=torch.device('cpu'))
    assert type(indexer.k_norm) is nn.LayerNorm
    assert indexer.k_norm.weight.dtype == indexer.k_norm.bias.dtype == torch.float32
