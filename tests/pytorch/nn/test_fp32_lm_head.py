# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import BuildModelContext
from lmdeploy.pytorch.models.patch import build_model_context
from lmdeploy.pytorch.nn import ParallelEmbedding, ParallelLMHead


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('tied', [False, True])
def test_direct_head_storage_and_tied_embedding(enabled, tied):
    with build_model_context(BuildModelContext(fp32_lm_head=enabled, tie_word_embeddings=tied)):
        emb = ParallelEmbedding(128, 64, None, dtype=torch.bfloat16, device='cpu')
        head = ParallelLMHead(128, 64, dtype=torch.bfloat16, device='cpu', is_tp=False)
        if tied:
            head.tie_weights(emb)
    assert head.weight.dtype == (torch.float32 if enabled else torch.bfloat16)
    assert emb.weight.dtype == (torch.float32 if enabled and tied else torch.bfloat16)
    assert emb.out_dtype == torch.bfloat16
    assert (head.weight is emb.weight) == tied


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('tied', [False, True])
def test_fp32_head_computes_fp32_not_cast_bf16_logits(enabled, tied):
    with build_model_context(BuildModelContext(fp32_lm_head=enabled, tie_word_embeddings=tied)):
        emb = ParallelEmbedding(128, 64, None, dtype=torch.bfloat16, device='cuda')
        head = ParallelLMHead(128, 64, dtype=torch.bfloat16, device='cuda', is_tp=False)
        if tied:
            head.tie_weights(emb)
    torch.manual_seed(123)
    w = torch.randn(128, 64, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(3, 64, device='cuda', dtype=torch.bfloat16)
    head.weight_loader(head.weight, w)
    logits = head(x)
    expected = torch.nn.functional.linear(x.to(head.weight.dtype), w.to(head.weight.dtype))
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert logits.dtype == head.weight.dtype
    if enabled:
        assert not torch.equal(logits, torch.nn.functional.linear(x, w).float())
    if tied:
        assert emb(torch.tensor([0, 1], device='cuda')).dtype == torch.bfloat16
    head.weight_loader(head.weight, w * 2)
    torch.testing.assert_close(head(x), expected * 2, rtol=0, atol=0)
    for _ in range(3):
        head(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = head(x)
    graph.replay()
    torch.testing.assert_close(captured, head(x), rtol=0, atol=0)


@pytest.mark.parametrize('enabled', [False, True])
def test_glm_target_and_mtp_direct_builders(monkeypatch, enabled):
    from lmdeploy.pytorch.models.deepseek_mtp import SharedHead
    from lmdeploy.pytorch.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    class Backbone(nn.Module):
        def __init__(self, config, dtype, device):
            super().__init__()

    monkeypatch.setattr(GlmMoeDsaForCausalLM, 'model_cls', Backbone)
    cfg = SimpleNamespace(vocab_size=128, hidden_size=64, tie_word_embeddings=False, rms_norm_eps=1e-6)
    with build_model_context(BuildModelContext(fp32_lm_head=enabled)):
        target = GlmMoeDsaForCausalLM(cfg, None, dtype=torch.bfloat16, device='cpu')
        mtp = SharedHead(cfg, dtype=torch.bfloat16, device='cpu')
    expected = torch.float32 if enabled else torch.bfloat16
    assert target.lm_head.weight.dtype == expected
    assert mtp.head.weight.dtype == expected
    assert target.dtype == torch.bfloat16
