from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.nn import nsa


@pytest.mark.parametrize('query_width', [1, 5])
def test_indexer_meta_preserves_decoding_query_width(monkeypatch, query_width):
    batch_size = 2
    step_ctx = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=16))
    ctx_mgr = SimpleNamespace(current_context=lambda: step_ctx)
    monkeypatch.setattr(nsa, 'get_step_ctx_manager', lambda: ctx_mgr)
    attn_metadata = SimpleNamespace(
        is_decoding=True,
        max_q_seqlen=None,
        kv_flatten_size=None,
        cu_seqlens_q=torch.arange(batch_size + 1) * query_width,
        q_seqlens=torch.full((batch_size, ), query_width),
        kv_seqlens=torch.full((batch_size, ), query_width),
        block_offsets=torch.zeros(batch_size, 1, dtype=torch.int32))
    q = torch.empty(batch_size * query_width, 1, 128)

    meta = nsa.IndexerTopKFP8._build_meta(q, attn_metadata)

    assert meta.max_q_seqlen == query_width
