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
        indexer_kv_seqlens=None,
        block_offsets=torch.zeros(batch_size, 1, dtype=torch.int32))
    q = torch.empty(batch_size * query_width, 1, 128)

    nsa.update_nsa_indexer_kv_seqlens(q.size(0), attn_metadata)
    meta = nsa.IndexerTopKFP8._build_meta(q, attn_metadata)

    assert meta.max_q_seqlen == query_width
    assert meta.indexer_kv_seqlens is attn_metadata.indexer_kv_seqlens


def test_update_nsa_indexer_kv_seqlens_builds_causal_rows():
    attn_metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        q_seqlens=torch.tensor([2, 3]),
        kv_seqlens=torch.tensor([5, 8]),
        indexer_kv_seqlens=None,
    )

    nsa.update_nsa_indexer_kv_seqlens(5, attn_metadata)

    expected = torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32)
    assert torch.equal(attn_metadata.indexer_kv_seqlens, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA backend')
def test_sparse_index_topk_is_resolved_at_init(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa

    selector = object()
    monkeypatch.setattr(cuda_nsa, '_get_sparse_index_topk', lambda topk: selector)

    index_impl = cuda_nsa.TritonNSAIndexFP8(topk=512, softmax_scale=1.0, block_size=128, fill=-1)

    assert index_impl._sparse_index_topk is selector

    packed_cache = torch.zeros(3, 64, 1, 132, dtype=torch.uint8)
    values, scales = cuda_nsa._get_dsa_indexer_k_cache_views(packed_cache, head_dim=128)

    assert values.shape == (3, 64, 128)
    assert scales.shape == (3, 64, 1)
    assert values.untyped_storage().data_ptr() == packed_cache.untyped_storage().data_ptr()
    assert scales.untyped_storage().data_ptr() == packed_cache.untyped_storage().data_ptr()
