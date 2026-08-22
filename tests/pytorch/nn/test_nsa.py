from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends import nsa


def _build_indexer_meta(q_seqlens: torch.Tensor,
                        kv_seqlens: torch.Tensor,
                        is_decoding: bool):
    num_tokens = int(q_seqlens.sum())
    cu_seqlens_q = torch.nn.functional.pad(q_seqlens.cumsum(0), (1, 0))
    sequence_metadata = SimpleNamespace(
        block_offsets=torch.zeros(q_seqlens.size(0), 1, dtype=torch.int32),
        q_start_loc=cu_seqlens_q[:-1],
        q_seqlens=q_seqlens,
        kv_start_loc=None,
        kv_seqlens=kv_seqlens,
        kv_flatten_size=int(kv_seqlens.sum()),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=torch.nn.functional.pad(kv_seqlens.cumsum(0), (1, 0)),
        max_kv_seqlen=int(kv_seqlens.max()),
    )
    return nsa.build_nsa_index_meta(
        num_tokens=num_tokens,
        is_decoding=is_decoding,
        block_size=64,
        num_gpu_blocks=16,
        sequence_metadata=sequence_metadata,
    )


@pytest.mark.parametrize('query_width', [1, 5])
def test_indexer_meta_preserves_decoding_query_width(query_width):
    batch_size = 2
    q_seqlens = torch.full((batch_size, ), query_width, dtype=torch.int32)
    meta = _build_indexer_meta(q_seqlens, q_seqlens, is_decoding=True)

    assert meta.max_q_seqlen == query_width
    assert meta.kv_flatten_size is None
    assert meta.max_kv_seqlen == 64 * 16


def test_indexer_meta_builds_causal_rows():
    meta = _build_indexer_meta(
        q_seqlens=torch.tensor([2, 3], dtype=torch.int32),
        kv_seqlens=torch.tensor([5, 8], dtype=torch.int32),
        is_decoding=False,
    )

    expected = torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32)
    assert torch.equal(meta.indexer_kv_seqlens, expected)
    assert meta.kv_flatten_size == 13
    assert meta.max_kv_seqlen == 8


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='requires CUDA device with cc>=9.0')
def test_deepgemm_prefill_scores_match_triton():
    pytest.importorskip('deep_gemm')
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa
    from lmdeploy.pytorch.consts import dsa_packed_indexer_k_cache_shape

    torch.manual_seed(0)
    q_seqlens = torch.tensor([2, 3], device='cuda', dtype=torch.int32)
    kv_seqlens = torch.tensor([5, 8], device='cuda', dtype=torch.int32)
    meta = _build_indexer_meta(q_seqlens, kv_seqlens, is_decoding=False)
    meta.block_offset = torch.tensor([[1], [0]], device='cuda', dtype=torch.int32)
    meta.score_meta = cuda_nsa._build_deep_gemm_score_meta(meta)

    q = torch.randn(5, 32, 128, device='cuda').to(torch.float8_e4m3fn)
    q_s = torch.randn(5, 32, device='cuda', dtype=torch.float32)
    packed_cache = torch.empty(
        2,
        *dsa_packed_indexer_k_cache_shape(64, 128),
        device='cuda',
        dtype=torch.uint8,
    )
    k_cache, k_s_cache = cuda_nsa._get_dsa_indexer_k_cache_views(
        packed_cache, 128)
    k_cache.copy_(torch.randn_like(k_cache.float()).to(k_cache.dtype))
    k_s_cache.copy_(torch.rand_like(k_s_cache) * 0.01)
    impl = cuda_nsa.TritonNSAIndexFP8Impl(
        topk=2, softmax_scale=1.0, block_size=128, fill=-1)

    deepgemm_scores = impl._compute_scores(q, q_s, packed_cache, meta)
    meta.score_meta = None
    triton_scores = impl._compute_scores(q, q_s, packed_cache, meta)
    assert deepgemm_scores.shape == triton_scores.shape == (5, 8)
    for row, row_len in enumerate(meta.indexer_kv_seqlens.tolist()):
        torch.testing.assert_close(
            deepgemm_scores[row, :row_len],
            triton_scores[row, :row_len],
            rtol=1e-3,
            atol=1e-3,
        )


def test_indexer_meta_skips_reused_mtp_topk():
    assert nsa.should_skip_nsa_indexer([dict(skip_topk=True)])
    assert not nsa.should_skip_nsa_indexer([dict(skip_topk=False)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA backend')
def test_sparse_index_topk_is_resolved_at_init(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa

    selector = object()
    monkeypatch.setattr(cuda_nsa, '_get_sparse_index_topk',
                        lambda topk: selector)

    index_impl = cuda_nsa.TritonNSAIndexFP8Impl(
        topk=512, softmax_scale=1.0, block_size=128, fill=-1)

    assert index_impl._sparse_index_topk is selector
