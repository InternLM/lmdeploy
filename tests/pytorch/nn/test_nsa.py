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


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA backend')
def test_deepgemm_prefill_scores_are_chunked_by_logits_budget(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa

    class FakeDeepGemm:

        def __init__(self):
            self.row_counts = []

        def fp8_fp4_mqa_logits(self, q, kv, weights, cu_seq_len_k_start,
                               cu_seq_len_k_end, **kwargs):
            query = q[0]
            self.row_counts.append(query.size(0))
            assert kv == ('flat_k', 'flat_k_s')
            assert weights.size(0) == query.size(0)
            assert cu_seq_len_k_start.size(0) == query.size(0)
            assert cu_seq_len_k_end.size(0) == query.size(0)
            return query[:, :1] * 10 + torch.arange(4)

    deep_gemm = FakeDeepGemm()
    monkeypatch.setattr(cuda_nsa, '_get_deep_gemm', lambda: deep_gemm)

    impl = object.__new__(cuda_nsa.TritonNSAIndexFP8)
    impl.topk = 2
    impl.fill = -1
    impl.max_logits_bytes = 2 * 4 * 4
    flatten_calls = []

    def fake_flatten(indexer_k_cache, head_dim, meta):
        flatten_calls.append((indexer_k_cache, head_dim, meta))
        return 'flat_k', 'flat_k_s'

    def fake_topk(scores, q_seqlens, kv_seqlens, k, **kwargs):
        assert q_seqlens.tolist() == [5]
        assert kv_seqlens.numel() == scores.size(0)
        return torch.topk(scores, k).indices.to(torch.int32)

    impl._flatten_prefill_k = fake_flatten
    impl._sparse_index_topk = fake_topk
    q = torch.arange(5, dtype=torch.float32).unsqueeze(-1)
    q_s = torch.ones(5)
    indexer_k_cache = object()
    score_meta = cuda_nsa._DeepGemmContiguousScoreMeta(
        k_starts=torch.arange(5, dtype=torch.int32),
        k_ends=torch.arange(5, dtype=torch.int32) + 4,
        max_kv_seqlen=4,
    )
    meta = SimpleNamespace(
        score_meta=score_meta,
        indexer_kv_seqlens=torch.full((5, ), 4, dtype=torch.int32),
        q_seqlens=torch.tensor([5], dtype=torch.int32),
    )

    selected = impl._score_and_select(q, q_s, indexer_k_cache, meta)

    assert cuda_nsa._get_max_score_rows(4, impl.max_logits_bytes) == 2
    assert deep_gemm.row_counts == [2, 2, 1]
    assert len(flatten_calls) == 1
    assert flatten_calls[0][:2] == (indexer_k_cache, 1)
    assert torch.equal(selected, torch.tensor([[3, 2]] * 5, dtype=torch.int32))


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
    impl = cuda_nsa.TritonNSAIndexFP8(
        topk=2048, softmax_scale=1.0, block_size=128, fill=-1)

    flat_k, flat_k_s = impl._flatten_prefill_k(packed_cache, q.size(-1), meta)
    deepgemm_scores = impl._compute_prefill_scores(
        q, q_s, flat_k, flat_k_s, meta.score_meta)
    expected_topk = impl._select_topk(deepgemm_scores, meta)
    impl.max_logits_bytes = 2 * meta.max_kv_seqlen * 4
    chunked_topk = impl._score_and_select(q, q_s, packed_cache, meta)
    torch.testing.assert_close(chunked_topk.sort().values,
                               expected_topk.sort().values)

    meta.score_meta = None
    triton_scores = cuda_nsa.fp8_index(
        q,
        q_s,
        k_cache,
        k_s_cache[..., 0],
        meta.cu_seqlen_q,
        meta.k_seqlens,
        meta.block_offset,
        max_q_seqlen=meta.max_q_seqlen,
        max_k_seqlen=meta.max_kv_seqlen,
        causal=True,
    )
    assert deepgemm_scores.shape == triton_scores.shape == (5, 8)
    for row, row_len in enumerate(meta.indexer_kv_seqlens.tolist()):
        torch.testing.assert_close(
            deepgemm_scores[row, :row_len],
            triton_scores[row, :row_len],
            rtol=1e-3,
            atol=1e-3,
        )

    impl.max_logits_bytes = triton_scores.numel() * triton_scores.element_size() - 1
    with pytest.raises(RuntimeError, match='DeepGEMM installation is required'):
        impl._score_and_select(q, q_s, packed_cache, meta)


def test_indexer_meta_skips_reused_mtp_topk():
    assert nsa.should_skip_nsa_indexer([dict(skip_topk=True)])
    assert not nsa.should_skip_nsa_indexer([dict(skip_topk=False)])


@pytest.mark.parametrize(
    ('allow_skip', 'is_decoding', 'max_kv_seqlen', 'should_skip'),
    [(True, False, 2048, True), (True, False, 2049, False),
     (True, True, 2048, False), (False, False, 2048, False)])
def test_short_prefill_caches_indexer_k_before_optional_scoring(
        monkeypatch, allow_skip, is_decoding, max_kv_seqlen, should_skip):
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa

    prepare_q = torch.tensor(0)
    monkeypatch.setattr(cuda_nsa, '_get_dsa_indexer_k_cache_views',
                        lambda *_: (torch.empty(0), torch.empty(1, 1)))
    monkeypatch.setattr(cuda_nsa, 'prepare_dsa_indexer_q',
                        lambda *args, **kwargs: (prepare_q, prepare_q))

    k_cache_calls = []

    def prepare_k_cache(*args, **kwargs):
        k_cache_calls.append((args, kwargs))

    monkeypatch.setattr(cuda_nsa, 'prepare_dsa_indexer_k_cache',
                        prepare_k_cache)

    impl = object.__new__(cuda_nsa.TritonNSAIndexFP8)
    impl.topk = 2048
    impl.softmax_scale = 1.0
    impl._allow_short_prefill_scoring_skip = allow_skip
    selected = torch.tensor(1)
    score_calls = []

    def score_and_select(*args):
        score_calls.append(args)
        return selected

    impl._score_and_select = score_and_select
    meta = SimpleNamespace(
        is_decoding=is_decoding,
        max_kv_seqlen=max_kv_seqlen,
        cu_seqlen_q=torch.tensor([0, 1]),
        k_seqlens=torch.tensor([1]),
        block_offset=torch.tensor([[0]]),
        max_q_seqlen=1,
    )

    output = impl.forward_fused(
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        torch.empty(1),
        norm_eps=1e-6,
        head_gate_scale=1.0,
        rope_interleaved=False,
        meta=meta,
    )

    assert len(k_cache_calls) == 1
    assert len(score_calls) == (0 if should_skip else 1)
    if should_skip:
        assert output is None
    else:
        assert output is selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA backend')
def test_sparse_index_topk_is_resolved_at_init(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa

    selector = object()
    monkeypatch.setattr(cuda_nsa, '_get_sparse_index_topk',
                        lambda topk: selector)

    index_impl = cuda_nsa.TritonNSAIndexFP8(
        topk=512, softmax_scale=1.0, block_size=128, fill=-1)

    assert index_impl._sparse_index_topk is selector
