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
    assert meta.max_kv_seqlen == 64 * 16


def test_indexer_meta_builds_causal_rows():
    meta = _build_indexer_meta(
        q_seqlens=torch.tensor([2, 3], dtype=torch.int32),
        kv_seqlens=torch.tensor([5, 8], dtype=torch.int32),
        is_decoding=False,
    )

    expected = torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32)
    assert torch.equal(meta.indexer_kv_seqlens, expected)


def test_indexer_meta_skips_reused_mtp_topk():
    assert nsa.should_skip_nsa_indexer([dict(skip_topk=True)])
    assert not nsa.should_skip_nsa_indexer([dict(skip_topk=False)])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA backend')
def test_sparse_index_topk_is_resolved_at_init(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import nsa as cuda_nsa

    selector = object()
    monkeypatch.setattr(cuda_nsa, '_get_sparse_index_topk',
                        lambda topk: selector)

    index_impl = cuda_nsa.TritonNSAIndexFP8(
        topk=512, softmax_scale=1.0, block_size=128, fill=-1)

    assert index_impl._sparse_index_topk is selector
