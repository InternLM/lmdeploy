# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cp_utils import get_dcp_local_causal_seq_lens
from lmdeploy.pytorch.config import CacheConfig


def test_dcp_prefill_chunks_cover_uneven_prefixes_with_bounded_workspace():
    from lmdeploy.pytorch.backends.cp_utils import build_dcp_prefill_chunks, get_dcp_prefill_workspace_size

    prefix_lens = torch.tensor([131072, 259, 0], dtype=torch.int32)
    plans = [
        build_dcp_prefill_chunks(prefix_lens=prefix_lens,
                                 prefix_limit=131072,
                                 block_size=64,
                                 head_dim=576,
                                 dcp_world_rank=(4, rank)) for rank in range(4)
    ]
    budget = get_dcp_prefill_workspace_size(batch_size=3, head_dim=576, block_size=64, dcp_size=4)
    # Incoming query length is deliberately absent from the planner. Short
    # continuations must not fall back to hundreds of 256-token collectives.
    assert len(plans[0]) < 64
    for rank, chunks in enumerate(plans):
        torch.testing.assert_close(sum(chunk.kv_seqlens for chunk in chunks), prefix_lens)
        for chunk, reference in zip(chunks, plans[0], strict=True):
            assert (chunk.start, chunk.size) == (reference.start, reference.size)
            assert chunk.start % 256 == chunk.size % 256 == 0
            assert 3 * (chunk.size // 4 + 2 * chunk.size) * 576 * 2 <= budget
            torch.testing.assert_close(chunk.local_kv_seqlens.sum(0), chunk.kv_seqlens.long())
            torch.testing.assert_close(chunk.local_cu_seqlens[1:] - chunk.local_cu_seqlens[:-1],
                                       chunk.local_kv_seqlens[rank])


@pytest.mark.parametrize('dcp_size', [2, 4])
@pytest.mark.parametrize('query_len', [1, 2, 6])
def test_dcp_local_causal_lengths_match_token_ownership(dcp_size, query_len):
    lengths = [0, query_len, 64 * dcp_size - 1, 64 * dcp_size + 1]
    kv_seqlens = torch.tensor(lengths, dtype=torch.int32)
    for rank in range(dcp_size):
        actual = get_dcp_local_causal_seq_lens(kv_seqlens, query_len, (dcp_size, rank))
        expected = [len(range(rank, end - query_len + row + 1, dcp_size))
                    for end in lengths for row in range(query_len)]
        assert actual.tolist() == expected


def test_dcp_block_allocation_uses_virtual_block_size():
    from lmdeploy.pytorch.engine.engine import _build_seq_meta
    from lmdeploy.pytorch.paging.block_manager import build_block_manager
    from lmdeploy.pytorch.strategies.ar.sequence import SchedulerSequenceDefault

    cache_config = CacheConfig(max_batches=4,
                               block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=8,
                               dcp=4)
    seq_meta = _build_seq_meta(SimpleNamespace(use_mrope=False),
                               cache_config,
                               seq_strategy=None,
                               sampling_strategy=None)
    session = SimpleNamespace(seq_meta=seq_meta)
    sequence = SchedulerSequenceDefault(seq_id=0, session=session)
    sequence._num_token_ids = 256
    block_manager = build_block_manager(cache_config)

    assert seq_meta.block_size == 64
    assert block_manager.num_required_blocks(sequence) == 1
    sequence._num_token_ids += 1
    assert block_manager.num_required_blocks(sequence) == 2


@pytest.mark.parametrize('is_decoding', [False, True])
def test_nsa_metadata_localizes_each_causal_row(is_decoding):
    from lmdeploy.pytorch.backends.nsa import build_nsa_index_meta

    q_seqlens = torch.tensor([2, 3], dtype=torch.int32)
    kv_seqlens = torch.tensor([5, 8], dtype=torch.int32)
    cu_q = torch.nn.functional.pad(q_seqlens.cumsum(0), (1, 0))
    sequence_metadata = SimpleNamespace(
        q_seqlens=q_seqlens,
        kv_seqlens=kv_seqlens,
        cu_seqlens_q=cu_q,
        block_offsets=torch.zeros(2, 1, dtype=torch.int32),
        max_kv_seqlen=8,
        kv_flatten_size=13,
    )
    meta = build_nsa_index_meta(num_tokens=5,
                                is_decoding=is_decoding,
                                block_size=64,
                                num_gpu_blocks=4,
                                sequence_metadata=sequence_metadata,
                                dcp_world_rank=(2, 1))

    assert meta.dcp_local_kv_seqlens.tolist() == [2, 4]
    assert meta.indexer_kv_seqlens.tolist() == [2, 2, 3, 3, 4]
    if is_decoding:
        assert meta.cu_seqlen_k is None
    else:
        assert meta.cu_seqlen_k.tolist() == [0, 2, 6]


def test_dcp_prefill_scoring_uses_global_sparse_boundary():
    from lmdeploy.pytorch.backends.cuda.nsa import TritonNSAIndexFP8Impl
    from lmdeploy.pytorch.backends.nsa import build_nsa_index_meta

    q_seqlens = torch.tensor([2], dtype=torch.int32)
    kv_seqlens = torch.tensor([3000], dtype=torch.int32)
    sequence_metadata = SimpleNamespace(
        q_seqlens=q_seqlens,
        kv_seqlens=kv_seqlens,
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        block_offsets=torch.zeros(1, 1, dtype=torch.int32),
        max_kv_seqlen=3000,
        kv_flatten_size=3000,
    )
    meta = build_nsa_index_meta(num_tokens=2,
                                is_decoding=False,
                                block_size=64,
                                num_gpu_blocks=100,
                                sequence_metadata=sequence_metadata,
                                dcp_world_rank=(2, 0))
    impl = object.__new__(TritonNSAIndexFP8Impl)
    impl.topk = 2048
    impl._allow_short_prefill_scoring_skip = True

    assert meta.max_kv_seqlen == 1500
    assert meta.global_max_kv_seqlen == 3000
    assert not impl._should_skip_scoring(meta)
