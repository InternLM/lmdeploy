# Copyright (c) OpenMMLab. All rights reserved.
"""CPU checks for GQA DCP prefill return values and temporary lifetimes."""
import weakref
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.attention.cp import DCPAttentionImpl
from lmdeploy.pytorch.backends.cuda.attention.default import TritonAttentionMetadata


@pytest.fixture
def prefill_impl(monkeypatch):
    # Exercise Python orchestration without initializing CUDA collectives.
    impl = object.__new__(DCPAttentionImpl)
    monkeypatch.setattr(impl, '_fill_kv_cache_impl', lambda *args, **kwargs: None)
    return impl


@pytest.mark.parametrize('tuple_return', [False, True], ids=['tensor_api', 'tuple_api'])
def test_fa3_fresh_prefill_returns_tensor(prefill_impl, tuple_return, monkeypatch):
    pytest.importorskip('flash_attn_interface')
    from lmdeploy.pytorch.third_party import flash_attn_interface

    query = torch.ones(2, 1, 4)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    prefill_impl.use_fa3 = True
    prefill_impl.scale = 0.5
    prefill_impl.logit_softcapping = 0.0

    def fa3(*args, **kwargs):
        assert kwargs.get('return_attn_probs', False) is False
        return (query, torch.zeros(1, 2)) if tuple_return else query

    monkeypatch.setattr(flash_attn_interface, '_flash_attn_varlen_func', fa3)
    prefill_impl._fa3_prefill = flash_attn_interface.flash_attn_varlen_func
    metadata = TritonAttentionMetadata(
        is_decoding=False, block_offsets=None, cu_seqlens_q=cu_q)
    result = prefill_impl.forward(query, query, query, None, None, metadata)
    assert result is query


def test_cached_prefill_releases_initial_state(prefill_impl, monkeypatch):
    from lmdeploy.pytorch.kernels.cuda import dcp

    query = torch.ones(2, 1, 4)
    cu_q = torch.tensor([0, 2], dtype=torch.int32)
    chunks = tuple(SimpleNamespace(size=2, cu_seqlens=cu_q) for _ in range(2))
    metadata = TritonAttentionMetadata(
        is_decoding=False, block_offsets=None, cu_seqlens_q=cu_q, dcp_prefix_chunks=chunks)
    initial_refs = []
    calls = 0

    def prefill(*args, **kwargs):
        nonlocal calls
        calls += 1
        output = torch.full_like(query, calls)
        lse = torch.zeros(query.shape[:2])
        if calls == 1:
            initial_refs.extend((weakref.ref(output), weakref.ref(lse)))
        return output, lse

    def gather(k_cache, v_cache, metadata, chunk, dtype, **kwargs):
        if chunk is chunks[1]:
            assert all(ref() is None for ref in initial_refs), 'initial attention state is still retained'
        return query, query

    # Allocate new merged states, as the real merge kernel does. The initial
    # states must be released before allocating the next chunk's K/V buffers.
    monkeypatch.setattr(dcp, 'merge_attention_states',
                        lambda output, lse, chunk_output, chunk_lse: (output + chunk_output, lse + chunk_lse))
    monkeypatch.setattr(prefill_impl, '_prefill_attention', prefill)
    monkeypatch.setattr(prefill_impl, '_gather_prefix', gather)
    result = prefill_impl.forward(query, query, query, None, None, metadata)
    assert calls == 3
    torch.testing.assert_close(result, torch.full_like(query, 6))
