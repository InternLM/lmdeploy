# Copyright (c) OpenMMLab. All rights reserved.
"""Host-known prefill extents must never extract device scalars."""
from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from lmdeploy.pytorch.backends.cuda.attention.v4 import CudaV4AttentionMetadata
from lmdeploy.pytorch.backends.cuda.attention.v4_utils import build_prefill_token_meta


class NoHostRead(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs=None):
        name = getattr(func, '__name__', '')
        assert name not in {'item', 'tolist', 'cpu', 'numpy', 'nonzero', 'argwhere',
                            '__bool__', '__int__', '__float__', '__index__'}, name
        assert not (name == 'where' and len(args) == 1), 'one-argument where'
        if name == 'to':
            targets = list(args[1:]) + [(kwargs or {}).get('device')]
            assert not any((isinstance(d, str) and d.startswith('cpu'))
                           or (isinstance(d, torch.device) and d.type == 'cpu') for d in targets), 'D2H copy'
        return func(*args, **(kwargs or {}))


class NoDynamicScalar(TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert not any(op in str(func) for op in ('_local_scalar_dense', 'nonzero', 'masked_select'))
        return func(*args, **(kwargs or {}))


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    return request.param


@pytest.mark.parametrize('lengths', [[], [0, 0], [1], [0, 2, 0, 3, 0], [2, 137, 1], [4096]])
@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('supply_cu', [False, True])
def test_prefill_token_mapping(device, lengths, dtype, supply_cu):
    q = torch.tensor(lengths, dtype=dtype, device=device)
    cu = torch.tensor([0] + list(accumulate(lengths)), dtype=dtype, device=device)
    with NoHostRead(), NoDynamicScalar():
        got = build_prefill_token_meta(q, cu if supply_cu else None, total_tokens=sum(lengths))
    expected_seq = [i for i, n in enumerate(lengths) for _ in range(n)]
    expected_pos = [p for n in lengths for p in range(n)]
    torch.testing.assert_close(got.seq_id, torch.tensor(expected_seq, dtype=torch.long, device=device))
    expected_pos = torch.tensor(expected_pos, dtype=dtype if supply_cu else torch.int64, device=device)
    torch.testing.assert_close(got.token_pos, expected_pos)


def test_prefill_extent_contract(device):
    q = torch.tensor([2, 3], dtype=torch.int32, device=device)
    extent = torch.tensor(5, device=device)
    with NoHostRead(), NoDynamicScalar():
        with pytest.raises(TypeError, match='total_tokens'):
            build_prefill_token_meta(q)
        with pytest.raises(TypeError, match='host int'):
            build_prefill_token_meta(q, total_tokens=extent)
        with pytest.raises(ValueError, match='non-negative'):
            build_prefill_token_meta(q, total_tokens=-1)


@pytest.mark.parametrize('rectangular', [False, True])
def test_prefill_metadata_context_and_graph(device, rectangular, monkeypatch):
    # Retain all metadata operations; only the external DeepGEMM scheduler is
    # stubbed. Ragged total 5 != B * max_q 12. Rectangular includes padded rows.
    lengths = [3, 3, 3, 3] if rectangular else [2, 0, 3, 0]
    q = torch.tensor(lengths, dtype=torch.int32, device=device)
    cu = torch.tensor([0] + list(accumulate(lengths)), dtype=torch.int32, device=device)
    history = torch.tensor([133, 0, 260, 0], dtype=torch.int32, device=device)
    kv = history + q
    slots = torch.tensor([2, -1, 0, -1], device=device)
    attn = SimpleNamespace(is_decoding=rectangular, q_seqlens=q, kv_seqlens=kv,
                           cu_seqlens_q=cu, cu_seqlens_k=cu,
                           block_offsets=torch.zeros((4, 8), dtype=torch.int32, device=device))
    step = SimpleNamespace(input_ids=torch.empty((1, sum(lengths)), dtype=torch.long, device=device),
                           max_q_seqlen=max(lengths), max_kv_seqlen=300, sum_kv_seqlen=600,
                           cache_config=SimpleNamespace(block_size=64, num_gpu_blocks=32, max_session_len=512))
    if rectangular:
        # Graph capacity must not accidentally depend on an unpadded context.
        step.input_ids = torch.empty((1, 6), dtype=torch.long, device=device)
    monkeypatch.setattr(CudaV4AttentionMetadata, '_build_index_score_meta',
                        staticmethod(lambda *args, **kwargs: None))

    def build():
        with NoHostRead(), NoDynamicScalar():
            return CudaV4AttentionMetadata.from_step_context(
                attn, step, window_size=128, ring_storage_capacity=134, slot=slots, causal=not rectangular)

    def check(got):
        seq = torch.tensor([i for i, n in enumerate(lengths) for _ in range(n)], device=device)
        pos = torch.tensor([p for n in lengths for p in range(n)], dtype=torch.int32, device=device)
        torch.testing.assert_close(got.prefill_shared.token_seq, seq)
        torch.testing.assert_close(got.prefill_shared.token_pos, pos)
        abs_pos = (kv.long() - q.long())[seq] + pos
        ring = torch.where(abs_pos < (kv[seq] - 128).clamp(min=0), -1, abs_pos % 134)
        torch.testing.assert_close(got.prefill_window.slot, slots[seq])
        torch.testing.assert_close(got.prefill_window.ring_pos, ring)
        assert got.is_rectangular_decode == rectangular

    check(build())
    if device == 'cuda':
        old_sync_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode(2)
            got = build()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                got = build()
            torch.cuda.set_sync_debug_mode(old_sync_mode)
            for delta in [1, 129, 134]:
                kv.add_(delta)
                if not rectangular:
                    # Same capacity, different ragged mapping on replay.
                    lengths[:] = [1, 1, 0, 3]
                    q.copy_(torch.tensor(lengths, dtype=q.dtype, device=device))
                    cu.copy_(torch.tensor([0, 1, 2, 2, 5], dtype=cu.dtype, device=device))
                torch.cuda.set_sync_debug_mode(2)
                graph.replay()
                torch.cuda.set_sync_debug_mode(old_sync_mode)
                check(got)
        finally:
            torch.cuda.set_sync_debug_mode(old_sync_mode)


@pytest.mark.parametrize('operation', [lambda x: x.item(), lambda x: x.tolist(), lambda x: torch.nonzero(x),
                                       lambda x: torch.where(x > 0), lambda x: bool(x),
                                       lambda x: torch.arange(x), lambda x: x.to('cpu')])
def test_no_host_read_guard_negative_controls(operation):
    with pytest.raises(AssertionError), NoHostRead(), NoDynamicScalar():
        operation(torch.tensor(1))
