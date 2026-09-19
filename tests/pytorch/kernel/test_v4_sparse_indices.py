# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='CUDA is required for V4 sparse-index kernels')


def _pad_ref(indices: torch.Tensor, block: int):
    topk = indices.size(-1)
    padded_topk = ((topk + block - 1) // block) * block
    if padded_topk == topk:
        return indices
    return torch.nn.functional.pad(indices, (0, padded_topk - topk), value=-1)


@torch.inference_mode()
def test_rectangular_split_scheduler_partitions_actual_kv_blocks():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_split_scheduler,
    )

    primary = torch.tensor([0, 65, 512, 33], dtype=torch.int32,
                           device='cuda')
    extra = torch.tensor([1, 128, 90, 128], dtype=torch.int32,
                         device='cuda')
    num_partitions = 7
    metadata = torch.empty((num_partitions, 8), dtype=torch.int32,
                           device='cuda')
    num_splits = torch.empty((primary.numel() + 1,), dtype=torch.int32,
                             device='cuda')

    build_rectangular_decode_split_scheduler(
        primary, extra, metadata, num_splits)

    # P=7 and B=4 gives one base partition per request, with the remaining
    # three assigned to the first three requests that have another block.
    # Each request's blocks are then divided into balanced local chunks.
    assert num_splits.cpu().tolist() == [0, 2, 4, 6, 7]
    assert metadata.cpu().tolist() == [
        [0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 1, 2, 1, 1, 1, 0],
        [1, 1, 0, 2, 0, 1, 1, 0],
        [1, 1, 2, 4, 1, 1, 1, 0],
        [2, 2, 0, 5, 0, 1, 1, 0],
        [2, 2, 5, 10, 1, 1, 1, 0],
        [3, 3, 0, 3, 0, 0, 0, 0],
    ]


@torch.inference_mode()
def test_rectangular_split_scheduler_spans_requests_when_partitions_are_fewer():
    """Cover the production B=32/P=22 scheduler branch."""
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_split_scheduler,
    )

    batch_size = 32
    num_partitions = 22
    primary = torch.tensor(
        [0, 65, 129, 257] * 8, dtype=torch.int32, device='cuda')
    extra = torch.tensor(
        [1, 64, 65, 128] * 8, dtype=torch.int32, device='cuda')
    metadata = torch.empty((num_partitions, 8), dtype=torch.int32,
                           device='cuda')
    num_splits = torch.empty((batch_size + 1,), dtype=torch.int32,
                             device='cuda')

    build_rectangular_decode_split_scheduler(
        primary, extra, metadata, num_splits)

    primary_blocks = torch.clamp(
        torch.div(primary.cpu() + 63, 64, rounding_mode='floor'), min=1)
    extra_blocks = torch.div(extra.cpu() + 63, 64, rounding_mode='floor')
    blocks = (primary_blocks + extra_blocks).tolist()
    total_blocks = sum(blocks)
    payload = (total_blocks + num_partitions - 1) // num_partitions
    prefixes = [0]
    for count in blocks:
        prefixes.append(prefixes[-1] + count)

    expected_splits = [0]
    for request, count in enumerate(blocks):
        first = prefixes[request] // payload
        last = (prefixes[request] + count - 1) // payload
        expected_splits.append(expected_splits[-1] + last - first + 1)
    assert num_splits.cpu().tolist() == expected_splits

    rows = metadata.cpu().tolist()
    covered = []
    for partition, row in enumerate(rows):
        begin_request, end_request, begin_block, end_block = row[:4]
        if begin_request == batch_size:
            assert partition * payload >= total_blocks
            continue
        global_begin = prefixes[begin_request] + begin_block
        global_end = prefixes[end_request] + end_block
        assert global_begin == partition * payload
        assert global_end == min((partition + 1) * payload, total_blocks)
        covered.extend(range(global_begin, global_end))
    assert covered == list(range(total_blocks))


@torch.inference_mode()
def test_rectangular_window_indices_preserve_wrapped_history_and_candidate_prefix():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_window_sparse_indices,
    )

    start_pos = torch.tensor([130, 5, 99], dtype=torch.int64,
                             device='cuda')
    slot = torch.tensor([7, 8, -1], dtype=torch.int64, device='cuda')
    q_seqlens = torch.tensor([6, 6, 6], dtype=torch.int64, device='cuda')
    window_size = 128
    query_len = 6
    ring_storage_capacity = window_size + query_len

    (indices, lengths, write_slot, write_pos, token_total_lens, disabled,
     disabled_len) = build_rectangular_decode_window_sparse_indices(
         start_pos, q_seqlens, slot, query_len, window_size,
         ring_storage_capacity)

    # At absolute position 130 the visible window is 3..130. Candidate writes
    # through 135 cannot alias that history in the W+N physical ring.
    base0 = 7 * ring_storage_capacity
    expected0 = [base0 + (i % ring_storage_capacity)
                 for i in range(3, 131)]
    assert indices[0, 0, :128].tolist() == expected0

    # At verifier position 5 the logical window advances to 8..135 while the
    # physical ring wraps at 134, independently of the logical W=128 bound.
    expected5 = [base0 + (i % ring_storage_capacity)
                 for i in range(8, 136)]
    assert indices[5, 0, :128].tolist() == expected5
    base1 = 8 * ring_storage_capacity
    assert indices[query_len + 5, 0, :11].tolist() == [
        base1 + i for i in range(11)]
    assert lengths.reshape(3, query_len)[:, [0, 5]].tolist() == [
        [128, 128], [6, 11], [1, 1]]
    assert write_slot.reshape(3, query_len)[2].tolist() == [-1] * query_len
    assert write_pos.reshape(3, query_len)[0].tolist() == [130, 131, 132, 133, 0, 1]
    assert token_total_lens.reshape(3, query_len)[0].tolist() == [131, 132, 133, 134, 135, 136]

    # A padded graph row exposes one in-bounds entry and disables the
    # compressed-cache path.  Its remaining indices are ignored.
    assert torch.all(indices[2 * query_len:, 0, 0] == 0)
    assert torch.all(disabled[2 * query_len:, 0, 0] == 0)
    assert torch.all(disabled[:2 * query_len] == -1)
    assert indices.dtype == lengths.dtype == disabled.dtype \
        == disabled_len.dtype == torch.int32
    assert disabled_len.reshape(3, query_len)[:, 0].tolist() == [0, 0, 1]

    # The whole simultaneously-live interval maps injectively into R=W+N.
    for start in (0, 127, 128, 130):
        live = range(max(0, start - window_size + 1), start + query_len)
        physical = [p % ring_storage_capacity for p in live]
        assert len(physical) == len(set(physical))


@torch.inference_mode()
def test_dflash_v4_geometry_accepts_multirow_target_verification():
    from types import SimpleNamespace

    from lmdeploy.pytorch.configurations.deepseek_v4 import (
        get_v4_ring_geometry,
    )
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_window_sparse_indices,
    )

    geometry = get_v4_ring_geometry(
        SimpleNamespace(sliding_window=128),
        spec_method='dflash',
        num_spec_tokens=5,
    )
    output = build_rectangular_decode_window_sparse_indices(
        start_pos=torch.tensor([130], dtype=torch.long, device='cuda'),
        q_seqlens=torch.tensor([6], dtype=torch.long, device='cuda'),
        slot=torch.tensor([0], dtype=torch.long, device='cuda'),
        query_len=6,
        window_size=geometry.logical_window_size,
        ring_storage_capacity=geometry.ring_storage_capacity,
    )

    assert geometry.ring_storage_capacity == 134
    assert output[0].shape == (6, 1, 128)
    assert output[1].tolist() == [128] * 6


@torch.inference_mode()
def test_build_decode_window_sparse_indices_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_decode_window_sparse_indices

    kv_seqlens = torch.tensor([3, 9, 17], dtype=torch.int32, device='cuda')
    start_pos = torch.tensor([2, 8, 16], dtype=torch.int64, device='cuda')
    slot = torch.tensor([2, -1, 0], dtype=torch.int64, device='cuda')
    window_size = 6
    ring_storage_capacity = 8
    block = 4

    indices, topk_length, window_pos, disabled_indices, disabled_topk_length = build_decode_window_sparse_indices(
        kv_seqlens, start_pos, slot, window_size, ring_storage_capacity,
        block=block)

    cols = torch.arange(window_size, dtype=torch.int32, device='cuda').unsqueeze(0)
    window_lens = kv_seqlens.clamp(max=window_size)
    first_abs = kv_seqlens - window_lens
    valid = cols < window_lens.unsqueeze(1)
    ref_indices = torch.remainder(first_abs.unsqueeze(1) + cols,
                                  ring_storage_capacity)
    safe_slot = slot.clamp_min(0)
    ref_indices = torch.where(
        valid,
        ref_indices + safe_slot.to(torch.int32).unsqueeze(1)
        * ring_storage_capacity,
        torch.full((), -1, dtype=torch.int32, device='cuda'))
    ref_indices[slot < 0, 0] = 0
    ref_indices = _pad_ref(ref_indices.unsqueeze(1), block=block).to(torch.int32)
    ref_topk_length = torch.where(slot < 0, torch.ones_like(window_lens), window_lens)
    ref_window_pos = torch.remainder(start_pos, ring_storage_capacity).to(torch.int32)
    ref_disabled_indices = torch.full((3, 1, block), -1, dtype=torch.int32, device='cuda')
    ref_disabled_indices[slot < 0, 0, 0] = 0
    ref_disabled_topk_length = torch.where(
        slot < 0,
        torch.ones_like(window_lens),
        torch.zeros_like(window_lens))

    torch.testing.assert_close(indices.cpu(), ref_indices.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())
    torch.testing.assert_close(window_pos.cpu(), ref_window_pos.cpu())
    torch.testing.assert_close(disabled_indices.cpu(), ref_disabled_indices.cpu())
    torch.testing.assert_close(disabled_topk_length.cpu(), ref_disabled_topk_length.cpu())


@torch.inference_mode()
def test_build_decode_compressed_sparse_indices_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_decode_compressed_sparse_indices

    logical_topk = torch.tensor(
        [[[0, 1, 4, -1, 99]], [[2, 7, 8, 11, -1]]],
        dtype=torch.int32,
        device='cuda')
    block_offsets = torch.tensor(
        [[7, 3, 9], [4, 8, 2]],
        dtype=torch.int64,
        device='cuda')
    block_size = 16
    compress_ratio = 4
    block = 4

    out = build_decode_compressed_sparse_indices(
        logical_topk, block_offsets, block_size, compress_ratio, block=block)

    bsz = logical_topk.size(0)
    safe_logical = logical_topk.clamp(min=0)
    token_positions = safe_logical * compress_ratio
    block_idx = torch.div(token_positions, block_size, rounding_mode='floor')
    max_block_idx = block_offsets.size(1)
    safe_block_idx = block_idx.clamp(max=max_block_idx - 1)
    block_idx_valid = block_idx < max_block_idx
    phys_block = block_offsets.gather(1, safe_block_idx.view(bsz, -1)).view_as(logical_topk)
    entries_per_block = block_size // compress_ratio
    block_off = torch.remainder(safe_logical, entries_per_block)
    phys_indices = phys_block * entries_per_block + block_off
    valid = (logical_topk >= 0) & block_idx_valid
    ref = torch.where(valid, phys_indices, phys_indices.new_full((), -1))
    ref = _pad_ref(ref, block=block).to(torch.int32)

    assert out.shape == (2, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@torch.inference_mode()
def test_build_decode_prefix_compressed_sparse_indices_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_decode_prefix_compressed_sparse_indices

    num_compressed = torch.tensor([3, 5], dtype=torch.int32, device='cuda')
    block_offsets = torch.tensor(
        [[7, 3, 9], [4, 8, 2]],
        dtype=torch.int64,
        device='cuda')
    block_size = 16
    compress_ratio = 4
    block = 4
    max_topk = 6

    out = build_decode_prefix_compressed_sparse_indices(
        num_compressed, block_offsets, block_size, compress_ratio, max_topk=max_topk, block=block)

    cols = torch.arange(8, dtype=torch.int64, device='cuda').unsqueeze(0)
    entries_per_block = block_size // compress_ratio
    token_positions = cols * compress_ratio
    block_idx = torch.div(token_positions, block_size, rounding_mode='floor')
    phys_block = block_offsets.gather(1, block_idx.clamp(max=block_offsets.size(1) - 1).expand(2, -1))
    block_off = torch.remainder(cols, entries_per_block)
    phys = phys_block * entries_per_block + block_off
    valid = cols < num_compressed.to(torch.int64).unsqueeze(1)
    ref = torch.where(valid, phys, phys.new_full((), -1)).unsqueeze(1).to(torch.int32)

    assert out.shape == (2, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())


@torch.inference_mode()
def test_rectangular_compressed_indices_use_request_page_table():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_compressed_sparse_indices,
        build_rectangular_decode_prefix_compressed_sparse_indices,
    )

    query_len = 3
    block_size = 16
    ratio = 4
    block = 4
    block_offsets = torch.tensor([[7, 3], [4, 8]], dtype=torch.int64,
                                 device='cuda')
    logical = torch.tensor([
        [0, 1, -1], [1, 2, -1], [2, 3, -1],
        [0, 1, -1], [1, 2, -1], [2, 3, -1],
    ], dtype=torch.int32, device='cuda')
    row_is_padded = torch.tensor([False, False, False, False, False, True],
                                 device='cuda')

    out = build_rectangular_decode_compressed_sparse_indices(
        logical, block_offsets, block_size, ratio, query_len,
        row_is_padded=row_is_padded, block=block)
    request_ids = torch.arange(6, device='cuda') // query_len
    safe = logical.clamp_min(0)
    block_idx = torch.div(safe * ratio, block_size,
                          rounding_mode='floor')
    phys_block = block_offsets[request_ids[:, None], block_idx]
    ref = phys_block * (block_size // ratio) \
        + torch.remainder(safe, block_size // ratio)
    ref = torch.where(logical >= 0, ref, ref.new_full((), -1))
    ref = _pad_ref(ref.unsqueeze(1), block)
    ref[-1, 0, 0] = 0
    torch.testing.assert_close(out.cpu(), ref.to(torch.int32).cpu())

    token_total_lens = torch.tensor([1, 4, 8, 5, 9, 12],
                                    dtype=torch.int32, device='cuda')
    prefix = build_rectangular_decode_prefix_compressed_sparse_indices(
        token_total_lens, block_offsets, block_size, ratio, query_len,
        max_topk=3, row_is_padded=row_is_padded, block=block)
    cols = torch.arange(4, device='cuda', dtype=torch.int64).unsqueeze(0)
    num_compressed = torch.div(token_total_lens, ratio,
                               rounding_mode='floor').unsqueeze(1)
    prefix_block = torch.div(cols * ratio, block_size,
                             rounding_mode='floor')
    prefix_phys_block = block_offsets[
        request_ids[:, None], prefix_block.expand(6, -1)]
    prefix_ref = prefix_phys_block * (block_size // ratio) \
        + torch.remainder(cols, block_size // ratio)
    prefix_ref = torch.where(cols < num_compressed, prefix_ref,
                             prefix_ref.new_full((), -1)).unsqueeze(1)
    prefix_ref[-1, 0, 0] = 0
    torch.testing.assert_close(prefix.cpu(), prefix_ref.to(torch.int32).cpu())


@torch.inference_mode()
@pytest.mark.parametrize(('batch_size', 'query_len'), [
    (1, 2),
    (3, 6),
    (8, 2),
    (8, 6),
])
def test_rectangular_executor_matches_position_loop_with_one_flashmla_call(
        monkeypatch, batch_size, query_len):
    """The single-call path preserves the qlen-one attention reference."""
    flash_mla = pytest.importorskip('flash_mla')
    from types import SimpleNamespace

    from lmdeploy.pytorch.backends.cuda.attention.v4 import (
        _V4RectangularDecodeExecutor,
    )
    from lmdeploy.pytorch.kernels.cuda.v4_pack_window import (
        pack_window_tokens_fp8,
    )
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_window_sparse_indices,
    )

    window_size = 128
    ring_capacity = window_size + query_len - 1
    ring_storage_capacity = (ring_capacity + 1) // 2 * 2
    num_slots = batch_size + 2
    num_heads = 64
    head_dim = 512
    packed_dim = 584
    boundary_starts = [0, 127, 128, 130]
    starts = torch.tensor(
        [boundary_starts[i % len(boundary_starts)]
         for i in range(batch_size)], dtype=torch.long, device='cuda')
    # Include odd state slots: FlashMLA's split value/scale layout requires an
    # aligned physical ring extent for these slot bases.
    slots = torch.arange(1, batch_size + 1, dtype=torch.long,
                         device='cuda')
    q_seqlens = torch.full((batch_size,), query_len, dtype=torch.long,
                           device='cuda')
    (indices, lengths, write_slot, write_pos, token_total_lens,
     disabled_indices, disabled_lengths) = \
        build_rectangular_decode_window_sparse_indices(
            starts, q_seqlens, slots, query_len, window_size,
            ring_storage_capacity)
    rectangular = SimpleNamespace(
        is_padded=write_slot < 0,
        write_slot=write_slot,
        write_pos=write_pos,
        token_total_lens=token_total_lens,
        indices=indices,
        topk_length=lengths,
        disabled_indices=disabled_indices,
        disabled_topk_length=disabled_lengths,
    )
    meta = SimpleNamespace(
        rectangular_decode=rectangular,
        max_q_seqlen=query_len,
        q_seqlens=q_seqlens,
        is_cuda_graph=False,
    )

    generator = torch.Generator(device='cuda').manual_seed(123)
    initial = torch.zeros(num_slots, ring_storage_capacity, packed_dim,
                          dtype=torch.float8_e4m3fn, device='cuda')
    history_slots = []
    history_pos = []
    for start, slot in zip(starts.tolist(), slots.tolist()):
        history_slots.extend([slot] * start)
        history_pos.extend([pos % ring_storage_capacity
                            for pos in range(start)])
    history_kv = torch.randn(len(history_slots), head_dim,
                             dtype=torch.bfloat16, device='cuda',
                             generator=generator)
    pack_window_tokens_fp8(
        history_kv, initial,
        torch.tensor(history_slots, dtype=torch.long, device='cuda'),
        torch.tensor(history_pos, dtype=torch.long, device='cuda'))
    candidate_kv = torch.randn(batch_size * query_len, head_dim,
                               dtype=torch.bfloat16, device='cuda',
                               generator=generator)
    query = torch.randn(1, batch_size * query_len, num_heads, head_dim,
                        dtype=torch.bfloat16, device='cuda',
                        generator=generator)
    sink = torch.randn(num_heads, dtype=torch.float32, device='cuda',
                       generator=generator)

    reference_cache = initial.clone()
    pack_window_tokens_fp8(candidate_kv, reference_cache, write_slot,
                           write_pos)

    def call_flash(q, extra_indices, extra_lengths, primary_indices,
                   primary_lengths, cache):
        schedule, _ = flash_mla.get_mla_metadata()
        return flash_mla.flash_mla_with_kvcache(
            q,
            k_cache=cache.unsqueeze(2),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=head_dim,
            tile_scheduler_metadata=schedule,
            softmax_scale=head_dim**-0.5,
            causal=False,
            is_fp8_kvcache=True,
            indices=primary_indices,
            attn_sink=sink,
            extra_k_cache=cache.unsqueeze(2),
            extra_indices_in_kvcache=extra_indices,
            topk_length=primary_lengths,
            extra_topk_length=extra_lengths,
        )[0]

    reference_rows = []
    for position in range(query_len):
        reference_rows.append(call_flash(
            query[:, position::query_len].transpose(0, 1).contiguous(),
            indices[position::query_len].contiguous(),
            lengths[position::query_len].contiguous(),
            disabled_indices[position::query_len].contiguous(),
            disabled_lengths[position::query_len].contiguous(),
            reference_cache,
        ).transpose(0, 1))
    reference = torch.stack(reference_rows, dim=2).flatten(1, 2)

    impl = SimpleNamespace(
        head_size=head_dim,
        scale=head_dim**-0.5,
        compress_ratio=0,
        flash_mla=flash_mla,
        ring_storage_capacity=ring_storage_capacity,
        window_size=window_size,
        compressed_kv_cache_name=None,
        _pack_window_fp8=pack_window_tokens_fp8,
    )
    executor = _V4RectangularDecodeExecutor(impl)
    calls = 0
    original_call = flash_mla.flash_mla_with_kvcache

    def counted_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_call(*args, **kwargs)

    monkeypatch.setattr(flash_mla, 'flash_mla_with_kvcache', counted_call)
    actual = executor.forward(
        query, candidate_kv.unsqueeze(0), sink, meta,
        initial, {}, slots)

    assert calls == 1
    torch.testing.assert_close(actual, reference, atol=4e-3, rtol=5e-3)

    # The first call allocates FlashMLA's architecture-specific metadata. The
    # next call uses the custom split tree and is exact for this uncompressed
    # one-block-per-partition case.
    calls = 0
    actual = executor.forward(
        query, candidate_kv.unsqueeze(0), sink, meta,
        initial, {}, slots)
    assert calls == 1
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@torch.inference_mode()
@pytest.mark.parametrize(('batch_size', 'compress_ratio'), [
    (3, 4),
    (3, 128),
    (32, 4),
])
def test_rectangular_executor_compressed_paths_match_position_loop(
        monkeypatch, batch_size, compress_ratio):
    """Cover r4 indexer, r128 fallback, ragged/padded, and P<B paths."""
    flash_mla = pytest.importorskip('flash_mla')
    from types import SimpleNamespace

    from lmdeploy.pytorch.backends.cuda.attention.v4 import (
        _V4RectangularDecodeExecutor,
    )
    from lmdeploy.pytorch.backends.indexer import V4IndexerOutput
    from lmdeploy.pytorch.kernels.cuda.v4_pack_window import (
        pack_window_tokens_fp8,
    )
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import (
        build_rectangular_decode_compressed_sparse_indices,
        build_rectangular_decode_prefix_compressed_sparse_indices,
        build_rectangular_decode_window_sparse_indices,
    )

    query_len = 6
    window_size = 128
    ring_storage_capacity = 134
    block_size = 256
    entries_per_block = block_size // compress_ratio
    num_heads = 64
    head_dim = 512
    packed_dim = 584
    num_slots = batch_size + 1
    starts = torch.tensor(
        [[0, 127, 128, 130][i % 4] for i in range(batch_size)],
        dtype=torch.long,
        device='cuda')
    slots = torch.arange(batch_size, dtype=torch.long, device='cuda')
    slots[-1] = -1
    q_seqlens = torch.full((batch_size,), query_len, dtype=torch.long,
                           device='cuda')
    if batch_size > 2:
        q_seqlens[-2] = query_len - 2
    (extra_indices, extra_lengths, write_slot, write_pos, token_total_lens,
     disabled_indices, disabled_lengths) = \
        build_rectangular_decode_window_sparse_indices(
            starts, q_seqlens, slots, query_len, window_size,
            ring_storage_capacity)
    rectangular = SimpleNamespace(
        is_padded=write_slot < 0,
        write_slot=write_slot,
        write_pos=write_pos,
        token_total_lens=token_total_lens,
        indices=extra_indices,
        topk_length=extra_lengths,
        disabled_indices=disabled_indices,
        disabled_topk_length=disabled_lengths,
    )

    generator = torch.Generator(device='cuda').manual_seed(
        1000 + batch_size + compress_ratio)
    window_cache = torch.zeros(
        num_slots, ring_storage_capacity, packed_dim,
        dtype=torch.float8_e4m3fn, device='cuda')
    history_slots = []
    history_pos = []
    for start, slot in zip(starts.tolist(), slots.tolist()):
        if slot < 0:
            continue
        history_slots.extend([slot] * start)
        history_pos.extend([pos % ring_storage_capacity
                            for pos in range(start)])
    history_kv = torch.randn(
        len(history_slots), head_dim, dtype=torch.bfloat16, device='cuda',
        generator=generator)
    pack_window_tokens_fp8(
        history_kv,
        window_cache,
        torch.tensor(history_slots, dtype=torch.long, device='cuda'),
        torch.tensor(history_pos, dtype=torch.long, device='cuda'))

    candidate_kv = torch.randn(
        batch_size * query_len, head_dim, dtype=torch.bfloat16,
        device='cuda', generator=generator)
    query = torch.randn(
        1, batch_size * query_len, num_heads, head_dim,
        dtype=torch.bfloat16, device='cuda', generator=generator)
    sink = torch.randn(num_heads, dtype=torch.float32, device='cuda',
                       generator=generator)
    reference_window = window_cache.clone()
    pack_window_tokens_fp8(candidate_kv, reference_window, write_slot,
                           write_pos)

    # Build a valid packed FP8 compressed cache and a one-page table per
    # request. r4 consumes simulated indexer output; r128 consumes the prefix
    # fallback precomputed in attention metadata.
    compressed_3d = torch.zeros(
        batch_size, entries_per_block, packed_dim,
        dtype=torch.float8_e4m3fn, device='cuda')
    compressed_kv = torch.randn(
        batch_size * entries_per_block, head_dim,
        dtype=torch.bfloat16, device='cuda', generator=generator)
    pack_window_tokens_fp8(
        compressed_kv,
        compressed_3d,
        torch.arange(batch_size, device='cuda').repeat_interleave(
            entries_per_block),
        torch.arange(entries_per_block, device='cuda').repeat(batch_size))
    compressed_cache = compressed_3d
    block_offsets = torch.arange(
        batch_size, dtype=torch.long, device='cuda').unsqueeze(1)
    num_compressed = torch.div(
        token_total_lens, compress_ratio, rounding_mode='floor').to(
            torch.int32)
    num_compressed = torch.where(
        rectangular.is_padded, torch.ones_like(num_compressed),
        num_compressed)
    max_topk = entries_per_block
    if compress_ratio == 4:
        cols = torch.arange(max_topk, dtype=torch.int32,
                            device='cuda').unsqueeze(0)
        logical = torch.where(
            cols < num_compressed.unsqueeze(1), cols,
            torch.full_like(cols, -1))
        index_out = V4IndexerOutput(
            indices_in_kvcache=logical,
            topk_length=num_compressed,
        )
        primary_indices = build_rectangular_decode_compressed_sparse_indices(
            logical,
            block_offsets,
            block_size,
            compress_ratio,
            query_len,
            row_is_padded=rectangular.is_padded,
        )
    else:
        index_out = None
        primary_indices = \
            build_rectangular_decode_prefix_compressed_sparse_indices(
                token_total_lens,
                block_offsets,
                block_size,
                compress_ratio,
                query_len,
                max_topk=max_topk,
                row_is_padded=rectangular.is_padded,
            )
    ratio_meta = SimpleNamespace(
        decode=SimpleNamespace(indices=primary_indices,
                               topk_length=num_compressed))
    meta = SimpleNamespace(
        rectangular_decode=rectangular,
        max_q_seqlen=query_len,
        q_seqlens=q_seqlens,
        block_offsets=block_offsets,
        block_size=block_size,
        get_ratio_meta=lambda ratio: ratio_meta,
    )

    def call_flash(q, row_indices, row_lengths, row_extra_indices,
                   row_extra_lengths):
        schedule, _ = flash_mla.get_mla_metadata()
        return flash_mla.flash_mla_with_kvcache(
            q,
            k_cache=compressed_cache.unsqueeze(2),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=head_dim,
            tile_scheduler_metadata=schedule,
            softmax_scale=head_dim**-0.5,
            causal=False,
            is_fp8_kvcache=True,
            indices=row_indices,
            attn_sink=sink,
            extra_k_cache=reference_window.unsqueeze(2),
            extra_indices_in_kvcache=row_extra_indices,
            topk_length=row_lengths,
            extra_topk_length=row_extra_lengths,
        )[0]

    reference_rows = []
    for position in range(query_len):
        reference_rows.append(call_flash(
            query[:, position::query_len].transpose(0, 1).contiguous(),
            primary_indices[position::query_len].contiguous(),
            num_compressed[position::query_len].contiguous(),
            extra_indices[position::query_len].contiguous(),
            extra_lengths[position::query_len].contiguous(),
        ).transpose(0, 1))
    reference = torch.stack(reference_rows, dim=2).flatten(1, 2)

    impl = SimpleNamespace(
        head_size=head_dim,
        scale=head_dim**-0.5,
        compress_ratio=compress_ratio,
        flash_mla=flash_mla,
        ring_storage_capacity=ring_storage_capacity,
        window_size=window_size,
        compressed_kv_cache_name={
            4: 'v4_compressed_kv_r4_fp8',
            128: 'v4_compressed_kv_r128_fp8',
        }[compress_ratio],
        _pack_window_fp8=pack_window_tokens_fp8,
    )
    executor = _V4RectangularDecodeExecutor(impl)
    calls = 0
    original_call = flash_mla.flash_mla_with_kvcache

    def counted_call(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_call(*args, **kwargs)

    monkeypatch.setattr(flash_mla, 'flash_mla_with_kvcache', counted_call)
    actual = executor.forward(
        query,
        candidate_kv.unsqueeze(0),
        sink,
        meta,
        window_cache,
        {impl.compressed_kv_cache_name: compressed_cache},
        slots,
        index_out=index_out,
    )
    assert calls == 1
    torch.testing.assert_close(actual, reference, atol=4e-3, rtol=5e-3)

    calls = 0
    actual = executor.forward(
        query,
        candidate_kv.unsqueeze(0),
        sink,
        meta,
        window_cache,
        {impl.compressed_kv_cache_name: compressed_cache},
        slots,
        index_out=index_out,
    )
    assert calls == 1
    torch.testing.assert_close(actual, reference, atol=4e-3, rtol=5e-3)
    if batch_size == 32:
        sched_meta = next(iter(executor._graph_sched_meta.values()))
        assert sched_meta.tile_scheduler_metadata.size(0) < batch_size


def _prefill_inputs(window_size: int, compress_ratio: int = 0):
    start_pos = torch.tensor([0, 5, 12], dtype=torch.int64, device='cuda')
    q_seqlens = torch.tensor([3, 2, 4], dtype=torch.int64, device='cuda')
    total_lens = start_pos + q_seqlens
    prev_window = start_pos.clamp(max=window_size)
    uncompressed_kv_lens = (prev_window + q_seqlens).to(torch.int64)
    compressed_lens = (
        torch.div(total_lens, compress_ratio, rounding_mode='floor')
        if compress_ratio else torch.zeros_like(total_lens))
    flat_lens = (uncompressed_kv_lens + compressed_lens).to(torch.int32)
    cu_seqlens_k = torch.zeros(4, dtype=torch.int32, device='cuda')
    torch.cumsum(flat_lens, dim=0, out=cu_seqlens_k[1:])

    cu_q = torch.zeros(4, dtype=torch.int64, device='cuda')
    torch.cumsum(q_seqlens, dim=0, out=cu_q[1:])
    token_id = torch.arange(int(cu_q[-1].item()), dtype=torch.int64, device='cuda')
    token_seq = torch.searchsorted(cu_q[1:], token_id, right=True)
    token_pos = token_id - cu_q[token_seq]
    return start_pos, total_lens, token_seq, token_pos, cu_seqlens_k, uncompressed_kv_lens


def _prefill_ref(
    start_pos,
    total_lens,
    token_seq,
    token_pos,
    cu_seqlens_k,
    uncompressed_kv_lens,
    window_size,
    compress_ratio=0,
    compress_topk=None,
    compress_width=0,
    block=4,
):
    abs_pos = start_pos[token_seq] + token_pos
    num_vis = (abs_pos + 1).clamp(max=window_size)
    window_start_abs = (start_pos - window_size).clamp(min=0)
    first_vis_abs = (abs_pos - window_size + 1).clamp(min=0)
    first_flat_pos = first_vis_abs - window_start_abs[token_seq]

    window_col = torch.arange(window_size, dtype=torch.int64, device='cuda').unsqueeze(0)
    window_vals = torch.where(
        window_col < num_vis.unsqueeze(1),
        first_flat_pos.unsqueeze(1) + window_col,
        torch.full((), -1, dtype=torch.int64, device='cuda'))

    if compress_ratio:
        comp_col = torch.arange(compress_width, dtype=torch.int64, device='cuda').unsqueeze(0)
        if compress_topk is not None:
            comp_vals = torch.where(
                compress_topk.to(torch.int64) >= 0,
                compress_topk.to(torch.int64) + uncompressed_kv_lens[token_seq].unsqueeze(1),
                torch.full((), -1, dtype=torch.int64, device='cuda'))
        else:
            num_compressed = torch.div(total_lens, compress_ratio, rounding_mode='floor')
            comp_vals = torch.where(
                comp_col < num_compressed[token_seq].unsqueeze(1),
                comp_col + uncompressed_kv_lens[token_seq].unsqueeze(1),
                torch.full((), -1, dtype=torch.int64, device='cuda'))
        vals = torch.cat([window_vals, comp_vals], dim=-1)
        padded_width = ((vals.size(-1) + block - 1) // block) * block
        topk_length = (window_size + torch.div(abs_pos + 1, compress_ratio, rounding_mode='floor')).clamp(
            max=padded_width).to(torch.int32)
    else:
        vals = window_vals
        topk_length = torch.full((token_seq.numel(),), window_size, dtype=torch.int32, device='cuda')

    vals = _pad_ref(vals.unsqueeze(1), block=block).squeeze(1)
    vals = torch.where(
        vals >= 0,
        vals + cu_seqlens_k[token_seq].to(torch.int64).unsqueeze(1),
        torch.full((), -1, dtype=torch.int64, device='cuda'))
    return vals.unsqueeze(1).to(torch.int32), topk_length


@torch.inference_mode()
def test_build_prefill_sparse_indices_without_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_prefill_sparse_indices

    window_size = 5
    args = _prefill_inputs(window_size)
    out, topk_length = build_prefill_sparse_indices(*args, window_size=window_size, block=4)
    ref, ref_topk_length = _prefill_ref(*args, window_size=window_size, block=4)

    assert out.shape == (9, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())


@torch.inference_mode()
def test_build_prefill_sparse_indices_prefix_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_prefill_sparse_indices

    window_size = 5
    compress_ratio = 4
    args = _prefill_inputs(window_size, compress_ratio)
    compress_width = 4
    out, topk_length = build_prefill_sparse_indices(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_width=compress_width,
        block=4)
    ref, ref_topk_length = _prefill_ref(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_width=compress_width,
        block=4)

    assert out.shape == (9, 1, 12)
    torch.testing.assert_close(out.cpu(), ref.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())


@torch.inference_mode()
def test_build_prefill_sparse_indices_indexer_compress_matches_reference():
    from lmdeploy.pytorch.kernels.cuda.v4_sparse_indices import build_prefill_sparse_indices

    window_size = 5
    compress_ratio = 4
    args = _prefill_inputs(window_size, compress_ratio)
    compress_topk = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [1, 0, -1],
         [0, 1, 2], [2, -1, 1], [1, 3, -1],
         [0, 2, 3], [3, 2, 1], [1, -1, 0]],
        dtype=torch.int32,
        device='cuda')

    out, topk_length = build_prefill_sparse_indices(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_topk=compress_topk,
        block=4)
    ref, ref_topk_length = _prefill_ref(
        *args,
        window_size=window_size,
        compress_ratio=compress_ratio,
        compress_topk=compress_topk,
        compress_width=compress_topk.size(1),
        block=4)

    assert out.shape == (9, 1, 8)
    torch.testing.assert_close(out.cpu(), ref.cpu())
    torch.testing.assert_close(topk_length.cpu(), ref_topk_length.cpu())
