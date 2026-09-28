# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


@pytest.mark.parametrize('mode', ['extend', 'decode'])
@pytest.mark.parametrize('round_scale', [False, True])
def test_kpool_compression_cpu_fallback(mode, round_scale, monkeypatch):
    from lmdeploy.pytorch.backends.cuda import kpool
    from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8

    def unexpected_cuda(*args, **kwargs):
        raise AssertionError('CPU compression must not invoke a CUDA kernel')

    monkeypatch.setattr(kpool, 'compress_kpool', unexpected_cuda)
    torch.manual_seed(74)
    keys = torch.randn(3, 4, 128, dtype=torch.bfloat16)
    scores = torch.randn(3, 4, 128)
    ape = torch.randn(4, 128)
    expected = kpool_quantize_fp8(kpool_compress(keys, scores, ape, mode=mode),
                                block_size=128, round_scale=round_scale)
    actual = kpool.kpool_compress_quantize_cuda(keys, scores, ape, mode=mode, round_scale=round_scale)
    torch.testing.assert_close(actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def reference_update(keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, round_scale):
    from lmdeploy.pytorch.nn.kpool import (
        kpool_compress,
        kpool_partition_update,
        kpool_quantize_fp8,
        kpool_write_packed_cache,
    )

    start = 0
    for request, (q_len, kv_len, state_id) in enumerate(zip(q_lens.tolist(), kv_lens.tolist(), ids.tolist())):
        if state_id >= 0:
            history = kv_len - q_len
            old_tail = history % 4
            update = kpool_partition_update(keys[start:start + q_len], scores[start:start + q_len], history, 4,
                                            states[0][state_id, :old_tail], states[1][state_id, :old_tail])
            if update.closed_group_ids.numel():
                pooled = kpool_compress(update.closed_keys, update.closed_scores, ape, mode='extend')
                values, scales = kpool_quantize_fp8(pooled, block_size=pooled.size(-1), round_scale=round_scale)
                kpool_write_packed_cache(cache, blocks[request], update.closed_group_ids, values, scales, 4)
            for state, tail in zip(states, (update.tail_keys, update.tail_scores)):
                # Clone because the empty-query reference can alias the source tail.
                tail = tail.clone()
                state[state_id].zero_()
                state[state_id, :tail.size(0)].copy_(tail)
        start += q_len


def make_case(lengths, histories, round_scale, page_size=64):
    torch.manual_seed(17)
    batch = len(lengths)
    keys = torch.randn(sum(lengths), 128, device='cuda', dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    q_lens = torch.tensor(lengths, device='cuda', dtype=torch.int32)
    kv_lens = q_lens + torch.tensor(histories, device='cuda', dtype=torch.int32)
    ids = torch.arange(batch, 0, -1, device='cuda')
    states = tuple(torch.randn(batch + 2, 4, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    blocks = torch.arange(1, batch * 160 + 1, device='cuda').reshape(batch, 160)
    cache = torch.randint(0, 256, (batch * 160 + 1, page_size, 1, 132), device='cuda', dtype=torch.uint8)
    ape = torch.randn(4, 128, device='cuda')
    return keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, round_scale


def candidate_update(keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, round_scale):
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_prefill_update_cuda

    kpool_prefill_update_cuda(keys, scores, *states, ids, q_lens, kv_lens, cache, blocks, ape, 4, round_scale)


@pytest.mark.parametrize('lengths,histories', [
    ([0], [3]), ([0, 0], [0, 3]), ([1, 2], [0, 0]),
    ([0, 1, 3, 4, 5], [0, 3, 1, 4, 7]), ([511, 513], [3, 256]), ([8192], [0]),
])
@pytest.mark.parametrize('round_scale', [False, True])
@pytest.mark.parametrize('metadata_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('page_size', [16, 64])
def test_kpool_ragged_prefill_matches_reference(lengths, histories, round_scale, metadata_dtype, page_size):
    args = make_case(lengths, histories, round_scale, page_size)
    keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, _ = args
    q_lens, kv_lens = q_lens.to(metadata_dtype), kv_lens.to(metadata_dtype)
    initial_states = tuple(state.clone() for state in states)
    expected_states = tuple(state.clone() for state in states)
    expected_cache = cache.clone()
    reference_update(keys, scores, expected_states, ids, q_lens, kv_lens, expected_cache, blocks, ape, round_scale)
    candidate_update(keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, round_scale)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    for actual, initial in zip(states, initial_states):
        torch.testing.assert_close(actual, initial, rtol=0, atol=0)


@pytest.mark.parametrize('page_size', [16, 64])
def test_kpool_ragged_prefill_graph_changes_layout_and_padding(page_size):
    args = make_case([3, 5, 0], [0, 1, 2], True, page_size)
    keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, _ = args
    ids[-1] = -1
    initial_states = tuple(state.clone() for state in states)
    initial_cache = cache.clone()
    candidate_update(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate_update(*args)
    for lengths, histories, reordered in [([1, 0, 7], [5, 4, 8], [1, 3, 2]),
                                         ([4, 3, 1], [7, 1, 9], [-1, 2, 1])]:
        q_lens.copy_(torch.tensor(lengths, device='cuda'))
        kv_lens.copy_(q_lens + torch.tensor(histories, device='cuda'))
        ids.copy_(torch.tensor(reordered, device='cuda'))
        keys.normal_()
        scores.normal_()
        expected_states = tuple(state.clone() for state in initial_states)
        expected_cache = initial_cache.clone()
        reference_update(keys, scores, expected_states, ids, q_lens, kv_lens, expected_cache, blocks, ape, True)
        for state, initial in zip(states, initial_states):
            state.copy_(initial)
        cache.copy_(initial_cache)
        graph.replay()
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        for actual, initial in zip(states, initial_states):
            torch.testing.assert_close(actual, initial, rtol=0, atol=0)


def test_kpool_prefill_layer_views_and_chunk_continuation():
    keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, _ = make_case([3, 5], [6, 7], True)
    # Runtime states are layer views of [request, layer, pool, width] storage.
    banks = tuple(torch.randn(state.size(0), 11, 4, 128, device='cuda', dtype=state.dtype) for state in states)
    expected_banks = tuple(bank.clone() for bank in banks)
    states = tuple(bank[:, 3] for bank in banks)
    expected_states = tuple(bank[:, 3] for bank in expected_banks)
    expected_cache = cache.clone()
    scores = scores.float()
    for lengths in ([3, 5], [1, 7], [6, 2]):
        history = kv_lens - q_lens
        q_lens.copy_(torch.tensor(lengths, device='cuda'))
        kv_lens.copy_(history + q_lens)
        keys.normal_()
        scores.normal_()
        reference_update(keys, scores, expected_states, ids, q_lens, kv_lens, expected_cache, blocks, ape, True)
        before = tuple(bank.clone() for bank in banks)
        candidate_update(keys, scores, states, ids, q_lens, kv_lens, cache, blocks, ape, True)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
        for bank, initial in zip(banks, before):
            torch.testing.assert_close(bank, initial, rtol=0, atol=0)
        # The pool assembler consumes scratch; the caller reconstructs the
        # next tail from its raw owner. Use the independent oracle's tail.
        for state, expected in zip(states, expected_states):
            state.copy_(expected)
        kv_lens.add_(q_lens)


def test_indexed_key_scatter_step_major_strides_and_mask():
    from lmdeploy.pytorch.kernels.cuda.fill_kv_cache import fill_indexed_key_cache
    from lmdeploy.pytorch.nn.kpool import kpool_packed_cache_views

    torch.manual_seed(59)
    cache = torch.randint(0, 256, (17, 64, 1, 132), device='cuda', dtype=torch.uint8)
    expected = cache.clone()
    keys, scales = kpool_packed_cache_views(cache, 128)
    ref_keys, ref_scales = kpool_packed_cache_views(expected, 128)
    values = torch.randn(128, 6, device='cuda').to(torch.float8_e4m3fn).T
    value_scales = torch.rand(12, device='cuda')[::2]
    blocks = torch.arange(1, 17, device='cuda').reshape(2, 8)
    groups = torch.tensor([0, -1, 1, 2, -1, 3], device='cuda')
    valid = groups >= 0
    for row in [0, 2, 3, 5]:
        ref_keys[blocks[row % 2, 0], groups[row]] = values[row]
        ref_scales[blocks[row % 2, 0], groups[row], 0] = value_scales[row]
    fill_indexed_key_cache(values, value_scales, groups, valid, blocks, keys, scales, page_step=4)
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
                    reason='Native FP8 compression requires Hopper or newer.')
@pytest.mark.parametrize('width,pool', [(64, 2), (128, 4)])
@pytest.mark.parametrize('mode', ['extend', 'decode'])
@pytest.mark.parametrize('round_scale', [False, True])
def test_kpool_compression_preserves_fp8_bytes_and_scales(width, pool, mode, round_scale):
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_compress_quantize_cuda
    from lmdeploy.pytorch.kernels.cuda.kpool import compress_kpool
    from lmdeploy.pytorch.nn.kpool import kpool_compress, kpool_quantize_fp8

    torch.manual_seed(73)
    keys = torch.randn(257, pool, width, device='cuda', dtype=torch.bfloat16)
    scores = torch.randn_like(keys, dtype=torch.float32 if mode == 'decode' else torch.bfloat16)
    ape = torch.randn(pool, width, device='cuda')
    keys[0].zero_()
    keys[1].mul_(1e-6)
    keys[2].mul_(1000)
    scores[3, 0].fill_(90)
    scores[3, 1:].fill_(-90)
    expected = kpool_quantize_fp8(kpool_compress(keys, scores, ape, mode=mode),
                                 block_size=width, round_scale=round_scale)
    actual = compress_kpool(keys, scores, ape, mode=mode, round_scale=round_scale)
    torch.testing.assert_close(actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    dispatched = kpool_compress_quantize_cuda(keys, scores, ape, mode=mode, round_scale=round_scale)
    torch.testing.assert_close(dispatched[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(dispatched[1], expected[1], rtol=0, atol=0)


@pytest.mark.parametrize('page_size', [16, 64])
def test_decode_masked_writer_matches_full_arena_reference(page_size):
    from lmdeploy.pytorch.backends.cuda.kpool import kpool_write_decode_cuda
    from lmdeploy.pytorch.nn.kpool import kpool_write_packed_cache_batched

    torch.manual_seed(333)
    # Layer/step-major views, inactive rows with duplicate and negative indices.
    arena = torch.randint(0, 256, (40, 3, page_size, 1, 132), device='cuda', dtype=torch.uint8)
    expected = arena.clone()
    table = torch.arange(1, 33, device='cuda').reshape(4, 8)
    values = torch.randn(4, 2, 128, device='cuda').to(torch.float8_e4m3fn)[:, 1]
    scales = torch.rand(4, 2, device='cuda')[:, 1]
    groups = torch.tensor([15, 16, -1, 500], device='cuda')
    valid = torch.tensor([True, True, False, False], device='cuda')
    kpool_write_packed_cache_batched(expected[:, 1], table, groups, values, scales, 4, valid)
    kpool_write_decode_cuda(arena[:, 1], table, groups, values, scales, 4, valid)
    assert torch.equal(arena, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kpool_write_decode_cuda(arena[:, 1], table, groups, values, scales, 4, valid)
    groups.copy_(torch.tensor([16, 63, 64, 1], device='cuda'))
    valid.copy_(torch.tensor([True, False, True, False], device='cuda'))
    kpool_write_packed_cache_batched(expected[:, 1], table, groups, values, scales, 4, valid)
    graph.replay()
    assert torch.equal(arena, expected)
