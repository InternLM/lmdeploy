# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from lmdeploy.pytorch.kernels.cuda.copy_cache import copy_cache_blocks, decode_save_mask

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize(
    ('shape', 'entry_axis', 'dtype', 'pages_per_block'),
    [
        ((2, 12, 5), 1, torch.uint8, 2),
        ((18, 3, 7), 0, torch.float16, 3),
        ((2, 3, 12, 4), 2, torch.float32, 2),
    ],
)
def test_copy_cache_blocks_handles_contiguous_pool_axes(shape, entry_axis, dtype, pages_per_block):
    num_logical_blocks = shape[entry_axis] // pages_per_block
    cache = torch.full(shape, 99, dtype=dtype, device='cuda')
    logical_cache = cache.unflatten(entry_axis, (num_logical_blocks, pages_per_block))
    src_blocks = torch.tensor([0, 2, 0], dtype=torch.long, device='cuda')
    dst_blocks = torch.tensor([3, 1, 4], dtype=torch.long, device='cuda')
    for block_id in set(src_blocks.tolist()):
        logical_cache.select(entry_axis, block_id).fill_(block_id + 1)
    expected = logical_cache.index_select(entry_axis, src_blocks).clone()

    copy_cache_blocks(cache, entry_axis, src_blocks, dst_blocks, pages_per_block)

    assert torch.equal(logical_cache.index_select(entry_axis, dst_blocks), expected)


@pytest.mark.parametrize('entry_axis', [0, 1])
def test_masked_copy_freezes_only_selected_state_slots(entry_axis):
    shape = (6, 2, 2051) if entry_axis == 0 else (2, 6, 2051)
    cache = torch.arange(6 * 2 * 2051, dtype=torch.float32, device='cuda').reshape(shape)
    plan = torch.tensor([[0, 0, 1], [64, 64, 64], [1, 1, 2], [3, 4, 5]], device='cuda')
    mask = torch.tensor([True, True, False], device='cuda')
    expected = cache.clone()
    expected.select(entry_axis, 3).copy_(cache.select(entry_axis, 1))
    expected.select(entry_axis, 4).copy_(cache.select(entry_axis, 1))
    copy_cache_blocks(cache, entry_axis, plan[2], plan[3], 1, mask)
    torch.testing.assert_close(cache, expected)


@pytest.mark.parametrize('is_decoding', [False, True])
def test_decode_save_mask_checks_retention_acceptance_and_stop(is_decoding):
    # Rejection before/at boundary, stop before/at boundary, stale candidate,
    # query ending before boundary, and accepted prompt-end output.
    history = torch.tensor([62, 62, 62, 62, 64, 60, 0], device='cuda')
    lengths = torch.tensor([4, 4, 4, 4, 4, 3, 64], device='cuda')
    tokens = torch.tensor([[1, -1, -1, -1], [1, 2, -1, -1], [1, 2, 3, 4],
                           [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, -1, -1, -1]], device='cuda')
    stops = torch.tensor([-1, -1, 0, 1, -1, -1, -1], device='cuda')
    rows = torch.arange(7, device='cuda')
    plan = torch.stack((rows, torch.full_like(rows, 64), rows, rows + 7))
    mask = decode_save_mask(plan, history, lengths, tokens, stops, is_decoding=is_decoding, lookahead=1)
    expected = [False, True, False, True, False, False, False] if is_decoding else [False] * 6 + [True]
    assert mask.tolist() == expected


@pytest.mark.parametrize('graph', [False, True])
@pytest.mark.parametrize('lookahead', [0, 1])
def test_recomputed_prefill_snapshot_copies_only_the_final_state(graph, lookahead):
    history = torch.tensor([128, 128, 0], device='cuda')
    lengths = torch.tensor([142, 192, 256], device='cuda')
    tokens = torch.tensor([[7], [8], [9]], device='cuda')
    stops = torch.full((3,), -1, device='cuda')
    # Replaying a 256-token prompt through token 270 has only the 270 state.
    # A replay ending at 320 and an initial 256-token prefill remain saveable.
    plan = torch.tensor([[0, 0, 1, 2], [256, 270, 320, 256], [1, 1, 2, 3], [4, 5, 6, 7]], device='cuda')
    cache = torch.full((8, 32), -1., device='cuda')
    cache[1].fill_(270)
    cache[2].fill_(320)
    cache[3].fill_(256)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def save():
        mask = decode_save_mask(plan, history, lengths, tokens, stops, is_decoding=False, lookahead=lookahead)
        copy_cache_blocks(cache, 0, plan[2], plan[3], 1, mask)
        return mask

    with torch.cuda.stream(stream):
        mask = save()
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                mask = save()
            cache[4:].fill_(-1)
            captured.replay()
        cache[1:4].fill_(999)
    torch.cuda.current_stream().wait_stream(stream)
    assert mask.tolist() == [False, True, True, True]
    expected = torch.tensor([-1., 270., 320., 256.], device='cuda')[:, None].expand_as(cache[4:])
    torch.testing.assert_close(cache[4:], expected, rtol=0, atol=0)
