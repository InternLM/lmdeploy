# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def test_r3_scatter_partial_chunks_speculative_overwrite_and_block_copy():
    from lmdeploy.pytorch.kernels.cuda.copy_cache import copy_cache_blocks
    from lmdeploy.pytorch.kernels.cuda.routed_experts import scatter_routed_experts

    # Two kernel pages per logical block; rows cross both page and block edges.
    cache = torch.zeros((4, 8, 2, 3), dtype=torch.uint16, device='cuda')
    blocks = torch.tensor([[4, 5, 0, 1], [6, 7, 2, 3]], device='cuda')
    starts = torch.tensor([0, 6], device='cuda')
    lengths = torch.tensor([6, 3], device='cuda')
    history = torch.tensor([3, 7], device='cuda')
    routes = torch.arange(9 * 6, device='cuda').to(torch.uint16).view(9, 2, 3)
    scatter_routed_experts(routes, cache, starts, lengths, history, blocks, 6, 4)
    expected = torch.zeros_like(cache, device='cpu')
    for row, (start, count, past) in enumerate(((0, 6, 3), (6, 3, 7))):
        for i in range(count):
            pos = past + i
            page = blocks[row, pos // 4].item()
            expected.view(-1, 2, 3)[page * 4 + pos % 4] = routes[start + i].cpu()
    # Reject the last two speculative rows of sequence 1 and overwrite them.
    replacement = torch.full((2, 2, 3), 99, dtype=torch.uint16, device='cuda')
    scatter_routed_experts(replacement, cache, starts[:1], torch.tensor([2], device='cuda'),
                          torch.tensor([8], device='cuda'), blocks[1:], 2, 4)
    expected[1, :2] = 99
    copy_cache_blocks(cache.view(1, 8, 4, 2, 3), 1, torch.tensor([2], device='cuda'),
                      torch.tensor([0], device='cuda'), 2)
    expected[0].copy_(expected[2])
    torch.testing.assert_close(cache.cpu(), expected)

    # A restored API history must outlive reuse of the registered GPU blocks.
    from lmdeploy.pytorch.kv_connector.mooncake.store.data import MooncakeStoreLoadRequest
    from lmdeploy.pytorch.kv_connector.mooncake.store.worker_threads import KVCacheStoreRecvingThread

    receiver = KVCacheStoreRecvingThread.__new__(KVCacheStoreRecvingThread)
    receiver.routed_experts_cache = cache
    history = receiver._copy_routes(MooncakeStoreLoadRequest(7, (3, 1), (bytes(32), bytes(32))))
    cache.zero_()
    torch.testing.assert_close(torch.from_numpy(history), torch.cat((expected[3], expected[1])))
