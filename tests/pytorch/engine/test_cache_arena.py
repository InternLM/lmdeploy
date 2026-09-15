# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.default.cache import TorchBlockCacheCopy
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.cache_engine import SharedCacheArena, StateCacheEngine
from lmdeploy.pytorch.engine.cache_engine.layout import PackedBlockCacheLayout
from lmdeploy.pytorch.engine.cache_engine.plan import BlockCachePlan
from lmdeploy.pytorch.engine.cache_engine.schema import CacheDesc, CacheTensorSpec


def _plan():
    specs = (
        CacheTensorSpec('k_cache', CacheDesc([4], torch.float32)),
        CacheTensorSpec('v_cache', CacheDesc([4], torch.float32)),
    )
    return BlockCachePlan(
        tensor_specs=specs,
        layout=PackedBlockCacheLayout(specs, num_layers=2),
        kernel_blocks_per_logical_block=1,
    )


def test_shared_arena_projects_overlapping_kv_and_state_views():
    plan = _plan()
    cache_config = CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=4,
        num_state_caches=2,
        states_shapes=[((2, ), torch.float32)],
        enable_kv_state_cache_sharing=True,
        arena_units_per_group=2,
        arena_num_protected_groups=2,
    )
    model_config = SimpleNamespace(state_cache_specs=None)
    arena = SharedCacheArena.allocate(cache_config, model_config, plan, device='cpu')

    kv = arena.project_kv(plan)
    state = arena.project_state(cache_config, model_config)

    assert arena.num_groups == 5  # two flexible, two protected, one padding
    assert kv.owns_storage is False
    assert state.owns_storage is False
    assert kv.pools[0].tensor.shape == (2, arena.address_span, 512)
    assert state.pools[0].tensor.shape == (arena.num_groups, 256)

    kv.tensor_views[0][0, 0].fill_(7)
    # A state slot reuses the same group-strided bytes as its physical group.
    assert torch.all(state.tensor_views[0][0] == 7)

    state_engine = StateCacheEngine(cache_config, model_config, allocation=state)
    assert state_engine.num_slots == arena.num_groups
    state_engine.copy_slots(0, 1)
    assert torch.all(state_engine.state_caches[0][1] == 7)

    copy = TorchBlockCacheCopy.build(kv, arena.address_span, pages_per_block=1)
    kv.tensor_views[1][0, 0].fill_(11)
    copy.copy(torch.tensor([0]), torch.tensor([3]))
    assert torch.all(kv.tensor_views[1][0, 3] == 11)


def test_shared_arena_rejects_kernel_page_splitting():
    plan = BlockCachePlan(
        tensor_specs=(CacheTensorSpec('k_cache', CacheDesc([4], torch.float32)), ),
        layout=PackedBlockCacheLayout((CacheTensorSpec('k_cache', CacheDesc([4], torch.float32)), ), num_layers=1),
        kernel_blocks_per_logical_block=2,
    )
    cache_config = CacheConfig(
        max_batches=1,
        block_size=64,
        num_cpu_blocks=0,
        num_gpu_blocks=2,
        enable_kv_state_cache_sharing=True,
    )
    model_config = SimpleNamespace(state_cache_specs=None)

    try:
        SharedCacheArena.allocate(cache_config, model_config, plan, device='cpu')
    except ValueError as exc:
        assert 'kernel_block_size == block_size' in str(exc)
    else:
        raise AssertionError('kernel-page splitting must be rejected')
