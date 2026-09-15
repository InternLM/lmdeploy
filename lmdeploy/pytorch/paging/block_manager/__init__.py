# Copyright (c) OpenMMLab. All rights reserved.
from ...config import CacheConfig
from .base_block_manager import BaseBlockManager
from .default_block_manager import DefaultBlockManager
from .group_allocator import GroupAllocator
from .shared_block_manager import SharedBlockManager
from .window_block_manager import WindowBlockManager

__all__ = [
    'BaseBlockManager', 'DefaultBlockManager', 'GroupAllocator', 'SharedBlockManager', 'WindowBlockManager',
    'build_block_manager'
]


def build_block_manager(cache_config: CacheConfig) -> BaseBlockManager:
    """Build block manager.

    Args:
        cache_config (CacheConfig):  cache_config.
    """

    num_cpu_blocks = cache_config.num_cpu_blocks
    num_gpu_blocks = cache_config.num_gpu_blocks
    window_size = cache_config.window_size
    num_gpu_reserved = cache_config.num_reserved_gpu_blocks

    if cache_config.enable_kv_state_cache_sharing:
        if num_cpu_blocks != 0:
            raise ValueError('Shared KV/state allocation requires num_cpu_blocks=0.')
        if window_size is not None and window_size >= 0:
            raise ValueError('Shared KV/state allocation does not support sliding-window blocks.')
        if num_gpu_reserved != 0:
            raise ValueError('Shared KV/state allocation uses a complete padding group; '
                             'num_reserved_gpu_blocks must be zero.')
        num_protected_groups = cache_config.arena_num_protected_groups
        if cache_config.states_shapes or cache_config.num_state_caches is not None:
            num_state_caches = cache_config.num_state_caches or 1
            num_runtime_states = max(0, num_state_caches - 1 - cache_config.prefix_cache_state_budget)
            num_runtime_states = min(num_runtime_states, max(0, num_state_caches - 1))
            num_protected_groups = max(num_protected_groups, 1 + num_runtime_states)
        cache_config.arena_num_protected_groups = num_protected_groups
        return SharedBlockManager(
            num_gpu_blocks,
            num_cpu_blocks,
            group_size=cache_config.arena_units_per_group,
            num_protected_groups=num_protected_groups,
            reserve_padding_group=True)

    if window_size < 0:
        return DefaultBlockManager(num_gpu_blocks, num_cpu_blocks, num_gpu_reserved=num_gpu_reserved)
    else:
        return WindowBlockManager(num_gpu_blocks,
                                  num_cpu_blocks,
                                  window_size=window_size,
                                  num_gpu_reserved=num_gpu_reserved)
