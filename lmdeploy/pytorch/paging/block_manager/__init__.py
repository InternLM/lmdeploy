# Copyright (c) OpenMMLab. All rights reserved.
from ...config import CacheConfig, SharedCacheArenaGeometry
from .base_block_manager import BaseBlockManager
from .default_block_manager import DefaultBlockManager
from .group_allocator import GroupAllocator, GroupRole
from .shared_block_manager import SharedBlockManager
from .window_block_manager import WindowBlockManager

__all__ = [
    'BaseBlockManager', 'DefaultBlockManager', 'GroupAllocator', 'GroupRole', 'SharedBlockManager',
    'WindowBlockManager', 'build_block_manager'
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
        geometry = SharedCacheArenaGeometry.from_cache_config(cache_config, require_capacity=True)
        cache_config.arena_num_protected_groups = geometry.num_protected_groups
        return SharedBlockManager(
            num_gpu_blocks,
            num_cpu_blocks,
            group_size=geometry.units_per_group,
            num_protected_groups=geometry.num_protected_groups,
            reserve_padding_group=True)

    if window_size is None or window_size < 0:
        return DefaultBlockManager(num_gpu_blocks, num_cpu_blocks, num_gpu_reserved=num_gpu_reserved)
    else:
        return WindowBlockManager(num_gpu_blocks,
                                  num_cpu_blocks,
                                  window_size=window_size,
                                  num_gpu_reserved=num_gpu_reserved)
