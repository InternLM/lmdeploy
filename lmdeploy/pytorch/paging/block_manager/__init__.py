# Copyright (c) OpenMMLab. All rights reserved.
from ...config import CacheConfig
from .base_block_manager import BaseBlockManager
from .default_block_manager import DefaultBlockManager
from .window_block_manager import WindowBlockManager


def build_block_manager(cache_config: CacheConfig) -> BaseBlockManager:
    """Build block manager.

    Args:
        cache_config (CacheConfig):  cache_config.
    """

    num_cpu_blocks = cache_config.num_cpu_blocks
    num_gpu_blocks = cache_config.num_gpu_blocks
    window_size = cache_config.window_size
    num_gpu_reserved = cache_config.num_reserved_gpu_blocks

    if window_size < 0:
        return DefaultBlockManager(num_gpu_blocks, num_cpu_blocks, num_gpu_reserved=num_gpu_reserved)
    else:
        return WindowBlockManager(num_gpu_blocks,
                                  num_cpu_blocks,
                                  window_size=window_size,
                                  num_gpu_reserved=num_gpu_reserved)
