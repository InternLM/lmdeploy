# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

from ...config import CacheConfig
from .base_block_manager import BaseBlockManager
from .default_block_manager import DefaultBlockManager
from .window_block_manager import WindowBlockManager


def build_block_manager(cache_config: CacheConfig,
                        adapter_manager: Any = None) -> BaseBlockManager:
    """build block manager.

    Args:
        cache_config (CacheConfig):  cache_config.
    """

    num_cpu_blocks = cache_config.num_cpu_blocks
    num_gpu_blocks = cache_config.num_gpu_blocks
    window_size = cache_config.window_size

    if window_size < 0:
        return DefaultBlockManager(num_gpu_blocks, num_cpu_blocks,
                                   adapter_manager)
    else:
        return WindowBlockManager(num_gpu_blocks,
                                  num_cpu_blocks,
                                  window_size=window_size,
                                  adapter_manager=adapter_manager)
