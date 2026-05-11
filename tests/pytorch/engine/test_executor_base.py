# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.executor.base import ExecutorBase, _CacheBlockSize


def test_get_num_gpu_blocks_without_spec_cache():
    available_mem = 4096
    cache_block_size = 256

    num_gpu_blocks = ExecutorBase._get_num_gpu_blocks(available_mem, cache_block_size)

    assert num_gpu_blocks == 16


def test_get_num_gpu_blocks_with_spec_cache():
    available_mem = 4096
    cache_block_size = 256
    spec_cache_block_size = 256

    num_gpu_blocks = ExecutorBase._get_num_gpu_blocks(available_mem, cache_block_size, spec_cache_block_size)

    assert num_gpu_blocks == 8


def test_get_num_gpu_blocks_rejects_empty_cache_block():
    with pytest.raises(RuntimeError, match='No enough gpu memory for kv cache.'):
        ExecutorBase._get_num_gpu_blocks(available_mem=4096, cache_block_size=0)


def test_sync_spec_cache_block_size_updates_kernel_block_size():
    executor = object.__new__(ExecutorBase)
    executor.cache_config = CacheConfig(max_batches=1,
                                        block_size=32,
                                        kernel_block_size=16,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0)
    spec_cache_config = CacheConfig(max_batches=1,
                                    block_size=64,
                                    kernel_block_size=64,
                                    num_cpu_blocks=0,
                                    num_gpu_blocks=0)
    executor.specdecode_config = SimpleNamespace(cache_config=spec_cache_config)

    executor._sync_spec_cache_block_size()

    assert spec_cache_config.block_size == 32
    assert spec_cache_config.kernel_block_size == 16


def test_get_rank_cache_block_sizes_only_charges_spec_rank():
    executor = object.__new__(ExecutorBase)
    executor.dist_config = SimpleNamespace(attn_tp=2)

    cache_block_sizes = executor._get_rank_cache_block_sizes(4, _CacheBlockSize(target=256, spec=128))

    assert cache_block_sizes == [384, 256, 384, 256]


def test_update_num_gpu_blocks_can_be_limited_by_non_spec_rank():
    executor = object.__new__(ExecutorBase)
    executor.dist_config = SimpleNamespace(attn_tp=2)
    executor.cache_config = CacheConfig(max_batches=1,
                                        block_size=64,
                                        num_cpu_blocks=0,
                                        num_gpu_blocks=0,
                                        cache_max_entry_count=1.0)
    spec_cache_config = CacheConfig(max_batches=1, block_size=64, num_cpu_blocks=0, num_gpu_blocks=0)

    executor._update_num_gpu_blocks([2048, 768], _CacheBlockSize(target=256, spec=256), spec_cache_config)

    assert executor.cache_config.num_gpu_blocks == 3
    assert spec_cache_config.num_gpu_blocks == 3
