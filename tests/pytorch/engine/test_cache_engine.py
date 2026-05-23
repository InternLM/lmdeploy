# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import pytest
import torch

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine


def test_allocate_caches_requires_block_size_divisible_by_kernel_block_size():
    cache_config = CacheConfig(max_batches=1,
                               block_size=96,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    with pytest.raises(ValueError, match='block_size 96 must be divisible by kernel_block_size 64'):
        CacheEngine.allocate_caches(num_blocks=1,
                                    model_config=None,
                                    cache_config=cache_config,
                                    world_size=1,
                                    device='meta')


def test_pd_migration_rejects_split_kernel_blocks():
    cache_engine = object.__new__(CacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=96,
                                            kernel_block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0)
    migration_inputs = MigrationExecutionBatch(protocol=MigrationProtocol.RDMA, requests=[])

    with pytest.raises(RuntimeError, match='PD migration does not support block_size != kernel_block_size'):
        asyncio.run(cache_engine.migrate(migration_inputs))


def _make_state_cache_engine(num_caches: int = 4):
    cache_engine = object.__new__(StateCacheEngine)
    cache_engine.cache_config = CacheConfig(max_batches=1,
                                            block_size=64,
                                            num_cpu_blocks=0,
                                            num_gpu_blocks=0,
                                            num_state_caches=num_caches,
                                            states_shapes=[((2, 3), torch.float32), ((2, ), torch.float16)])
    cache_engine.mem_pool, cache_engine._state_caches = StateCacheEngine.allocate_caches(
        num_caches=num_caches,
        state_shapes=cache_engine.cache_config.states_shapes,
        device='cpu',
    )
    return cache_engine


def test_state_cache_engine_copy_caches_copies_all_state_views():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(5.0)

    cache_engine.copy_caches(1, 2)

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_caches_supports_batched_indices():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[0].fill_(1.0)
    recurrent_state[0].fill_(2.0)
    conv_state[1].fill_(3.0)
    recurrent_state[1].fill_(4.0)

    cache_engine.copy_caches(torch.tensor([0, 1]), torch.tensor([2, 3]))

    assert torch.equal(conv_state[2], conv_state[0])
    assert torch.equal(recurrent_state[2], recurrent_state[0])
    assert torch.equal(conv_state[3], conv_state[1])
    assert torch.equal(recurrent_state[3], recurrent_state[1])


def test_state_cache_engine_copy_caches_rejects_mismatched_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='same number of elements'):
        cache_engine.copy_caches([0, 1], [2])
