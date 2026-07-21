# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.config import CacheConfig, ModelConfig
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.messages import MigrationExecutionBatch
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine


def test_bf16_sparse_mla_cache_layout():
    model_config = ModelConfig(hidden_size=6144,
                               num_layers=2,
                               num_attention_heads=64,
                               num_key_value_heads=1,
                               bos_token_id=None,
                               eos_token_id=[1],
                               head_dim=576,
                               k_head_dim=576,
                               v_head_dim=0,
                               dtype=torch.bfloat16,
                               use_flash_mla=True,
                               mla_kv_cache_dtype='bfloat16',
                               mla_index_topk=2048)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0)

    k_desc = CacheEngine.get_k_cache_desc(model_config, cache_config)
    v_desc = CacheEngine.get_v_cache_desc(model_config, cache_config)

    assert k_desc.shape == [64, 1, 576]
    assert k_desc.dtype == torch.bfloat16
    assert v_desc.shape == [64, 1, 0]
    assert v_desc.dtype == torch.bfloat16


def test_fp8_sparse_mla_cache_layout():
    model_config = ModelConfig(hidden_size=6144,
                               num_layers=2,
                               num_attention_heads=64,
                               num_key_value_heads=1,
                               bos_token_id=None,
                               eos_token_id=[1],
                               head_dim=576,
                               k_head_dim=576,
                               v_head_dim=0,
                               dtype=torch.bfloat16,
                               use_flash_mla=True,
                               mla_kv_cache_dtype='bfloat16',
                               mla_index_topk=2048)
    cache_config = CacheConfig(max_batches=1,
                               block_size=64,
                               kernel_block_size=64,
                               num_cpu_blocks=0,
                               num_gpu_blocks=0,
                               quant_policy=QuantPolicy.FP8)
    CacheEngine.get_cache_block_size(cache_config, model_config)

    assert model_config.mla_kv_cache_dtype == 'fp8_ds_mla'

    k_desc = CacheEngine.get_k_cache_desc(model_config, cache_config)
    v_desc = CacheEngine.get_v_cache_desc(model_config, cache_config)

    assert k_desc.shape == [64, 1, 656]
    assert k_desc.dtype == torch.float8_e4m3fn
    assert v_desc.shape == [64, 1, 0]
    assert v_desc.dtype == torch.float8_e4m3fn


@pytest.mark.parametrize('quant_policy',
                         [QuantPolicy.INT4, QuantPolicy.INT8, QuantPolicy.FP8_E5M2, QuantPolicy.TURBO_QUANT])
def test_sparse_mla_rejects_other_cache_policies(quant_policy):
    model_config = SimpleNamespace(mla_index_topk=2048)
    cache_config = SimpleNamespace(quant_policy=quant_policy)

    with pytest.raises(ValueError, match='Sparse MLA does not support quant_policy'):
        CacheEngine.get_cache_block_size(cache_config, model_config)


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

    cache_engine.copy_caches((1, 0), (3, 2))

    assert torch.equal(conv_state[2], conv_state[0])
    assert torch.equal(recurrent_state[2], recurrent_state[0])
    assert torch.equal(conv_state[3], conv_state[1])
    assert torch.equal(recurrent_state[3], recurrent_state[1])


def test_state_cache_engine_copy_caches_accepts_host_integer_scalars():
    cache_engine = _make_state_cache_engine()
    conv_state, recurrent_state = cache_engine.state_caches

    conv_state[1].fill_(7.0)
    recurrent_state[1].fill_(9.0)

    cache_engine.copy_caches(np.int64(1), np.int64(2))

    assert torch.equal(conv_state[2], conv_state[1])
    assert torch.equal(recurrent_state[2], recurrent_state[1])


def test_state_cache_engine_copy_caches_coalesces_contiguous_ranges():
    ranges = list(StateCacheEngine._copy_ranges([4, 1, 5, 0, 6, 9], [20, 11, 21, 10, 22, 30]))

    assert ranges == [(0, 10, 2), (4, 20, 3), (9, 30, 1)]
    assert list(StateCacheEngine._copy_ranges([], [])) == []


def test_state_cache_engine_copy_caches_rejects_mismatched_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='same number of elements'):
        cache_engine.copy_caches([0, 1], [2])


def test_state_cache_engine_copy_caches_rejects_tensor_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_caches(torch.tensor([0, 1]), torch.tensor([2, 3]))


def test_state_cache_engine_copy_caches_rejects_tensor_sequence_items():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(TypeError, match='host integers'):
        cache_engine.copy_caches([torch.tensor(0)], [2])


def test_state_cache_engine_copy_caches_rejects_out_of_range_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_caches([-1], [2])

    with pytest.raises(ValueError, match='out of range'):
        cache_engine.copy_caches([0], [4])


def test_state_cache_engine_copy_caches_rejects_overlapping_indices():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='must not overlap'):
        cache_engine.copy_caches([0, 1], [1, 2])


def test_state_cache_engine_copy_caches_rejects_duplicate_destinations():
    cache_engine = _make_state_cache_engine()

    with pytest.raises(ValueError, match='duplicate'):
        cache_engine.copy_caches([0, 1], [2, 2])
