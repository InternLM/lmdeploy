# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.cache_engine import CacheEngine
from lmdeploy.pytorch.engine.executor.base import ExecutorBase, _CacheBlockSize


class _RecordingExecutor(ExecutorBase):

    def __init__(self, empty_init: bool):
        super().__init__(
            model_path='',
            model_config=SimpleNamespace(sliding_window=None, states_shapes=None),
            cache_config=SimpleNamespace(),
            backend_config=SimpleNamespace(),
            dist_config=SimpleNamespace(dp=1, world_size=1),
            misc_config=SimpleNamespace(empty_init=empty_init),
        )
        self.calls = []

    def build_model(self):
        self.calls.append('build_model')

    def update_configs(self):
        self.calls.append('update_configs')

    def build_graph_runner(self):
        self.calls.append('build_graph_runner')

    def build_cache_engine(self):
        self.calls.append('build_cache_engine')

    def warmup(self):
        self.calls.append('warmup')


def test_init_warms_up_model_by_default():
    executor = _RecordingExecutor(empty_init=False)

    executor.init()

    assert executor.calls == [
        'build_model',
        'update_configs',
        'build_graph_runner',
        'build_cache_engine',
        'warmup',
    ]


def test_init_skips_model_warmup_for_empty_init():
    executor = _RecordingExecutor(empty_init=True)

    executor.init()

    assert executor.calls == [
        'build_model',
        'update_configs',
        'build_graph_runner',
        'build_cache_engine',
    ]


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


def test_get_rank_cache_block_sizes_charges_all_ranks_when_spec_tp_matches_target():
    executor = object.__new__(ExecutorBase)
    executor.dist_config = SimpleNamespace(attn_tp=4)
    executor.specdecode_config = SimpleNamespace(dist_config=SimpleNamespace(attn_tp=4))

    cache_block_sizes = executor._get_rank_cache_block_sizes(4, _CacheBlockSize(target=256, spec=128))

    assert cache_block_sizes == [384, 384, 384, 384]


def test_get_cache_block_sizes_uses_spec_tp(monkeypatch):
    executor = object.__new__(ExecutorBase)
    executor.dist_config = SimpleNamespace(attn_tp=4)
    executor.cache_config = object()
    executor.model_config = object()
    executor.specdecode_config = SimpleNamespace(dist_config=SimpleNamespace(attn_tp=4))
    spec_cache_config = object()
    spec_model_config = object()
    world_sizes = []

    def fake_get_cache_block_size(cache_config, model_config, world_size):
        world_sizes.append(world_size)
        return 256 if model_config is executor.model_config else 128

    monkeypatch.setattr(CacheEngine, 'get_cache_block_size', fake_get_cache_block_size)

    cache_block_size = executor._get_cache_block_sizes(spec_cache_config, spec_model_config)

    assert cache_block_size == _CacheBlockSize(target=256, spec=128)
    assert world_sizes == [4, 4]


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
