# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner
from lmdeploy.pytorch.backends.graph_runner import GraphRunnerMeta
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


def _cache_config(max_batches=8, cudagraph_capture_batch_sizes=None):
    return CacheConfig(max_batches=max_batches,
                       block_size=64,
                       num_cpu_blocks=0,
                       num_gpu_blocks=1,
                       cudagraph_capture_batch_sizes=cudagraph_capture_batch_sizes)


def test_custom_capture_batch_sizes_include_max_batch_size():
    engine_config = PytorchEngineConfig(max_batch_size=8, cudagraph_capture_batch_sizes=[4, 1, 4, 16])

    engine_config = ConfigBuilder.update_engine_config(engine_config)

    assert engine_config.cudagraph_capture_batch_sizes == [1, 4, 8]


def test_cache_config_normalizes_capture_batch_sizes():
    cache_config = _cache_config(max_batches=8, cudagraph_capture_batch_sizes=[4, 1, 4, 16])

    assert cache_config.cudagraph_capture_batch_sizes == [1, 4, 8]


@pytest.mark.parametrize('sizes', [[], [0], [-1], [1.5], ['1'], [16]])
def test_invalid_capture_batch_sizes_raise(sizes):
    with pytest.raises(AssertionError):
        _cache_config(max_batches=8, cudagraph_capture_batch_sizes=sizes)


def test_capture_batch_size_miss_raises():
    engine_config = PytorchEngineConfig(max_batch_size=8, cudagraph_capture_batch_sizes=[1, 4])
    engine_config = ConfigBuilder.update_engine_config(engine_config)
    runner = object.__new__(CUDAGraphRunner)
    runner.cache_config = ConfigBuilder.build_cache_config(engine_config)

    assert runner._get_capture_tokens(5) == 8
    with pytest.raises(AssertionError):
        runner._get_capture_tokens(9)


def test_graph_runner_defensively_normalizes_capture_batch_sizes():
    cache_config = _cache_config(max_batches=8, cudagraph_capture_batch_sizes=[1, 8])
    cache_config.cudagraph_capture_batch_sizes = [4, 1, 4, 16]
    runner = object.__new__(CUDAGraphRunner)
    runner.cache_config = cache_config

    assert runner.get_capture_batch_sizes() == [1, 4, 8]


def test_graph_runner_reset_clears_padding_batch_size(monkeypatch):
    from lmdeploy.pytorch.backends.cuda import graph_runner as cuda_graph_runner

    runner = object.__new__(CUDAGraphRunner)
    runner._runner_meta = GraphRunnerMeta(padding_batch_size=1)
    runner._runner_map = {'stale': object()}
    monkeypatch.setattr(cuda_graph_runner.get_deepep_state(), 'enabled', lambda: False)

    runner.reset()

    assert runner.get_meta().padding_batch_size is None
    assert runner._runner_map == {}
