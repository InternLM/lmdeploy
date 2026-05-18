# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from types import SimpleNamespace

import pytest

from lmdeploy.cli.utils import ArgumentHelper
from lmdeploy.messages import PytorchEngineConfig
from lmdeploy.pytorch.backends.cuda.graph_runner import (
    CUDAGraphRunner,
    _get_capture_batch_size_impl,
)
from lmdeploy.pytorch.config import CacheConfig
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder


def _cache_config(max_batches=8, cudagraph_capture_batch_sizes=None):
    return CacheConfig(max_batches=max_batches,
                       block_size=64,
                       num_cpu_blocks=0,
                       num_gpu_blocks=1,
                       cudagraph_capture_batch_sizes=cudagraph_capture_batch_sizes)


def test_default_capture_batch_sizes_are_unchanged():
    cache_config = _cache_config(max_batches=512)
    runner = object.__new__(CUDAGraphRunner)
    runner.cache_config = cache_config

    assert runner.get_capture_batch_sizes() == _get_capture_batch_size_impl(512)


def test_custom_capture_batch_sizes_are_normalized_in_engine_config():
    engine_config = PytorchEngineConfig(max_batch_size=8, cudagraph_capture_batch_sizes=[8, 1, 4, 4])

    engine_config = ConfigBuilder.update_engine_config(engine_config)
    cache_config = ConfigBuilder.build_cache_config(engine_config)

    assert engine_config.cudagraph_capture_batch_sizes == [1, 4, 8]
    assert cache_config.cudagraph_capture_batch_sizes == [1, 4, 8]


def test_capture_batch_sizes_larger_than_max_batch_size_are_filtered():
    engine_config = PytorchEngineConfig(max_batch_size=8, cudagraph_capture_batch_sizes=[1, 4, 16])

    engine_config = ConfigBuilder.update_engine_config(engine_config)

    assert engine_config.cudagraph_capture_batch_sizes == [1, 4]


@pytest.mark.parametrize('sizes', [[], [0], [-1], [1.5], ['1'], [16]])
def test_invalid_capture_batch_sizes_raise(sizes):
    engine_config = PytorchEngineConfig(max_batch_size=8, cudagraph_capture_batch_sizes=sizes)

    with pytest.raises(AssertionError):
        ConfigBuilder.update_engine_config(engine_config)


def test_graph_runner_uses_custom_capture_batch_sizes():
    cache_config = _cache_config(max_batches=8, cudagraph_capture_batch_sizes=[1, 4])
    runner = object.__new__(CUDAGraphRunner)
    runner.cache_config = cache_config

    assert runner.get_capture_batch_sizes() == [1, 4]
    assert runner._get_capture_tokens(2) == 4
    assert runner._get_capture_tokens(8) is None


def test_runtime_batch_larger_than_capture_sizes_falls_back_to_eager_forward():

    class FakeModel:

        def __call__(self, **kwargs):
            return 'eager-output'

        def make_output_buffers(self, output):
            return {'output': output}

    runner = object.__new__(CUDAGraphRunner)
    runner.backend_config = SimpleNamespace(eager_mode=True)
    runner.model = FakeModel()
    runner._prepare_inputs = lambda **kwargs: kwargs
    runner.enable_graph = lambda **kwargs: True
    runner.get_graph_key = lambda **kwargs: (None, True, False, 1)

    assert runner(input_ids='dummy', attn_metadata='dummy') == {'output': 'eager-output'}


def test_cudagraph_capture_batch_sizes_cli_arg():
    parser = argparse.ArgumentParser()
    ArgumentHelper.cudagraph_capture_batch_sizes(parser)

    args = parser.parse_args(['--cudagraph-capture-batch-sizes', '1', '2', '4', '8'])

    assert args.cudagraph_capture_batch_sizes == [1, 2, 4, 8]
