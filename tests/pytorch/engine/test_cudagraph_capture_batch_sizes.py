# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch

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
    from lmdeploy.pytorch.backends.cuda.graph_runner import runner as cuda_graph_runner

    runner = object.__new__(CUDAGraphRunner)
    runner._runner_meta = GraphRunnerMeta(padding_batch_size=1)
    runner._full_graph_runners = {'stale': object()}
    runner._piecewise_graph_manager = None
    monkeypatch.setattr(cuda_graph_runner.get_deepep_state(), 'enabled', lambda: False)

    runner.reset()

    assert runner.get_meta().padding_batch_size is None
    assert runner._full_graph_runners == {}


def test_model_context_kwarg_does_not_collide_with_graph_routing(monkeypatch):
    from lmdeploy.pytorch.backends.cuda.graph_runner import runner as cuda_graph_runner

    step_context = SimpleNamespace(global_is_decoding=lambda: False)
    runner = object.__new__(CUDAGraphRunner)
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: step_context)
    runner.enable_graph = lambda **kwargs: False
    runner._piecewise_graph_manager = None
    monkeypatch.setattr(cuda_graph_runner.get_deepep_state(), 'enabled', lambda: False)

    model_context = object()
    runner._forward_eager = lambda **kwargs: kwargs['context']
    output = runner(
        attn_metadata=SimpleNamespace(block_offsets=torch.zeros(1, 1, dtype=torch.int32)),
        context=model_context,
    )

    assert output is model_context


@pytest.mark.parametrize('tokens,decoding,microbatch,capacity,low_latency', [
    ([31, 42, 1], False, False, 192, True),
    ([192, 1], False, False, 192, True),
    ([193, 1], False, False, 192, False),
    ([31, 42], False, True, 192, False),
    ([31, 42], False, False, 0, False),
    ([6, 7], True, False, 0, True),
])
def test_small_prefill_mode_agrees_across_ranks_without_changing_attention(
        monkeypatch, tokens, decoding, microbatch, capacity, low_latency):
    from lmdeploy.pytorch.backends.cuda.graph_runner import runner as cuda_graph_runner
    from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPBuffer, DeepEPMode
    from lmdeploy.pytorch.model_inputs import DPMeta, StepContext

    state = cuda_graph_runner.get_deepep_state()
    monkeypatch.setattr(state, 'enabled', lambda: True)
    monkeypatch.setattr(state, 'low_latency_prefill_tokens', capacity)
    modes = []
    monkeypatch.setattr(DeepEPBuffer, 'set_deepep_mode', modes.append)
    runner = object.__new__(CUDAGraphRunner)
    runner.enable_graph = lambda **kwargs: True

    for rank, count in enumerate(tokens):
        context = object.__new__(StepContext)
        context.input_ids = torch.zeros(1, count, dtype=torch.long)
        context.dp_meta = DPMeta.build(count, tokens)
        context.dp_meta.dp_is_decoding = decoding
        context.is_decoding = decoding or rank > 0
        context.enable_microbatch = microbatch
        cuda_graph_runner._update_deepep_mode(context)
        assert context.moe_low_latency is low_latency
        assert context.global_is_decoding() is decoding
        assert context.is_decoding is (decoding or rank > 0)
        assert runner._should_use_full_graph(context, {}) is decoding

    expected = DeepEPMode.LOW_LATENCY if low_latency else DeepEPMode.NORMAL
    assert modes == [expected] * len(tokens)
