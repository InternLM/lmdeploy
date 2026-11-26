# Copyright (c) 2026, DeepLink. All rights reserved.

from types import SimpleNamespace

import torch

from lmdeploy.pytorch.backends.ascend.graph_runner import (
    AscendGraphRunner,
    _get_capture_batch_size_impl,
)
from lmdeploy.pytorch.backends.graph_runner import GraphRunnerMeta


def test_capture_size_generation_is_pure():
    _get_capture_batch_size_impl.cache_clear()
    first = _get_capture_batch_size_impl(33)
    _get_capture_batch_size_impl.cache_clear()
    second = _get_capture_batch_size_impl(33)

    assert first == second
    assert first[-1] == 33


def test_reset_preserves_configured_capture_sizes(monkeypatch):
    runner = object.__new__(AscendGraphRunner)
    runner._runner_meta = GraphRunnerMeta(padding_batch_size=16)
    runner._runner_map = {}
    runner.graph_pool_handle = object()
    runner.cache_config = SimpleNamespace(
        max_batches=32,
        cudagraph_capture_batch_sizes=[1, 4, 16, 32],
    )
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(empty_cache=lambda: None), raising=False)

    before_reset = runner.get_capture_batch_sizes().copy()
    runner.reset()
    after_reset = runner.get_capture_batch_sizes().copy()

    assert runner.get_meta().padding_batch_size is None
    assert runner.graph_pool_handle is None
    assert before_reset == after_reset == [1, 4, 16, 32]
