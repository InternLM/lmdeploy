# Copyright (c) OpenMMLab. All rights reserved.
"""Operator-owned PCG slicing; real capture/state checks are separate GPU tests."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import piecewise_graph_execution
from lmdeploy.pytorch.backends.cuda.kda import CudaKdaImpl
from lmdeploy.pytorch.backends.cuda.kpool import CudaKPoolAttention


@pytest.mark.parametrize('raw_tokens', [1, 3, 8])
@pytest.mark.parametrize('kind', ['kda', 'kpool'])
def test_piecewise_boundary_uses_live_raw_extent_and_preserves_eager(monkeypatch, kind, raw_tokens):
    # No graph builder: this tests the operator's slicing/ownership contract,
    # not graph capture, padded bridges, recurrence, or numerical equivalence.
    from lmdeploy.pytorch.backends.cuda import step_metadata
    registered = []
    monkeypatch.setattr(step_metadata, 'register_piecewise_graph_impl', registered.append)
    original = Mock(side_effect=lambda *args, **kwargs: args[0])
    if kind == 'kda':
        operator = CudaKdaImpl.__new__(CudaKdaImpl)
        operator.forward = original
        operator._piecewise_forward = None
        axes = [1, 1, 1]
    else:
        operator = CudaKPoolAttention(original)
        assert registered == [operator]
        axes = [1, 1, 0, 0, 0]
    inputs = [torch.arange(32).reshape(1, 8, 4) if axis == 1
              else torch.arange(32).reshape(8, 4) for axis in axes]
    state = object()
    assert operator.supports_piecewise_cuda_graph()
    operator.enable_piecewise_cuda_graph()
    forward = operator.forward
    operator.enable_piecewise_cuda_graph()
    assert operator.forward is forward
    for live_count in [raw_tokens, 8]:
        metadata = SimpleNamespace(version=live_count)
        with piecewise_graph_execution(raw_tokens=live_count, token_bucket=8):
            result = operator.forward(*inputs, metadata=metadata, cache=state)
        args, kwargs = original.call_args
        assert kwargs['metadata'] is metadata and kwargs['cache'] is state
        for value, source, axis in zip(args, inputs, axes):
            assert value.shape[axis] == live_count
            assert value.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        assert result is args[0]
    # Leaving PCG must not retain the most recent extent or metadata.
    metadata = object()
    result = operator.forward(*inputs, metadata=metadata, cache=state)
    args, kwargs = original.call_args
    assert all(value is source for value, source in zip(args, inputs))
    assert result is inputs[0]
    assert kwargs == {'metadata': metadata, 'cache': state}
