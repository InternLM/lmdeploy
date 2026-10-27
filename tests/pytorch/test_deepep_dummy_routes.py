# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from lmdeploy.pytorch.backends.cuda.graph_runner.full_graph import CUDASingleGraphRunner
from lmdeploy.pytorch.backends.cuda.token_dispatcher import DeepEPTokenDispatcherLowLatency
from lmdeploy.pytorch.kernels.cuda.moe.route_mask import mask_invalid_routes


def test_full_graph_rebinds_dummy_and_unpadded_count():
    """Changing a Python dummy flag must change an already-captured graph."""
    context = SimpleNamespace(dp_meta=object(), is_dummy=False, moe_valid_token_count=None)
    runner = object.__new__(CUDASingleGraphRunner)
    runner._graph = None
    runner._moe_valid_token_count = None
    runner._ctx_mgr = SimpleNamespace(current_context=lambda: context)
    runner.meta = SimpleNamespace(device=torch.device('cuda'))
    runner.model = Mock()
    runner.model.fill_buffers_cudagraph.return_value = {'padded': True}
    original = torch.arange(64, device='cuda', dtype=torch.int64).view(8, 8)
    routes = original.clone()
    runner._bind_inputs(input_ids=torch.zeros(1, 4, device='cuda', dtype=torch.long))
    count = context.moe_valid_token_count
    mask_invalid_routes(routes, count)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mask_invalid_routes(routes, count)
    for dummy, tokens in [(True, 4), (False, 3), (False, 8), (True, 1), (False, 4)]:
        context.is_dummy = dummy
        runner._bind_inputs(input_ids=torch.zeros(1, tokens, device='cuda', dtype=torch.long))
        assert context.moe_valid_token_count.data_ptr() == count.data_ptr()
        routes.copy_(original)
        graph.replay()
        expected = original.clone()
        expected[0 if dummy else tokens:] = -1
        torch.testing.assert_close(routes, expected, rtol=0, atol=0)


@pytest.mark.parametrize('asynchronous', [False, True])
def test_eager_dummy_keeps_collective_and_combine_routes(monkeypatch, asynchronous):
    """All ranks must dispatch even when this rank has no real token."""
    import lmdeploy.pytorch.model_inputs as inputs

    context = SimpleNamespace(is_dummy=True, moe_valid_token_count=None)
    monkeypatch.setattr(inputs, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    dispatcher = object.__new__(DeepEPTokenDispatcherLowLatency)
    dispatcher.num_experts = 288
    dispatcher.num_max_dispatch_tokens_per_rank = 128
    dispatcher.return_recv_hook = False
    event = Mock()
    buffer = Mock(group_size=8)
    buffer.low_latency_dispatch.return_value = ((torch.empty(0), torch.empty(0)),
                                               torch.zeros(36, dtype=torch.int32), 'handle', event, Mock())
    dispatcher.buffer_low_latency = buffer
    routes = torch.zeros(4, 8, dtype=torch.int64)
    weights = torch.ones(4, 8)
    hidden = torch.empty(4, 4096, dtype=torch.bfloat16)
    if asynchronous:
        dispatcher.dispatch_async(hidden, routes)
    else:
        result = dispatcher.dispatch(hidden, routes, weights, 288)
        assert result[1] is routes
        assert result[2] is weights
        event.current_stream_wait.assert_called_once()
    buffer.low_latency_dispatch.assert_called_once()
    assert buffer.low_latency_dispatch.call_args.args[1] is routes
    assert torch.all(routes == -1)
