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


@pytest.mark.parametrize('bf16', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
@pytest.mark.parametrize('valid_count', [None, 0, 1, 3, 4])
def test_all_ll_entries_mask_before_transport(monkeypatch, bf16, asynchronous, valid_count):
    """Exercise the real FP8/BF16 helpers with only DeepEP transport mocked."""
    import lmdeploy.pytorch.model_inputs as inputs
    from lmdeploy.pytorch.backends.cuda.moe.default import dispatch_async_ll, dispatch_ll

    count = None if valid_count is None else torch.tensor(valid_count, device='cuda', dtype=torch.int32)
    context = SimpleNamespace(is_dummy=valid_count is None, moe_valid_token_count=count)
    monkeypatch.setattr(inputs, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    dispatcher = object.__new__(DeepEPTokenDispatcherLowLatency)
    dispatcher.num_experts = 288
    dispatcher.num_max_dispatch_tokens_per_rank = 128
    dispatcher.return_recv_hook = False
    event = Mock()
    hook = Mock()
    buffer = Mock(group_size=4)
    received = torch.empty(0, device='cuda', dtype=torch.bfloat16)
    buffer.low_latency_dispatch.return_value = (
        received if bf16 else (received, received),
        torch.zeros(72, device='cuda', dtype=torch.int32), 'handle', event, hook)
    dispatcher.buffer_low_latency = buffer
    original = torch.arange(8, device='cuda', dtype=torch.int64).view(4, 2)
    original[0, 0] = -1
    routes = original.clone()
    weights = torch.ones(4, 2, device='cuda')
    hidden = torch.empty(4, 4096, device='cuda', dtype=torch.bfloat16)
    if asynchronous:
        if bf16:
            result = dispatch_async_ll(dispatcher, hidden, routes, use_fp8=False, async_finish=False)
        else:
            result = dispatcher.dispatch_async(hidden, routes, async_finish=False)
        assert result[2:] == ('handle', event, hook)
    else:
        if bf16:
            result = dispatch_ll(dispatcher, hidden, routes, weights, 288, use_fp8=False)
        else:
            result = dispatcher.dispatch(hidden, routes, weights, 288)
        assert result[1] is routes
        assert result[2] is weights
        event.current_stream_wait.assert_called_once()
    expected = original.clone()
    expected[0 if valid_count is None else valid_count:] = -1
    buffer.low_latency_dispatch.assert_called_once()
    call = buffer.low_latency_dispatch.call_args
    assert call.args[0] is hidden
    assert call.args[1] is routes
    assert call.kwargs['use_fp8'] is (not bf16)
    torch.testing.assert_close(routes, expected, rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.ones_like(weights), rtol=0, atol=0)


@pytest.mark.parametrize('rows,attn_tp,attn_rank',
                         [(4, 2, 0), (4, 2, 1), (5, 2, 0), (5, 2, 1), (1, 2, 0), (1, 2, 1), (4, 1, 0)])
def test_tp_slices_keep_global_mask_coordinates_across_replay(monkeypatch, rows, attn_tp, attn_rank):
    """Count stays device-dynamic; TP rank 1 must not retain global padding."""
    import lmdeploy.pytorch.model_inputs as inputs
    from lmdeploy.pytorch.backends.cuda.moe import ep_utils
    from lmdeploy.pytorch.distributed import DistConfig

    dist_context = SimpleNamespace(dist_config=DistConfig(dp=2, tp=attn_tp, ep=2 * attn_tp),
                                   attn_tp_group=SimpleNamespace(rank=attn_rank))
    monkeypatch.setattr(ep_utils, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: dist_context))
    count = torch.tensor(rows, device='cuda', dtype=torch.int32)
    context = SimpleNamespace(is_dummy=False, moe_valid_token_count=count)
    monkeypatch.setattr(inputs, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    original = torch.arange(rows * 2, device='cuda', dtype=torch.int64).view(rows, 2)
    original[0, 0] = -1
    routes = original.clone()
    hidden = torch.randn(rows, 4, device='cuda')
    weights = torch.rand(rows, 2, device='cuda')

    def run():
        x, w, ids, sizes = ep_utils.split_inputs_by_attn_tp(hidden, weights, routes, mask_padding=True)
        # The shared LL entry mask must never unmask the already-filtered slice.
        ep_utils.mask_invalid_tokens(ids)
        return x, w, ids, sizes

    x, w, ids, sizes = run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        x, w, ids, sizes = run()
    for valid in [rows, 3 if rows >= 3 else rows, 1, 0, rows]:
        routes.copy_(original)
        count.fill_(valid)
        graph.replay()
        expected = original.clone()
        expected[valid:] = -1
        if sizes is not None:
            expected = expected.split(sizes)[attn_rank]
        torch.testing.assert_close(ids, expected, rtol=0, atol=0)
        expected_x = hidden if sizes is None else hidden.split(sizes)[attn_rank]
        expected_w = weights if sizes is None else weights.split(sizes)[attn_rank]
        torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
        torch.testing.assert_close(w, expected_w, rtol=0, atol=0)
    # The default splitter remains unchanged for normal-throughput callers.
    routes.copy_(original)
    count.zero_()
    ep_utils.split_inputs_by_attn_tp(hidden, weights, routes)
    torch.testing.assert_close(routes, original, rtol=0, atol=0)


@pytest.mark.parametrize('backend', ['bf16', 'fp8', 'packed'])
@pytest.mark.parametrize('valid_count', [1, 3])
def test_moe_forward_masks_before_tp_split(monkeypatch, backend, valid_count):
    """The three synchronous MoE frontends must opt into pre-split masking."""
    import lmdeploy.pytorch.model_inputs as inputs
    from lmdeploy.pytorch.backends.cuda.moe import blocked_fp8, compressed_tensors, default, ep_utils
    from lmdeploy.pytorch.distributed import DistConfig

    context = SimpleNamespace(is_dummy=False, global_is_decoding=lambda: True,
                              moe_valid_token_count=torch.tensor(valid_count, device='cuda', dtype=torch.int32))
    def manager():
        return SimpleNamespace(current_context=lambda: context)

    monkeypatch.setattr(inputs, 'get_step_ctx_manager', manager)
    dist_context = SimpleNamespace(dist_config=DistConfig(dp=2, tp=2, ep=4),
                                   attn_tp_group=SimpleNamespace(rank=1))
    monkeypatch.setattr(ep_utils, 'get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: dist_context))
    module = {'bf16': default, 'fp8': blocked_fp8, 'packed': compressed_tensors}[backend]
    monkeypatch.setattr(module, 'get_step_ctx_manager', manager)
    monkeypatch.setattr(module, 'gather_outputs_by_attn_tp', lambda x, sizes: x)
    original = torch.arange(8, device='cuda', dtype=torch.int64).view(4, 2)
    routes = original.clone()
    hidden = torch.randn(4, 4, device='cuda')
    weights = torch.rand(4, 2, device='cuda')
    seen = []

    def experts(x, w, ids, *args, **kwargs):
        # Check before the LL entry's second mask: TP-local rows already carry
        # global validity, including rank 1's entirely empty slice at count=1.
        seen.append(ids.clone())
        ep_utils.mask_invalid_tokens(ids)
        return x

    obj = SimpleNamespace(use_deep_gemm=True, renormalize=False,
                          do_renormalize=lambda x: x,
                          fusedmoe_build=lambda mode: SimpleNamespace(forward=experts),
                          _validate_local_weights=lambda *args: None,
                          _forward_low_latency=experts)
    if backend == 'bf16':
        default.FusedMoEEPImpl._forward_moe(obj, hidden, weights, routes, None, None)
    elif backend == 'fp8':
        blocked_fp8.FusedDeepEpMoEBlockedF8Impl._forward_moe(
            obj, hidden, weights, routes, None, None, None, None)
    else:
        compressed_tensors.DeepEPFusedMoEW4A16Impl.forward(
            obj, hidden, weights, routes, None, None, None, None)
    expected = original.clone()
    expected[valid_count:] = -1
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], expected[2:], rtol=0, atol=0)
