# Copyright (c) OpenMMLab. All rights reserved.
"""Verify the BF16 dispatch contract without any DeepEP extension API."""
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch import model_inputs
from lmdeploy.pytorch.backends.cuda import token_dispatcher

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')


@pytest.mark.parametrize('routes_prepared', [False, True])
@pytest.mark.parametrize('return_recv_hook', [False, True])
def test_dispatch_uses_native_bf16_api(monkeypatch, routes_prepared, return_recv_hook):
    hidden = torch.randn(6, 4096, device='cuda', dtype=torch.bfloat16)
    ids = torch.arange(8, device='cuda', dtype=torch.int64).expand(6, 8).contiguous()
    weights = torch.randn(6, 8, device='cuda', dtype=torch.float32)
    expected_ids = ids.clone()
    expected_ids[3:] = -1
    expected_weights = weights * 2.5
    if routes_prepared:
        ids.copy_(expected_ids)
        weights = expected_weights
    valid = torch.tensor(3, device='cuda', dtype=torch.int32)
    context = SimpleNamespace(moe_valid_token_count=valid, is_dummy=False)
    monkeypatch.setattr(model_inputs, 'get_step_ctx_manager',
                        lambda: SimpleNamespace(current_context=lambda: context))
    monkeypatch.setattr(token_dispatcher, 'mask_invalid_tokens',
                        lambda ids: pytest.fail('prepared routes must not launch the standalone mask'))
    calls = []

    class NativeBuffer:
        group_size = 8

        # Deliberately only accept the original public DeepEP arguments.
        def low_latency_dispatch(self, x, topk_idx, num_max_dispatch_tokens_per_rank,
                                 num_experts, use_fp8=True, async_finish=False, return_recv_hook=False):
            assert x is hidden and x.dtype == torch.bfloat16
            assert use_fp8 and async_finish == (not return_recv_hook)
            torch.testing.assert_close(topk_idx, expected_ids, rtol=0, atol=0)
            calls.append('dispatch')
            event = SimpleNamespace(current_stream_wait=lambda: calls.append('event'))
            return ((x, torch.ones(6, 32, device='cuda')), torch.zeros(36, device='cuda', dtype=torch.int32),
                    'handle', event, lambda: calls.append('hook'))

    dispatcher = token_dispatcher.DeepEPTokenDispatcherLowLatency.__new__(
        token_dispatcher.DeepEPTokenDispatcherLowLatency)
    dispatcher.buffer_low_latency = NativeBuffer()
    dispatcher.return_recv_hook = return_recv_hook
    dispatcher.num_max_dispatch_tokens_per_rank = 192
    result = dispatcher.dispatch(hidden, ids, weights, 288,
                                  output_scale=1.0 if routes_prepared else 2.5,
                                  routes_prepared=routes_prepared)
    torch.testing.assert_close(result[1], expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(result[2], expected_weights, rtol=0, atol=0)
    assert dispatcher.handle == 'handle'
    assert calls == ['dispatch', 'hook' if return_recv_hook else 'event']
