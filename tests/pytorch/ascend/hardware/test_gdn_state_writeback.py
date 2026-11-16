# Copyright (c) OpenMMLab. All rights reserved.
"""Check every speculative recurrent-state slot, including graph replay."""
import pytest
import torch


def _reference(q, k, v, g, beta, initial, history, slots):
    q, k, v, g, beta = (x.float().cpu() for x in (q, k, v, g, beta))
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    repeats = v.shape[2] // q.shape[2]
    q, k = (x.repeat_interleave(repeats, dim=2) for x in (q, k))
    states = initial.cpu().clone()
    state = states[slots + history % slots].clone()
    output = []
    for token in range(q.shape[1]):
        state = state * g[0, token].exp()[:, None, None]
        innovation = v[0, token] - (state * k[0, token, :, None, :]).sum(-1)
        state = state + (innovation * beta[0, token, :, None])[:, :, None] * k[0, token, :, None, :]
        output.append((state * (q[0, token, :, None, :] / q.shape[-1]**0.5)).sum(-1))
        states[slots + (history + token + 1) % slots] = state
    return torch.stack(output).unsqueeze(0), states


@pytest.mark.parametrize('tokens', [1, 2, 4])
@pytest.mark.parametrize('graph_mode', [False, True])
@torch.inference_mode()
def test_circular_state_writeback(tokens, graph_mode):
    from lmdeploy.pytorch.kernels.ascend.triton_ops.fla.fused_recurrent import fused_recurrent_gated_delta_rule

    torch.manual_seed(42)
    torch.npu.manual_seed_all(42)
    device = 'npu'
    slots, history = max(2, tokens), 27
    # Preserve the fixed-input reproducer's CPU/device RNG split. Generating
    # every input on device did not reliably expose the original write hazard.
    q = torch.randn(1, tokens, 8, 128, dtype=torch.bfloat16, device='cpu').to(device)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, 16, 128, dtype=torch.bfloat16, device='cpu').to(device)
    g = -torch.rand(1, tokens, 16, dtype=torch.float32, device='cpu').to(device)
    beta = torch.rand(1, tokens, 16, dtype=torch.bfloat16, device='cpu').to(device)
    initial = torch.randn(2 * slots, 16, 128, 128, dtype=torch.float32, device='cpu').to(device) * .01
    state = initial.clone()
    cu_seqlens = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    state_ids = torch.tensor([1], dtype=torch.int64, device=device)
    cache_seqlens = torch.tensor([history], dtype=torch.int32, device=device)
    expected_out, expected_state = _reference(q, k, v, g, beta, initial, history, slots)

    def forward():
        return fused_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=state, inplace_final_state=True,
            cu_seqlens=cu_seqlens, cache_seqlens_rb=cache_seqlens,
            state_ids_rb=state_ids, num_state=slots, use_qk_l2norm_in_kernel=True)[0]

    forward()
    torch.npu.synchronize()
    graph = None
    if graph_mode:
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, auto_dispatch_capture=True):
            captured = forward()
        torch.npu.synchronize()
    first_state = None
    try:
        for _ in range(32):
            state.copy_(initial)
            if graph is None:
                output = forward()
            else:
                graph.replay()
                output = captured
            torch.npu.synchronize()
            actual_state = state.cpu()
            torch.testing.assert_close(actual_state, expected_state, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(output.float().cpu(), expected_out, atol=1e-3, rtol=1e-2)
            if first_state is None:
                first_state = actual_state.clone()
            else:
                torch.testing.assert_close(actual_state, first_state, atol=0, rtol=0)
    finally:
        if graph is not None:
            graph.reset()
