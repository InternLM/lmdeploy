# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


@pytest.mark.parametrize('batch', [1, 3])
@pytest.mark.parametrize('steps', [1, 3, 6])
@pytest.mark.parametrize('channel_major', [False, True])
@pytest.mark.parametrize('state_dtype', [torch.float32, torch.bfloat16])
def test_kda_strided_inputs_match_contiguous_state_ring(batch, steps, channel_major, state_dtype):
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    torch.manual_seed(17)
    heads, dim, ring = 2, 128, 6
    if channel_major:
        mixed = torch.randn(batch, 3 * heads * dim, steps, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    else:
        mixed = torch.randn(batch, steps, 3 * heads * dim, device='cuda', dtype=torch.bfloat16)
    q, k, v = [x.unflatten(-1, (heads, dim)) for x in mixed.chunk(3, dim=-1)]
    gate = -torch.rand(batch, steps, heads, dim, device='cuda')
    beta = torch.rand(batch, steps, heads, device='cuda')
    initial = torch.randn(3, ring, heads, dim, dim, device='cuda', dtype=state_dtype) * 0.1
    ids = torch.tensor([2, -1, 0][:batch], device='cuda')
    history = torch.tensor([5, 0, 3][:batch], device='cuda', dtype=torch.int32)
    kwargs = dict(g=gate, beta=beta, state_indices=ids, cache_seqlens=history,
                  output_final_state=True, transpose_state_layout=True, use_qk_l2norm_in_kernel=True)
    expected, expected_state = fused_recurrent_gated_delta_rule(
        q.contiguous(), k.contiguous(), v.contiguous(), initial_state=initial.clone(), **kwargs)
    actual, actual_state = fused_recurrent_gated_delta_rule(q, k, v, initial_state=initial.clone(), **kwargs)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(actual_state[1], initial[1], rtol=0, atol=0)
    if batch == 3:
        assert torch.count_nonzero(actual[1]) == 0


def test_kda_strided_graph_replay_changes_inputs_and_history():
    from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule

    torch.manual_seed(31)
    heads, dim, steps = 2, 128, 6
    mixed = torch.randn(1, 3 * heads * dim, steps, device='cuda', dtype=torch.bfloat16)
    q, k, v = [x.unflatten(-1, (heads, dim)) for x in mixed.transpose(1, 2).chunk(3, dim=-1)]
    gate = -torch.rand_like(q, dtype=torch.float32).contiguous()
    beta = torch.rand(1, steps, heads, device='cuda')
    initial = torch.randn(1, steps, heads, dim, dim, device='cuda') * 0.1
    state = initial.clone()
    history = torch.tensor([5], device='cuda', dtype=torch.int32)
    kwargs = dict(g=gate, beta=beta, cache_seqlens=history, output_final_state=True,
                  transpose_state_layout=True, use_qk_l2norm_in_kernel=True)

    def run():
        return fused_recurrent_gated_delta_rule(q, k, v, initial_state=state, **kwargs)

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, _ = run()
    mixed.normal_()
    history.fill_(2)
    state.copy_(initial)
    expected, expected_state = fused_recurrent_gated_delta_rule(
        q.contiguous(), k.contiguous(), v.contiguous(), initial_state=initial.clone(), **kwargs)
    graph.replay()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
