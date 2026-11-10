# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch


@pytest.mark.parametrize('lengths', [(1, 72, 137), (129, 75, 63), (64, 64, 64)])
@torch.inference_mode()
def test_indexed_gated_delta_preserves_first_chunk_precision(lengths):
    """Fresh and cached sequences must use the same gate math as FLA."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    from lmdeploy.pytorch.backends.cuda.gated_delta_rule import CudaCompactGatedDeltaMeta, CudaGatedDeltaRuleImpl
    from lmdeploy.pytorch.compact_state import SSMPrefillMetadata, SSMStateMetadata

    torch.manual_seed(451)
    total, dim, heads = sum(lengths), 128, 2
    plan = SSMPrefillMetadata.from_lengths(lengths, [0, 128, 64]).to_device('cuda')
    banks = torch.tensor([[3, 4], [5, 6], [7, 8]], device='cuda')
    source = torch.tensor([0, 1, 1], device='cuda')
    destinations = banks[:, :1].contiguous()
    ids = SSMStateMetadata(banks, source, destinations)
    meta = CudaCompactGatedDeltaMeta(total, False, ids, plan)
    pool = torch.randn(10, heads, dim, dim, dtype=torch.bfloat16, device='cuda') / 16
    pool[0].zero_()
    before = pool.clone()
    q, k = [torch.randn(1, total, 1, dim, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    v = torch.randn(1, total, heads, dim, dtype=torch.bfloat16, device='cuda')
    a, b = [torch.randn(1, total, heads, dtype=torch.bfloat16, device='cuda') for _ in range(2)]
    dt, a_exp = torch.zeros(heads, device='cuda'), -torch.ones(heads, device='cuda')
    rule = CudaGatedDeltaRuleImpl(indexed_state=True)
    prep_q, prep_k, g, beta, _ = rule.prepare_inputs(q, k, b, a, dt, a_exp, 2, True, False, None)
    expected, final = chunk_gated_delta_rule(
        prep_q, prep_k, v, g=g, beta=beta, initial_state=pool[source].contiguous(),
        output_final_state=True, cu_seqlens=plan.cu_seqlens, use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True)
    actual = rule.forward(q, k, v, b, a, dt, a_exp, pool, meta, 2, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool[destinations[:, 0]], final.to(pool.dtype), rtol=0, atol=0)
    untouched = torch.ones(len(pool), dtype=torch.bool, device='cuda')
    untouched[destinations[:, 0]] = False
    torch.testing.assert_close(pool[untouched], before[untouched], rtol=0, atol=0)

    # Reused, dirty destinations must not become initial state for a fresh request.
    again = rule.forward(q, k, v, b, a, dt, a_exp, pool, meta, 2, True)
    torch.testing.assert_close(again, expected, rtol=0, atol=0)

    # Resume all three sequences from their saved states, including the formerly
    # fresh sequence. Its first gate must now retain the supplied history.
    resumed_source = destinations[:, 0].contiguous()
    resumed_ids = SSMStateMetadata(banks, resumed_source, banks[:, 1:].contiguous())
    resumed_plan = SSMPrefillMetadata.from_lengths(lengths, lengths).to_device('cuda')
    resumed_meta = CudaCompactGatedDeltaMeta(total, False, resumed_ids, resumed_plan)
    expected, final = chunk_gated_delta_rule(
        prep_q, prep_k, v, g=g, beta=beta, initial_state=pool[resumed_source].contiguous(),
        output_final_state=True, cu_seqlens=plan.cu_seqlens, use_qk_l2norm_in_kernel=False,
        transpose_state_layout=True)
    actual = rule.forward(q, k, v, b, a, dt, a_exp, pool, resumed_meta, 2, True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool[banks[:, 1]], final.to(pool.dtype), rtol=0, atol=0)
    torch.testing.assert_close(pool[0], before[0], rtol=0, atol=0)
