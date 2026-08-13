import inspect

import pytest
import torch

FP8_DTYPE = torch.float8_e4m3fn
SCALE_BLOCK = 128


def _has_gluon() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(
    )[0] != 9:
        return False
    try:
        from lmdeploy.pytorch.backends.cuda.blockedf8_modules import _has_gluon as has_gluon
    except (AttributeError, ImportError):
        return False
    return has_gluon()


pytestmark = pytest.mark.skipif(
    not _has_gluon(),
    reason='Gluon WGMMA kernel requires supported Triton on Hopper')


def _reference(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
    sorted_idx: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_offset: int,
    reindex_a: bool,
    reindex_c: bool,
) -> torch.Tensor:
    num_routes = topk_ids.numel()
    top_k = topk_ids.size(1)
    flat_experts = topk_ids.flatten()
    output = torch.empty((num_routes, b.size(1)),
                         device=a.device,
                         dtype=torch.bfloat16)

    for sorted_pos in range(num_routes):
        source_route = int(sorted_idx[sorted_pos])
        local_expert = int(flat_experts[source_route]) - expert_offset
        a_row = source_route // top_k if reindex_a else sorted_pos
        c_row = source_route if reindex_c else sorted_pos

        partials = []
        for k_block in range(a.size(1) // SCALE_BLOCK):
            k_start = k_block * SCALE_BLOCK
            partial = (
                a[a_row, k_start:k_start + SCALE_BLOCK].float()
                @ b[local_expert, :, k_start:k_start + SCALE_BLOCK].float().T)
            block_b_scale = b_scale[local_expert, :, k_block]
            block_b_scale = block_b_scale.repeat_interleave(SCALE_BLOCK)
            partials.append(partial * a_scale[a_row, k_block] *
                            block_b_scale[:b.size(1)])
        value = torch.stack(partials).sum(0)
        if bias is not None:
            value += bias[local_expert]
        output[c_row] = value.to(torch.bfloat16)
    return output


@torch.inference_mode()
@pytest.mark.parametrize(
    ('reindex_a', 'reindex_c', 'block_m', 'block_n', 'with_bias',
     'num_stages', 'num_k_blocks', 'num_tokens', 'top_k',
     'num_local_experts', 'n'),
    [
        (True, False, 8, 64, False, 1, 3, 13, 2, 2, 128),
        (True, False, 8, 128, True, None, 9, 13, 2, 2, 128),
        (False, True, 8, 128, False, 2, 2, 13, 2, 2, 128),
        (False, True, 8, 128, True, 2, 5, 13, 2, 2, 128),
        (True, False, 64, 128, True, 1, 9, 65, 1, 1, 256),
        (False, True, 64, 128, True, 2, 2, 197, 2, 3, 256),
        (True, False, 64, 128, False, 3, 2, 65, 1, 1, 128),
        (True, False, 64, 128, False, 2, 5, 65, 1, 1, 128),
        (False, True, 64, 128, False, 3, 9, 65, 1, 1, 128),
    ],
)
def test_blocked_fp8_fused_moe_gluon(reindex_a, reindex_c, block_m, block_n,
                                     with_bias, num_stages, num_k_blocks,
                                     num_tokens, top_k, num_local_experts, n):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_gluon import (
        STANDARD_WGMMA_BLOCK_M,
        TRANSPOSED_WGMMA_BLOCK_M,
        fused_moe_blocked_fp8_kernel_launcher,
    )
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks

    torch.manual_seed(7)
    assert block_m in (TRANSPOSED_WGMMA_BLOCK_M, STANDARD_WGMMA_BLOCK_M)
    num_experts = num_local_experts + 2
    expert_offset = 1
    k = num_k_blocks * SCALE_BLOCK
    num_routes = num_tokens * top_k

    token = torch.arange(num_tokens, device='cuda', dtype=torch.int64)[:, None]
    rank = torch.arange(top_k, device='cuda', dtype=torch.int64)[None, :]
    topk_ids = (token * top_k + rank) % num_local_experts + expert_offset
    metadata = _get_sorted_idx_blocks(
        topk_ids,
        num_experts,
        num_local_experts,
        expert_offset,
        block_m,
    )
    sorted_idx, _, exp_end, block_end, block_expert_ids, block_offsets = metadata

    a_rows = num_tokens if reindex_a else num_routes
    a = (torch.randn((a_rows, k), device='cuda') * 0.25).to(FP8_DTYPE)
    a_scale = torch.rand(
        (a_rows, k // SCALE_BLOCK), device='cuda', dtype=torch.float32) + 0.5
    b = (torch.randn(
        (num_local_experts, n, k), device='cuda') * 0.25).to(FP8_DTYPE)
    b_scale = torch.rand(
        (num_local_experts, n // SCALE_BLOCK, k // SCALE_BLOCK),
        device='cuda',
        dtype=torch.float32,
    ) + 0.5
    if block_m == STANDARD_WGMMA_BLOCK_M and num_stages == 1:
        # Exercise the rescaled accumulator across zero-scale K blocks.
        a_scale[0, 2] = 0
        b_scale[0, 0, 4] = 0
    elif (block_m == STANDARD_WGMMA_BLOCK_M and num_stages == 2
          and num_k_blocks == 2):
        # Exercise the two-block normalized accumulator at zero scale.
        a_scale[0, 1] = 0
        b_scale[0, 0, 1] = 0
    bias = torch.randn(
        (num_local_experts,
         n), device='cuda', dtype=torch.bfloat16) if with_bias else None
    output = torch.full((num_routes, n),
                        torch.nan,
                        device='cuda',
                        dtype=torch.bfloat16)

    def run(result):
        fused_moe_blocked_fp8_kernel_launcher(
            a,
            a_scale,
            b,
            b_scale,
            result,
            sorted_idx,
            exp_end,
            block_end,
            block_expert_ids,
            block_offsets,
            bias=bias,
            top_k=top_k,
            expert_offset=expert_offset,
            reindex_a=reindex_a,
            reindex_c=reindex_c,
            block_m=block_m,
            block_n=block_n,
            num_stages=num_stages,
        )

    run(output)

    expected = _reference(
        a,
        a_scale,
        b,
        b_scale,
        bias,
        sorted_idx,
        topk_ids,
        expert_offset,
        reindex_a,
        reindex_c,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.08)

    repeated_output = torch.empty_like(output)
    run(repeated_output)
    torch.testing.assert_close(repeated_output, output, rtol=0, atol=0)


def test_blocked_fp8_fused_moe_gluon_selects_pipeline_from_workload():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_gluon import _select_transposed_pipeline_stages

    assert _select_transposed_pipeline_stages(m=8, k=6144) == 3
    assert _select_transposed_pipeline_stages(m=32, k=6144) == 3
    assert _select_transposed_pipeline_stages(m=33, k=6144) == 2
    assert _select_transposed_pipeline_stages(m=128, k=6144) == 2
    assert _select_transposed_pipeline_stages(m=129, k=6144) == 1
    assert _select_transposed_pipeline_stages(m=256, k=6144) == 1
    assert _select_transposed_pipeline_stages(m=768, k=6144) == 1
    assert _select_transposed_pipeline_stages(m=8,
                                              k=7 * SCALE_BLOCK) == 1
    assert _select_transposed_pipeline_stages(m=768, k=256) == 1


def test_blocked_fp8_fused_moe_gluon_rejects_wrong_scale_block():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_gluon import (
        fused_moe_blocked_fp8_kernel_launcher,
    )

    a = torch.empty((1, 128), device='cuda', dtype=FP8_DTYPE)
    a_scale = torch.empty((1, 2), device='cuda', dtype=torch.float32)
    b = torch.empty((1, 128, 128), device='cuda', dtype=FP8_DTYPE)
    b_scale = torch.empty((1, 1, 1), device='cuda', dtype=torch.float32)
    output = torch.empty((1, 128), device='cuda', dtype=torch.bfloat16)
    metadata = [
        torch.zeros(1, device='cuda', dtype=torch.int64) for _ in range(5)
    ]

    with pytest.raises(AssertionError,
                       match='A_scale must use one scale per 128 A columns'):
        fused_moe_blocked_fp8_kernel_launcher(
            a,
            a_scale,
            b,
            b_scale,
            output,
            *metadata,
        )


def test_blocked_fp8_fused_moe_gluon_has_baseline_api():
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import fused_moe_blocked_fp8 as baseline
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_gluon import fused_moe_blocked_fp8 as candidate

    def contract(fn):
        return [(parameter.name, parameter.kind, parameter.default)
                for parameter in inspect.signature(fn).parameters.values()]

    assert contract(candidate) == contract(baseline)


@torch.inference_mode()
@pytest.mark.parametrize('schedule',
                         ('transposed_wgmma_both',
                          'standard_wgmma_gate_triton_down',
                          'standard_wgmma_both_two_k_block_down'))
@pytest.mark.parametrize('custom_act', (False, True))
def test_blocked_fp8_fused_moe_gluon_complete_api_matches_baseline(
        monkeypatch, schedule, custom_act):
    from lmdeploy.pytorch.kernels.cuda.moe import blocked_fp8_gluon as candidate_module
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8 import fused_moe_blocked_fp8 as baseline

    monkeypatch.setattr(candidate_module, '_select_gluon_moe_schedule',
                        lambda *args: schedule)

    torch.manual_seed(11)
    num_tokens, num_experts, topk = 13, 4, 2
    hidden_features, gate_features, intermediate_features = 384, 512, 256
    input = (torch.randn((num_tokens, hidden_features), device='cuda') * 0.25).to(FP8_DTYPE)
    input_scale = torch.rand((num_tokens, hidden_features // SCALE_BLOCK), device='cuda') + 0.5
    w1 = (torch.randn((num_experts, gate_features, hidden_features), device='cuda') * 0.25).to(FP8_DTYPE)
    w1_scale = torch.rand((num_experts, gate_features // SCALE_BLOCK, hidden_features // SCALE_BLOCK),
                          device='cuda') + 0.5
    w2 = (torch.randn((num_experts, hidden_features, intermediate_features), device='cuda') * 0.25).to(FP8_DTYPE)
    w2_scale = torch.rand((num_experts, hidden_features // SCALE_BLOCK, intermediate_features // SCALE_BLOCK),
                          device='cuda') + 0.5
    topk_ids = torch.stack((torch.arange(num_tokens, device='cuda') % num_experts,
                            (torch.arange(num_tokens, device='cuda') + 1) % num_experts),
                           dim=1)
    topk_weights = torch.rand((num_tokens, topk), device='cuda')
    w1_bias = torch.randn((num_experts, gate_features), device='cuda', dtype=torch.bfloat16)
    w2_bias = torch.randn((num_experts, hidden_features), device='cuda', dtype=torch.bfloat16)
    args = (input, input_scale, w1, w1_scale, w2, w2_scale, topk_weights, topk_ids)
    act_func = (lambda x: torch.nn.functional.gelu(x[:, :x.size(1) // 2])) if custom_act else None

    expected = baseline(*args,
                        topk=topk,
                        w1_bias=w1_bias,
                        w2_bias=w2_bias,
                        out_dtype=torch.bfloat16,
                        renormalize=True,
                        act_func=act_func)
    actual = candidate_module.fused_moe_blocked_fp8(*args,
                                                    topk=topk,
                                                    w1_bias=w1_bias,
                                                    w2_bias=w2_bias,
                                                    out_dtype=torch.bfloat16,
                                                    renormalize=True,
                                                    act_func=act_func)
    repeated = candidate_module.fused_moe_blocked_fp8(*args,
                                                      topk=topk,
                                                      w1_bias=w1_bias,
                                                      w2_bias=w2_bias,
                                                      out_dtype=torch.bfloat16,
                                                      renormalize=True,
                                                      act_func=act_func)

    expected_scale = expected.abs().max().clamp_min(1e-6)
    actual_scale = actual.abs().max().clamp_min(1e-6)
    torch.testing.assert_close(actual / actual_scale,
                               expected / expected_scale,
                               rtol=1e-3,
                               atol=0.05)
    relative_scale_error = (actual_scale - expected_scale).abs() / expected_scale
    assert relative_scale_error < 0.05
    torch.testing.assert_close(repeated, actual, rtol=0, atol=0)


@pytest.mark.parametrize(('num_tokens', 'num_experts', 'topk', 'hidden_features', 'intermediate_features',
                          'expected'), [
                              (63, 256, 8, 2048, 512, None),
                              (64, 256, 8, 2048, 512, 'transposed_wgmma_both'),
                              (96, 256, 8, 3072, 512, 'transposed_wgmma_both'),
                              (128, 256, 8, 2048, 512, 'transposed_wgmma_both'),
                              (129, 256, 8, 2048, 512, None),
                              (192, 256, 4, 2048, 512, 'transposed_wgmma_both'),
                              (144, 384, 8, 2048, 512, 'transposed_wgmma_both'),
                              (48, 128, 8, 2048, 512, None),
                              (2048, 256, 8, 6144, 256, 'standard_wgmma_gate_triton_down'),
                              (3072, 384, 8, 6144, 256, 'standard_wgmma_gate_triton_down'),
                              (2048, 256, 8, 6144, 1024, None),
                              (2560, 256, 8, 6144, 256, None),
                              (8192, 256, 8, 6144, 256, None),
                          ])
def test_blocked_fp8_fused_moe_gluon_selects_schedule_from_launch_features(
        num_tokens, num_experts, topk, hidden_features,
        intermediate_features, expected):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_gluon import _select_gluon_moe_schedule

    gate_features = 2 * intermediate_features
    input = torch.empty((num_tokens, hidden_features), device='meta')
    w1 = torch.empty((num_experts, gate_features, hidden_features), device='meta')
    w2 = torch.empty((num_experts, hidden_features, intermediate_features), device='meta')
    topk_ids = torch.empty((num_tokens, topk), device='meta', dtype=torch.int64)

    assert _select_gluon_moe_schedule(input, w1, w2, topk_ids, num_experts) == expected


@pytest.mark.parametrize(
    ('num_tokens', 'num_experts', 'topk', 'hidden_features',
     'gate_features', 'intermediate_features', 'output_features', 'expected'),
    [
        (319, 256, 8, 6144, 512, 256, 6144, None),
        (320, 256, 8, 6144, 512, 256, 6144,
         'standard_wgmma_both_two_k_block_down'),
        (768, 256, 8, 6144, 512, 256, 6144,
         'standard_wgmma_both_two_k_block_down'),
        (769, 256, 8, 6144, 512, 256, 6144, None),
        (512, 384, 8, 6144, 512, 256, 6144,
         'standard_wgmma_both_two_k_block_down'),
        (512, 512, 10, 4096, 512, 256, 4096,
         'standard_wgmma_both_two_k_block_down'),
        (1024, 512, 10, 4096, 512, 256, 4096,
         'standard_wgmma_both_two_k_block_down'),
        (256, 512, 10, 4096, 512, 256, 4096, None),
        (512, 256, 8, 2048, 512, 256, 2048, None),
        (512, 512, 10, 4096, 1024, 256, 4096, None),
        (512, 512, 10, 4096, 512, 256, 2048, None),
        (3072, 512, 8, 6144, 512, 256, 6144, None),
        (512, 768, 10, 4096, 512, 256, 4096, None),
    ],
)
def test_blocked_fp8_fused_moe_gluon_selects_two_k_block_down_schedule(
        num_tokens, num_experts, topk, hidden_features, gate_features,
        intermediate_features, output_features, expected):
    from lmdeploy.pytorch.kernels.cuda.moe.blocked_fp8_gluon import _select_gluon_moe_schedule

    input = torch.empty((num_tokens, hidden_features), device='meta')
    w1 = torch.empty((num_experts, gate_features, hidden_features),
                     device='meta')
    w2 = torch.empty((num_experts, output_features, intermediate_features),
                     device='meta')
    topk_ids = torch.empty((num_tokens, topk),
                           device='meta',
                           dtype=torch.int64)

    assert _select_gluon_moe_schedule(input, w1, w2, topk_ids,
                                      num_experts) == expected


def test_blocked_fp8_fused_moe_gluon_falls_back(monkeypatch):
    from lmdeploy.pytorch.kernels.cuda.moe import blocked_fp8_gluon as candidate_module

    sentinel = object()
    monkeypatch.setattr(candidate_module, '_supports_gluon_moe_contract', lambda *args: False)
    monkeypatch.setattr(candidate_module, 'triton_fused_moe_blocked_fp8', lambda *args, **kwargs: sentinel)

    input = torch.empty((1, 128), device='cuda', dtype=FP8_DTYPE)
    input_scale = torch.empty((1, 1), device='cuda')
    w1 = torch.empty((1, 256, 128), device='cuda', dtype=FP8_DTYPE)
    w1_scale = torch.empty((1, 2, 1), device='cuda')
    w2 = torch.empty((1, 128, 128), device='cuda', dtype=FP8_DTYPE)
    w2_scale = torch.empty((1, 1, 1), device='cuda')
    topk_weights = torch.ones((1, 1), device='cuda')
    topk_ids = torch.zeros((1, 1), device='cuda', dtype=torch.int64)

    assert candidate_module.fused_moe_blocked_fp8(input,
                                                   input_scale,
                                                   w1,
                                                   w1_scale,
                                                   w2,
                                                   w2_scale,
                                                   topk_weights,
                                                   topk_ids,
                                                   topk=1) is sentinel
