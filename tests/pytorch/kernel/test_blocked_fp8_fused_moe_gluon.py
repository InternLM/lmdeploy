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
            partials.append(partial * a_scale[a_row, k_block] *
                            b_scale[local_expert, :, k_block])
        value = torch.stack(partials).sum(0)
        if bias is not None:
            value += bias[local_expert]
        output[c_row] = value.to(torch.bfloat16)
    return output


@torch.inference_mode()
@pytest.mark.parametrize(
    ('reindex_a', 'reindex_c', 'block_n', 'with_bias', 'num_stages',
     'num_k_blocks'),
    [
        (True, False, 64, False, 1, 3),
        (True, False, 128, True, None, 9),
        (False, True, 128, True, 2, 5),
    ],
)
def test_blocked_fp8_fused_moe_gluon(reindex_a, reindex_c, block_n, with_bias,
                                     num_stages, num_k_blocks):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe_gluon import (
        SMALL_M_BLOCK,
        fused_moe_blocked_fp8_kernel_launcher,
    )
    from lmdeploy.pytorch.kernels.cuda.fused_moe import _get_sorted_idx_blocks

    torch.manual_seed(7)
    num_tokens = 13
    top_k = 2
    num_experts = 4
    num_local_experts = 2
    expert_offset = 1
    n = 128
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
        SMALL_M_BLOCK,
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
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe_gluon import _select_pipeline_stages

    assert _select_pipeline_stages(m=8, k=6144) == 3
    assert _select_pipeline_stages(m=32, k=6144) == 3
    assert _select_pipeline_stages(m=33, k=6144) == 2
    assert _select_pipeline_stages(m=128, k=6144) == 2
    assert _select_pipeline_stages(m=129, k=6144) == 1
    assert _select_pipeline_stages(m=256, k=6144) == 1
    assert _select_pipeline_stages(m=768, k=6144) == 1
    assert _select_pipeline_stages(m=8, k=7 * SCALE_BLOCK) == 1
    assert _select_pipeline_stages(m=768, k=256) == 1


def test_blocked_fp8_fused_moe_gluon_rejects_wrong_scale_block():
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe_gluon import (
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
