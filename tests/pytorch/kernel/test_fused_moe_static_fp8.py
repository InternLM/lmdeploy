# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.kernels.cuda.fused_moe import (
    moe_reduce,
)
from lmdeploy.pytorch.kernels.cuda.w8a8_fused_moe import (
    fused_moe_static_fp8,
)
from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import (
    matmul_kernel_static_quant,
    per_tensor_quant_fp8,
)


def _quantize_expert_weights(
    weight: torch.Tensor,
    quant_dtype: torch.dtype,
):
    """Quantize each expert using one weight scale."""
    num_experts, out_features, _ = weight.shape
    fp8_max = torch.finfo(quant_dtype).max

    scales = (
        weight.float()
        .abs()
        .amax(dim=(1, 2))
        / fp8_max
    ).clamp_min(1e-8)

    quantized = torch.empty_like(
        weight,
        dtype=quant_dtype,
    )

    for expert_id in range(num_experts):
        quantized[expert_id] = (
            per_tensor_quant_fp8(
                weight[expert_id],
                scales[expert_id],
                quant_dtype=quant_dtype,
            )
        )

    expanded_scales = (
        scales.float()
        .reshape(num_experts, 1, 1)
        .expand(
            num_experts,
            out_features,
            1,
        )
        .contiguous()
    )

    return quantized.contiguous(), expanded_scales


@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 9,
    reason='require device with cc>=9.0',
)
@pytest.mark.parametrize('num_tokens', [1, 6])
@torch.inference_mode()
def test_fused_moe_static_fp8(num_tokens):
    """Compare fused static FP8 MoE with per-route Linear."""
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)

    device = torch.device('cuda')
    quant_dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16

    num_experts = 3
    top_k = 2
    hidden_dim = 128
    ffn_dim = 64

    hidden_states = (
        torch.randn(
            num_tokens,
            hidden_dim,
            device=device,
            dtype=out_dtype,
        )
        * 0.2
    )

    gate_up_weight_bf16 = (
        torch.randn(
            num_experts,
            ffn_dim * 2,
            hidden_dim,
            device=device,
            dtype=out_dtype,
        )
        * 0.1
    )

    down_weight_bf16 = (
        torch.randn(
            num_experts,
            hidden_dim,
            ffn_dim,
            device=device,
            dtype=out_dtype,
        )
        * 0.1
    )

    (
        gate_up_weight,
        gate_up_weight_scale,
    ) = _quantize_expert_weights(
        gate_up_weight_bf16,
        quant_dtype,
    )

    (
        down_weight,
        down_weight_scale,
    ) = _quantize_expert_weights(
        down_weight_bf16,
        quant_dtype,
    )

    fp8_max = torch.finfo(quant_dtype).max

    gate_up_input_scale = (
        hidden_states.float()
        .abs()
        .max()
        / fp8_max
    ).clamp_min(1e-8).reshape(1)

    down_input_scale = torch.tensor(
        [1e-3],
        device=device,
        dtype=torch.float32,
    )

    route_pattern = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 2],
            [1, 0],
            [2, 1],
        ],
        device=device,
        dtype=torch.long,
    )

    weight_pattern = torch.tensor(
        [
            [0.7, 0.3],
            [0.6, 0.4],
            [0.8, 0.2],
            [0.55, 0.45],
            [0.65, 0.35],
            [0.75, 0.25],
        ],
        device=device,
        dtype=torch.float32,
    )

    topk_ids = route_pattern[
        :num_tokens
    ].contiguous()

    topk_weights = weight_pattern[
        :num_tokens
    ].contiguous()

    reference_routes = torch.empty(
        num_tokens,
        top_k,
        hidden_dim,
        device=device,
        dtype=out_dtype,
    )

    for token_id in range(num_tokens):
        token_input = hidden_states[
            token_id:token_id + 1
        ].contiguous()

        for route_id in range(top_k):
            expert_id = int(
                topk_ids[
                    token_id,
                    route_id,
                ].item()
            )

            gate_up = matmul_kernel_static_quant(
                token_input,
                gate_up_weight[expert_id],
                gate_up_input_scale,
                gate_up_weight_scale[
                    expert_id,
                    :,
                    0,
                ].contiguous(),
                output_dtype=out_dtype,
            )

            gate, up = gate_up.chunk(
                2,
                dim=-1,
            )

            activated = (
                F.silu(gate.float())
                * up.float()
            ).to(out_dtype)

            route_output = (
                matmul_kernel_static_quant(
                    activated,
                    down_weight[expert_id],
                    down_input_scale,
                    down_weight_scale[
                        expert_id,
                        :,
                        0,
                    ].contiguous(),
                    output_dtype=out_dtype,
                )
            )

            reference_routes[
                token_id,
                route_id,
            ].copy_(route_output[0])

    expected = moe_reduce(
        reference_routes,
        topk_weights,
    )

    observed = fused_moe_static_fp8(
        hidden_states,
        gate_up_input_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_input_scale,
        down_weight,
        down_weight_scale,
        topk_weights,
        topk_ids,
        topk=top_k,
        out_dtype=out_dtype,
        quant_dtype=quant_dtype,
        renormalize=False,
    )

    torch.testing.assert_close(
        observed.float(),
        expected.float(),
        atol=2e-3,
        rtol=5e-3,
    )
