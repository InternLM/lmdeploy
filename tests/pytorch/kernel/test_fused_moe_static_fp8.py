# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import lmdeploy.pytorch.kernels.cuda.moe.w8a8 as w8a8_module
from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import (
    moe_reduce,
)
from lmdeploy.pytorch.kernels.cuda.moe.w8a8 import (
    _per_tensor_quant_fp8_e4m3fn_inductor,
    _scalar_static_fp8_quant,
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
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='require device with cc>=9.0',
)
@pytest.mark.parametrize('num_tokens', [1, 6])
@pytest.mark.parametrize('scale_mode', ['scalar', 'gate_up', 'down', 'both'])
@pytest.mark.parametrize('use_compiled_quant', [False, True])
@torch.inference_mode()
def test_fused_moe_static_fp8(
    monkeypatch,
    num_tokens,
    scale_mode,
    use_compiled_quant,
):
    """Compare scalar and per-expert scale modes with per-route Linear."""
    monkeypatch.setattr(
        w8a8_module,
        '_USE_COMPILED_STATIC_FP8_QUANT',
        use_compiled_quant,
    )
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

    if scale_mode in ('gate_up', 'both'):
        gate_up_input_scale = (
            gate_up_input_scale
            * torch.tensor(
                [1.0, 1.25, 1.5],
                device=device,
                dtype=torch.float32,
            )
        ).contiguous()

    if scale_mode in ('down', 'both'):
        down_input_scale = torch.tensor(
            [8e-4, 1.5e-3, 3e-3],
            device=device,
            dtype=torch.float32,
        )
    else:
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

            gate_up_scale = gate_up_input_scale
            if gate_up_scale.numel() > 1:
                gate_up_scale = gate_up_scale[expert_id:expert_id + 1]

            gate_up = matmul_kernel_static_quant(
                token_input,
                gate_up_weight[expert_id],
                gate_up_scale,
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

            down_scale = down_input_scale
            if down_scale.numel() > 1:
                down_scale = down_scale[expert_id:expert_id + 1]

            route_output = (
                matmul_kernel_static_quant(
                    activated,
                    down_weight[expert_id],
                    down_scale,
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

@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='require device with cc>=9.0',
)
@torch.inference_mode()
def test_fused_moe_static_fp8_per_expert_scales_with_expert_offset():
    """Use local scale vectors with global expert IDs."""
    torch.manual_seed(2027)
    device = torch.device('cuda')
    quant_dtype = torch.float8_e4m3fn
    out_dtype = torch.bfloat16
    num_global_experts = 4
    expert_offset = 1
    num_local_experts = 2
    hidden_dim = 128
    ffn_dim = 64
    hidden_states = (
        torch.randn(
            num_global_experts,
            hidden_dim,
            device=device,
            dtype=out_dtype,
        )
        * 0.2
    )
    gate_up_weight, gate_up_weight_scale = _quantize_expert_weights(
        torch.randn(
            num_local_experts,
            ffn_dim * 2,
            hidden_dim,
            device=device,
            dtype=out_dtype,
        )
        * 0.1,
        quant_dtype,
    )
    down_weight, down_weight_scale = _quantize_expert_weights(
        torch.randn(
            num_local_experts,
            hidden_dim,
            ffn_dim,
            device=device,
            dtype=out_dtype,
        )
        * 0.1,
        quant_dtype,
    )
    gate_up_input_scale = torch.tensor([8e-4, 1.2e-3], device=device, dtype=torch.float32)
    down_input_scale = torch.tensor([1.5e-3, 2.5e-3], device=device, dtype=torch.float32)
    topk_ids = torch.arange(num_global_experts, device=device, dtype=torch.long).reshape(-1, 1)
    topk_weights = torch.ones(num_global_experts, 1, device=device, dtype=torch.float32)
    local_slice = slice(expert_offset, expert_offset + num_local_experts)
    local_reference = fused_moe_static_fp8(
        hidden_states[local_slice].contiguous(),
        gate_up_input_scale,
        gate_up_weight,
        gate_up_weight_scale,
        down_input_scale,
        down_weight,
        down_weight_scale,
        topk_weights[local_slice].contiguous(),
        torch.arange(
            num_local_experts,
            device=device,
            dtype=torch.long,
        ).reshape(-1, 1),
        topk=1,
        out_dtype=out_dtype,
        quant_dtype=quant_dtype,
    )
    expected = torch.zeros_like(hidden_states)
    expected[local_slice] = local_reference
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
        topk=1,
        out_dtype=out_dtype,
        quant_dtype=quant_dtype,
        expert_offset=expert_offset,
        num_experts=num_global_experts,
    )
    torch.testing.assert_close(observed, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='require device with cc>=9.0',
)
@pytest.mark.parametrize('num_tokens', [1, 32, 256])
@pytest.mark.parametrize('hidden_dim', [384, 4096])
@torch.inference_mode()
def test_moe_compiled_static_fp8_quant_is_exact(
    num_tokens,
    hidden_dim,
):
    generator = torch.Generator(device='cuda')
    generator.manual_seed(
        20260730 + num_tokens + hidden_dim,
    )
    x = torch.randn(
        (num_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device='cuda',
        generator=generator,
    )
    scale = torch.tensor(
        [0.0125],
        dtype=torch.float32,
        device='cuda',
    )

    expected = per_tensor_quant_fp8(
        x,
        scale,
        quant_dtype=torch.float8_e4m3fn,
    )
    actual = _per_tensor_quant_fp8_e4m3fn_inductor(
        x,
        scale,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='require device with cc>=9.0',
)
@torch.inference_mode()
def test_moe_compiled_quant_keeps_other_fp8_dtype_fallback(
    monkeypatch,
):
    monkeypatch.setattr(
        w8a8_module,
        '_USE_COMPILED_STATIC_FP8_QUANT',
        True,
    )
    x = torch.randn(
        (32, 384),
        dtype=torch.bfloat16,
        device='cuda',
    )
    scale = torch.tensor(
        [0.0125],
        dtype=torch.float32,
        device='cuda',
    )
    expected = per_tensor_quant_fp8(
        x,
        scale,
        quant_dtype=torch.float8_e5m2,
    )

    def _unexpected_compiled_call(*args, **kwargs):
        raise AssertionError(
            'compiled E4M3 quant must not handle E5M2',
        )

    monkeypatch.setattr(
        w8a8_module,
        '_per_tensor_quant_fp8_e4m3fn_inductor',
        _unexpected_compiled_call,
    )
    actual = _scalar_static_fp8_quant(
        x,
        scale,
        torch.float8_e5m2,
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason='require device with cc>=9.0',
)
@pytest.mark.parametrize(
    ('num_tokens', 'hidden_dim'),
    [
        (1, 4096),
        (256, 384),
    ],
)
@torch.inference_mode()
def test_moe_compiled_static_fp8_quant_cuda_graph(
    num_tokens,
    hidden_dim,
):
    static_x = torch.randn(
        (num_tokens, hidden_dim),
        dtype=torch.bfloat16,
        device='cuda',
    )
    scale = torch.tensor(
        [0.0125],
        dtype=torch.float32,
        device='cuda',
    )

    # Compile before capture.
    _per_tensor_quant_fp8_e4m3fn_inductor(
        static_x,
        scale,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = (
            _per_tensor_quant_fp8_e4m3fn_inductor(
                static_x,
                scale,
            )
        )

    for replay_id in range(1, 4):
        replay_x = torch.randn_like(static_x)
        replay_x.mul_(replay_id / 4)
        static_x.copy_(replay_x)
        graph.replay()
        torch.cuda.synchronize()

        expected = per_tensor_quant_fp8(
            replay_x,
            scale,
            quant_dtype=torch.float8_e4m3fn,
        )
        assert torch.equal(graph_output, expected)
