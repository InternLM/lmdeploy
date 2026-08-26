# Copyright (c) OpenMMLab. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason='requires CUDA')


def _pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Pack offset-binary INT4 codes, least-significant nibble first."""
    assert qweight.dtype == torch.int32
    assert qweight.shape[-1] % 8 == 0
    codes = (qweight.to(torch.int64) + 8) & 0xF
    shifts = torch.arange(8, dtype=torch.int64, device=qweight.device) * 4
    return torch.sum(codes.unflatten(-1, (-1, 8)) << shifts,
                     dim=-1).to(torch.int32)


def _dequantize(qweight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return (qweight.float() * scales.float().repeat_interleave(32, dim=-1)).to(
        torch.bfloat16)


def _quality(actual: torch.Tensor,
             expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float().flatten()
    expected = expected.float().flatten()
    nrmse = ((actual - expected).square().mean().sqrt() /
             expected.square().mean().sqrt()).item()
    cosine = F.cosine_similarity(actual, expected, dim=0).item()
    return nrmse, cosine


@pytest.mark.parametrize('activation_dtype',
                         [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('out_features,in_features',
                         [(96, 64), (96, 7168)])
@torch.inference_mode()
def test_direct_packed_w4a16_gemm_matches_dequantized_reference(
        activation_dtype, out_features, in_features):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_kernel_launcher

    torch.manual_seed(1)
    device = torch.device('cuda')
    num_experts, num_tokens = 1, 7

    qweight = torch.randint(-8,
                            8, (num_experts, out_features, in_features),
                            dtype=torch.int32,
                            device=device)
    scales = (torch.rand(
        num_experts, out_features, in_features // 32, device=device) * 0.05 +
              0.005).to(torch.bfloat16)
    packed = _pack_int4(qweight)
    hidden_states = torch.randn(num_tokens,
                                in_features,
                                dtype=activation_dtype,
                                device=device)
    output = torch.empty(num_tokens,
                         out_features,
                         dtype=activation_dtype,
                         device=device)
    sorted_idx = torch.arange(num_tokens, dtype=torch.int64, device=device)
    exp_start = torch.tensor([0], dtype=torch.int64, device=device)
    exp_end = torch.tensor([num_tokens], dtype=torch.int64, device=device)

    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        packed,
        scales,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        num_tokens=num_tokens,
    )

    reference = (
        hidden_states @
        _dequantize(qweight, scales)[0].to(activation_dtype).T)
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@pytest.mark.parametrize('top_k', [2, 8])
@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@torch.inference_mode()
def test_direct_packed_w4a16_tiny_routed_moe_matches_reference(
        top_k, index_dtype, monkeypatch):
    from lmdeploy.pytorch.kernels.cuda import compressed_tensors_w4a16

    reduce_kwargs = []
    original_moe_reduce = compressed_tensors_w4a16.moe_reduce

    def _record_moe_reduce(hidden_states, topk_weights, **kwargs):
        reduce_kwargs.append(kwargs)
        return original_moe_reduce(hidden_states, topk_weights, **kwargs)

    monkeypatch.setattr(compressed_tensors_w4a16, 'moe_reduce',
                        _record_moe_reduce)

    def _unexpected_sort(*args, **kwargs):
        raise AssertionError('small decode must use route-major scheduling')

    monkeypatch.setattr(compressed_tensors_w4a16, '_get_sorted_idx',
                        _unexpected_sort)
    monkeypatch.setattr(compressed_tensors_w4a16, '_get_sorted_idx_blocks',
                        _unexpected_sort)

    torch.manual_seed(2)
    device = torch.device('cuda')
    num_experts, num_tokens = 384, 5
    hidden_dim, ffn_dim = 64, 64

    gate_up_qweight = torch.randint(-8,
                                    8, (num_experts, 2 * ffn_dim, hidden_dim),
                                    dtype=torch.int32,
                                    device=device)
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(-8,
                                 8, (num_experts, hidden_dim, ffn_dim),
                                 dtype=torch.int32,
                                 device=device)
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    # Every token selects distinct experts; most Kimi-scale experts stay empty.
    topk_ids = torch.arange(top_k, dtype=index_dtype,
                            device=device).expand(num_tokens, -1).contiguous()
    topk_weights = torch.rand(num_tokens,
                              top_k,
                              dtype=torch.float32,
                              device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    output = compressed_tensors_w4a16.fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=top_k,
    )
    assert reduce_kwargs == [dict(fp32_acc=True)]

    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    reference = torch.zeros_like(hidden_states)
    for expert_id in range(num_experts):
        token_idx, route_idx = torch.where(topk_ids == expert_id)
        if token_idx.numel() == 0:
            continue
        gate_up = hidden_states[token_idx] @ gate_up_weight[expert_id].T
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        expert_output = activated @ down_weight[expert_id].T
        expert_output *= topk_weights[token_idx, route_idx, None]
        reference.index_add_(0, token_idx, expert_output)

    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@torch.inference_mode()
def test_route_major_w4a16_graph_replays_dynamic_expert_ids(monkeypatch):
    from lmdeploy.pytorch.kernels.cuda import compressed_tensors_w4a16

    def _unexpected_sort(*args, **kwargs):
        raise AssertionError('CUDA Graph decode must not build sort metadata')

    monkeypatch.setattr(compressed_tensors_w4a16, '_get_sorted_idx',
                        _unexpected_sort)
    monkeypatch.setattr(compressed_tensors_w4a16, '_get_sorted_idx_blocks',
                        _unexpected_sort)

    torch.manual_seed(11)
    device = torch.device('cuda')
    num_experts, num_tokens, top_k = 128, 4, 2
    hidden_dim = ffn_dim = 64
    gate_up_qweight = torch.randint(
        -8,
        8, (num_experts, 2 * ffn_dim, hidden_dim),
        dtype=torch.int32,
        device=device,
    )
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(
        -8,
        8, (num_experts, hidden_dim, ffn_dim),
        dtype=torch.int32,
        device=device,
    )
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    gate_up_packed = _pack_int4(gate_up_qweight)
    down_packed = _pack_int4(down_qweight)
    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)

    def run(hidden_states, topk_weights, topk_ids):
        return compressed_tensors_w4a16.fused_moe_w4a16(
            hidden_states,
            gate_up_packed,
            gate_up_scale,
            down_packed,
            down_scale,
            topk_weights,
            topk_ids,
            topk=top_k,
        )

    def reference(hidden_states, topk_weights, topk_ids):
        expected = torch.zeros_like(hidden_states)
        for token_id in range(num_tokens):
            for route_id in range(top_k):
                expert_id = int(topk_ids[token_id, route_id])
                gate_up = (
                    hidden_states[token_id] @
                    gate_up_weight[expert_id].T)
                gate, up = gate_up.chunk(2, dim=-1)
                expert_output = (
                    F.silu(gate) * up) @ down_weight[expert_id].T
                expected[token_id] += (
                    expert_output * topk_weights[token_id, route_id])
        return expected

    static_hidden = torch.randn(
        num_tokens,
        hidden_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    static_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.float32,
        device=device,
    )
    static_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 127], [127, 0]],
        dtype=torch.int32,
        device=device,
    )
    warm_output = run(static_hidden, static_weights, static_ids)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = run(static_hidden, static_weights, static_ids)

    next_hidden = torch.randn_like(static_hidden)
    next_weights = torch.rand_like(static_weights)
    # Include duplicate experts and the highest legal id.  These ids are read
    # by the captured kernel at replay time, not frozen on the host.
    next_ids = torch.tensor(
        [[127, 127], [0, 2], [1, 1], [2, 0]],
        dtype=torch.int32,
        device=device,
    )
    static_hidden.copy_(next_hidden)
    static_weights.copy_(next_weights)
    static_ids.copy_(next_ids)
    graph.replay()
    torch.cuda.synchronize()

    expected = reference(next_hidden, next_weights, next_ids)
    nrmse, cosine = _quality(graph_output, expected)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999
    assert not torch.equal(graph_output, warm_output)


@torch.inference_mode()
def test_direct_packed_w4a16_masks_deepep_invalid_local_routes():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

    torch.manual_seed(19)
    device = torch.device('cuda')
    num_experts, num_tokens, top_k = 4, 5, 2
    hidden_dim = ffn_dim = 64

    gate_up_qweight = torch.randint(
        -8,
        8, (num_experts, 2 * ffn_dim, hidden_dim),
        dtype=torch.int32,
        device=device)
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(
        -8,
        8, (num_experts, hidden_dim, ffn_dim),
        dtype=torch.int32,
        device=device)
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    topk_ids = torch.tensor(
        [[-1, 0], [1, -1], [2, 3], [-1, -1], [3, 1]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.0, 0.7], [0.4, 0.0], [0.6, 0.8], [0.0, 0.0], [0.5, 0.2]],
        dtype=torch.float32,
        device=device,
    )

    output = fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=top_k,
        allow_invalid_routes=True,
    )

    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    reference = torch.zeros_like(hidden_states)
    for token_id in range(num_tokens):
        for route_id in range(top_k):
            expert_id = int(topk_ids[token_id, route_id])
            if expert_id < 0:
                continue
            gate_up = hidden_states[token_id] @ gate_up_weight[expert_id].T
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = (F.silu(gate) * up) @ down_weight[expert_id].T
            reference[token_id] += (
                expert_output * topk_weights[token_id, route_id])

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[3]) == 0
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@torch.inference_mode()
def test_direct_packed_w4a16_allows_topk_wider_than_local_experts():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

    torch.manual_seed(23)
    device = torch.device('cuda')
    num_experts, num_tokens, top_k = 1, 3, 2
    hidden_dim = ffn_dim = 64

    gate_up_qweight = torch.randint(
        -8,
        8, (num_experts, 2 * ffn_dim, hidden_dim),
        dtype=torch.int32,
        device=device)
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(
        -8,
        8, (num_experts, hidden_dim, ffn_dim),
        dtype=torch.int32,
        device=device)
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    topk_ids = torch.tensor(
        [[0, -1], [-1, 0], [0, -1]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.tensor(
        [[0.7, 0.0], [0.0, 0.4], [0.6, 0.0]],
        dtype=torch.float32,
        device=device,
    )

    output = fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=top_k,
        allow_invalid_routes=True,
    )

    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    reference = torch.zeros_like(hidden_states)
    for token_id in range(num_tokens):
        route_id = int((topk_ids[token_id] == 0).nonzero().item())
        gate_up = hidden_states[token_id] @ gate_up_weight[0].T
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = (F.silu(gate) * up) @ down_weight[0].T
        reference[token_id] = (
            expert_output * topk_weights[token_id, route_id])

    assert torch.isfinite(output).all()
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@torch.inference_mode()
def test_masked_w4a16_static_expert_layout_matches_reference_and_graph():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_masked

    torch.manual_seed(29)
    device = torch.device('cuda')
    num_experts, capacity = 3, 4
    hidden_dim = ffn_dim = 64
    gate_up_qweight = torch.randint(
        -8,
        8, (num_experts, 2 * ffn_dim, hidden_dim),
        dtype=torch.int32,
        device=device,
    )
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(
        -8,
        8, (num_experts, hidden_dim, ffn_dim),
        dtype=torch.int32,
        device=device,
    )
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    gate_up_packed = _pack_int4(gate_up_qweight)
    down_packed = _pack_int4(down_qweight)
    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)

    def reference(hidden_states, masked_m):
        expected = torch.zeros_like(hidden_states)
        for expert_id, count in enumerate(masked_m.tolist()):
            if count == 0:
                continue
            gate_up = (
                hidden_states[expert_id, :count] @
                gate_up_weight[expert_id].T)
            gate, up = gate_up.chunk(2, dim=-1)
            expected[expert_id, :count] = (
                (F.silu(gate) * up) @ down_weight[expert_id].T)
        return expected

    # Warm every Triton specialization before CUDA Graph capture.
    hidden_states = torch.randn(
        num_experts,
        capacity,
        hidden_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    masked_m = torch.tensor([4, 2, 0], dtype=torch.int32, device=device)
    warm_output = fused_moe_w4a16_masked(
        hidden_states,
        gate_up_packed,
        gate_up_scale,
        down_packed,
        down_scale,
        masked_m,
    )
    warm_reference = reference(hidden_states, masked_m.cpu())
    warm_nrmse, warm_cosine = _quality(warm_output, warm_reference)
    assert warm_nrmse <= 1e-2
    assert warm_cosine >= 0.9999
    assert torch.count_nonzero(warm_output[1, 2:]) == 0
    assert torch.count_nonzero(warm_output[2]) == 0

    static_hidden = hidden_states.clone()
    static_masked_m = masked_m.clone()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        graph_output = fused_moe_w4a16_masked(
            static_hidden,
            gate_up_packed,
            gate_up_scale,
            down_packed,
            down_scale,
            static_masked_m,
        )

    next_hidden = torch.randn_like(static_hidden)
    next_masked_m = torch.tensor(
        [1, 4, 3], dtype=torch.int32, device=device)
    static_hidden.copy_(next_hidden)
    static_masked_m.copy_(next_masked_m)
    graph.replay()
    torch.cuda.synchronize()

    graph_reference = reference(next_hidden, next_masked_m.cpu())
    graph_nrmse, graph_cosine = _quality(graph_output, graph_reference)
    assert graph_nrmse <= 1e-2
    assert graph_cosine >= 0.9999
    assert torch.count_nonzero(graph_output[0, 1:]) == 0
    assert torch.count_nonzero(graph_output[2, 3:]) == 0


@torch.inference_mode()
def test_direct_packed_w4a16_kimi_combine_uses_fp32_semantics():
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import moe_reduce

    torch.manual_seed(17)
    device = torch.device('cuda')
    num_tokens, top_k, hidden_dim = 25, 8, 7168
    expert_output = torch.randn(num_tokens,
                                top_k,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    topk_weights = torch.rand(num_tokens,
                              top_k,
                              dtype=torch.float32,
                              device=device)
    topk_weights *= 2.827 / topk_weights.sum(dim=-1, keepdim=True)

    actual = moe_reduce(expert_output, topk_weights, fp32_acc=True)
    reference = (expert_output.float() *
                 topk_weights[..., None]).sum(dim=1).to(torch.bfloat16)
    bf16_acc = moe_reduce(expert_output, topk_weights, fp32_acc=False)

    fp32_nrmse, fp32_cosine = _quality(actual, reference)
    bf16_nrmse, _ = _quality(bf16_acc, reference)

    # Triton and torch may reduce the eight routes in a different order, so a
    # handful of values can land on adjacent BF16 codes.  The FP32 path should
    # nevertheless match the HF combine semantics within rounding noise and
    # be decisively more accurate than the former BF16 combine.
    assert fp32_nrmse <= 5e-5
    assert fp32_cosine >= 0.999999
    assert fp32_nrmse <= bf16_nrmse * 0.01


@torch.inference_mode()
def test_direct_packed_w4a16_single_expert_route():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16

    torch.manual_seed(7)
    device = torch.device('cuda')
    num_tokens, hidden_dim, ffn_dim = 3, 64, 64
    gate_up_qweight = torch.randint(-8,
                                    8, (1, 2 * ffn_dim, hidden_dim),
                                    dtype=torch.int32,
                                    device=device)
    gate_up_scale = (
        torch.rand(1, 2 * ffn_dim, hidden_dim // 32, device=device) * 0.03 +
        0.002).to(torch.bfloat16)
    down_qweight = torch.randint(-8,
                                 8, (1, hidden_dim, ffn_dim),
                                 dtype=torch.int32,
                                 device=device)
    down_scale = (
        torch.rand(1, hidden_dim, ffn_dim // 32, device=device) * 0.03 +
        0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    topk_ids = torch.zeros(num_tokens, 1, dtype=torch.int64, device=device)
    topk_weights = torch.ones(num_tokens,
                              1,
                              dtype=torch.float32,
                              device=device)

    output = fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=1,
    )

    gate_up = hidden_states @ _dequantize(gate_up_qweight, gate_up_scale)[0].T
    gate, up = gate_up.chunk(2, dim=-1)
    reference = (F.silu(gate) * up) @ _dequantize(down_qweight,
                                                  down_scale)[0].T
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@pytest.mark.parametrize('top_k', [2, 8])
@torch.inference_mode()
def test_direct_packed_w4a16_skewed_distinct_routes_and_down_reindex(top_k):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_kernel_launcher
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx

    torch.manual_seed(3 + top_k)
    device = torch.device('cuda')
    num_tokens, in_features, intermediate_features = 33, 64, 64
    num_experts = top_k + 1

    gate_qweight = torch.randint(
        -8,
        8, (num_experts, intermediate_features, in_features),
        dtype=torch.int32,
        device=device)
    gate_scale = (torch.rand(
        num_experts, intermediate_features, in_features // 32, device=device) *
                  0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(
        -8,
        8, (num_experts, in_features, intermediate_features),
        dtype=torch.int32,
        device=device)
    down_scale = (torch.rand(
        num_experts, in_features, intermediate_features // 32, device=device) *
                  0.03 + 0.002).to(torch.bfloat16)
    gate_weight = _dequantize(gate_qweight, gate_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    hidden_states = torch.randn(num_tokens,
                                in_features,
                                dtype=torch.bfloat16,
                                device=device)

    # Every token selects the same distinct experts. Each active expert therefore
    # receives the maximum legal skew (num_tokens routes); the last expert is empty.
    topk_ids = torch.arange(top_k, dtype=torch.int64,
                            device=device).expand(num_tokens, -1).contiguous()
    sorted_idx, exp_start, exp_end = _get_sorted_idx(topk_ids, num_experts)
    gate_sorted = torch.full((num_tokens, top_k, intermediate_features),
                             torch.nan,
                             dtype=torch.bfloat16,
                             device=device)
    fused_moe_w4a16_kernel_launcher(
        hidden_states,
        _pack_int4(gate_qweight),
        gate_scale,
        gate_sorted,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=top_k,
        num_tokens=num_tokens,
        reindex_a=True,
        reindex_c=False,
    )

    gate_reference = torch.empty_like(gate_sorted)
    for route_idx in range(top_k):
        gate_reference[:, route_idx] = hidden_states @ gate_weight[route_idx].T
    sorted_reference = gate_reference.flatten(0, 1)[sorted_idx]
    gate_nrmse, gate_cosine = _quality(gate_sorted.flatten(0, 1),
                                       sorted_reference)
    assert gate_nrmse <= 1e-2
    assert gate_cosine >= 0.9999

    # Gate/up writes expert-sorted rows. Down consumes those sorted offsets and
    # scatters each result back to its original route via sorted_idx.
    output = torch.full((num_tokens, top_k, in_features),
                        torch.nan,
                        dtype=torch.bfloat16,
                        device=device)
    fused_moe_w4a16_kernel_launcher(
        gate_sorted,
        _pack_int4(down_qweight),
        down_scale,
        output,
        sorted_idx,
        exp_start,
        exp_end,
        top_k=1,
        num_tokens=num_tokens,
        reindex_a=False,
        reindex_c=True,
    )

    reference = torch.empty_like(output)
    for route_idx in range(top_k):
        reference[:, route_idx] = gate_reference[:, route_idx] @ down_weight[
            route_idx].T
    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999


@pytest.mark.parametrize('index_dtype', [torch.int32, torch.int64])
@torch.inference_mode()
def test_direct_packed_w4a16_compacts_sparse_kimi_scale_routes(index_dtype):
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import (
        _should_use_compact_w4a16,
        _should_use_route_w4a16,
        _use_hopper_w4a16,
        _w4a16_block_m,
        fused_moe_w4a16,
    )
    from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx_blocks

    torch.manual_seed(13)
    device = torch.device('cuda')
    num_experts, num_tokens, top_k = 384, 65, 2
    hidden_dim = ffn_dim = 64

    gate_up_qweight = torch.randint(-8,
                                    8, (num_experts, 2 * ffn_dim, hidden_dim),
                                    dtype=torch.int32,
                                    device=device)
    gate_up_scale = (
        torch.rand(num_experts, 2 * ffn_dim, hidden_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    down_qweight = torch.randint(-8,
                                 8, (num_experts, hidden_dim, ffn_dim),
                                 dtype=torch.int32,
                                 device=device)
    down_scale = (
        torch.rand(num_experts, hidden_dim, ffn_dim // 32, device=device) *
        0.03 + 0.002).to(torch.bfloat16)
    hidden_states = torch.randn(num_tokens,
                                hidden_dim,
                                dtype=torch.bfloat16,
                                device=device)
    # Model the sparse routing pattern of Kimi's 384 experts: only two experts
    # are active, including the highest legal expert id.
    topk_ids = torch.tensor([0, num_experts - 1],
                            dtype=index_dtype,
                            device=device).expand(num_tokens, -1).contiguous()
    topk_weights = torch.rand(num_tokens,
                              top_k,
                              dtype=torch.float32,
                              device=device)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    output = fused_moe_w4a16(
        hidden_states,
        _pack_int4(gate_up_qweight),
        gate_up_scale,
        _pack_int4(down_qweight),
        down_scale,
        topk_weights,
        topk_ids,
        topk=top_k,
    )

    gate_up_weight = _dequantize(gate_up_qweight, gate_up_scale)
    down_weight = _dequantize(down_qweight, down_scale)
    reference = torch.zeros_like(hidden_states)
    for route_idx, expert_id in enumerate((0, num_experts - 1)):
        gate_up = hidden_states @ gate_up_weight[expert_id].T
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = (F.silu(gate) * up) @ down_weight[expert_id].T
        reference += expert_output * topk_weights[:, route_idx, None]

    nrmse, cosine = _quality(output, reference)
    assert nrmse <= 1e-2
    assert cosine >= 0.9999

    prefer_small_tiles = _use_hopper_w4a16(device)
    block_m = _w4a16_block_m(num_tokens, topk_ids.numel(), num_experts,
                              prefer_small_tiles)
    (_, _, _, block_end, block_expert_ids,
     block_offsets) = _get_sorted_idx_blocks(topk_ids, num_experts,
                                             num_experts, 0, block_m)
    expected_blocks_per_active_expert = (num_tokens + block_m - 1) // block_m
    actual_blocks = int(block_end[-1].item())
    assert actual_blocks == 2 * expected_blocks_per_active_expert
    assert block_expert_ids[:actual_blocks].tolist() == (
        [0] * expected_blocks_per_active_expert +
        [num_experts - 1] * expected_blocks_per_active_expert)
    expected_offsets = (list(range(0, num_tokens, block_m)) +
                        list(range(num_tokens, 2 * num_tokens, block_m)))
    assert block_offsets[:actual_blocks].tolist() == expected_offsets

    # For the M4 8K/top-8 target, launch capacity is route-proportional and is
    # over 40x smaller than the old experts-by-global-M grid.
    long_tokens, long_top_k = 8192, 8
    long_block_m = _w4a16_block_m(long_tokens,
                                   long_tokens * long_top_k,
                                   num_experts,
                                   prefer_small_tiles)
    origin_blocks = num_experts * (
        (long_tokens + long_block_m - 1) // long_block_m)
    compact_capacity = (
        (long_tokens * long_top_k + long_block_m - 1) // long_block_m +
        num_experts)
    assert origin_blocks >= 40 * compact_capacity
    assert _should_use_compact_w4a16(long_tokens, long_tokens * long_top_k,
                                     num_experts)
    assert not _should_use_compact_w4a16(1, long_top_k, num_experts)
    expected_prefill_block_m = 16 if prefer_small_tiles else 32
    assert _w4a16_block_m(256, 256 * long_top_k, num_experts,
                          prefer_small_tiles) == expected_prefill_block_m
    assert long_block_m == 32
    assert _should_use_route_w4a16(64, num_experts)
    assert not _should_use_route_w4a16(65, num_experts)
    assert not _should_use_route_w4a16(1, 1)


def test_direct_packed_w4a16_rejects_incompatible_layout():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_kernel_launcher

    device = torch.device('cuda')
    hidden_states = torch.empty(1, 64, dtype=torch.bfloat16, device=device)
    packed = torch.empty(1, 64, 8, dtype=torch.int32, device=device)
    scales = torch.empty(1, 64, 2, dtype=torch.float16, device=device)
    output = torch.empty(1, 64, dtype=torch.bfloat16, device=device)
    route_meta = torch.zeros(1, dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match='bfloat16 scales'):
        fused_moe_w4a16_kernel_launcher(hidden_states, packed, scales, output,
                                        route_meta, route_meta, route_meta)


def test_route_major_w4a16_rejects_partial_topk_row():
    from lmdeploy.pytorch.kernels.cuda.compressed_tensors_w4a16 import fused_moe_w4a16_route_launcher

    device = torch.device('cuda')
    hidden_states = torch.empty(3,
                                64,
                                dtype=torch.bfloat16,
                                device=device)
    packed = torch.empty(128,
                         64,
                         8,
                         dtype=torch.int32,
                         device=device)
    scales = torch.empty(128,
                         64,
                         2,
                         dtype=torch.bfloat16,
                         device=device)
    output = torch.empty(5, 64, dtype=torch.bfloat16, device=device)
    route_ids = torch.arange(5, dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match='must be divisible'):
        fused_moe_w4a16_route_launcher(
            hidden_states,
            packed,
            scales,
            output,
            route_ids,
            top_k=2,
            reindex_a=True,
        )


def test_cuda_backend_exposes_direct_packed_w4a16_builder():
    from lmdeploy.pytorch.backends.base import OpType
    from lmdeploy.pytorch.backends.cuda.op_backend import CudaOpsBackend
    from lmdeploy.pytorch.backends.moe import FusedMoEW4A16Impl

    builder = CudaOpsBackend.get_layer_impl_builder(OpType.FusedMoEW4A16)
    impl = builder.build(top_k=2, num_experts=4, num_bits=4, group_size=32)

    assert isinstance(impl, FusedMoEW4A16Impl)
    assert impl.top_k == 2
    assert impl.num_experts == 4

    with pytest.raises(ValueError, match='INT4 group-size 32'):
        builder.build(top_k=2, num_experts=4, num_bits=4, group_size=64)
