# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize(
    'hidden_size,num_tokens,n_group,topk_group,model_type,router_dtype',
    [
        (6144, 96, 1, 1, 'glm_moe_dsa', torch.float32),
        (7168, 16, 8, 4, 'deepseek_v32', torch.bfloat16),
    ],
)
def test_moe_gate_model_contract(hidden_size, num_tokens, n_group, topk_group, model_type, router_dtype):
    from lmdeploy.pytorch.models.deepseek_v2 import MoEGate

    config = SimpleNamespace(
        num_experts_per_tok=8,
        n_routed_experts=256,
        routed_scaling_factor=2.5,
        scoring_func='sigmoid',
        topk_method='noaux_tc',
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=True,
        router_n_groups=-1,
        hidden_size=hidden_size,
        model_type=model_type,
    )
    torch.manual_seed(hidden_size + n_group)
    gate = MoEGate(config, dtype=torch.bfloat16, device='cuda')
    assert gate.weight.dtype == torch.bfloat16
    torch.nn.init.normal_(gate.weight)
    torch.nn.init.uniform_(gate.e_score_correction_bias, -0.05, 0.05)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device='cuda')

    router_logits = gate.router_gemm(hidden_states, gate.weight)
    assert router_logits.dtype == router_dtype
    output_weights, output_ids = gate(hidden_states)
    reference_logits = F.linear(hidden_states.to(router_dtype), gate.weight.to(router_dtype))
    reference_weights, reference_ids = gate.noaux_tc_router(reference_logits, gate.e_score_correction_bias)

    torch.testing.assert_close(output_ids, reference_ids, atol=0, rtol=0)
    # Pre-Hopper CUDA falls back to BF16 linear before casting GLM logits to FP32.
    atol = 2e-4 if router_dtype == torch.float32 else 5e-7
    torch.testing.assert_close(output_weights, reference_weights, atol=atol, rtol=0)


@pytest.mark.parametrize('num_experts,num_tokens,hidden_size', [
    (256, 1024, 4096),   # DeepSeek-V4 router
    (192, 1024, 4096),   # Hy3 router
])
def test_router_gemm_fp32_weight_uses_split_bf16(num_experts, num_tokens, hidden_size):
    """A BF16 activation against an FP32 gate goes through the split kernel.

    The native path would degrade to an FP32 FFMA GEMM; the split kernel keeps the tensor cores busy while still
    producing FP32 logits.
    """
    from lmdeploy.pytorch.backends.cuda.moe_router import CudaRouterGemmImpl

    torch.manual_seed(num_experts)
    impl = CudaRouterGemmImpl(out_dtype=torch.float32)
    hidden_states = torch.randn(num_tokens, hidden_size, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(num_experts, hidden_size, device='cuda', dtype=torch.float32)

    logits = impl.forward(hidden_states, weight)
    reference = (hidden_states.double() @ weight.double().t()).float()

    assert logits.shape == (num_tokens, num_experts)
    assert logits.dtype == torch.float32
    # The weight is reconstructed from two BF16 halves, so the budget follows
    # the ~2^-17 relative error of that split rather than an rtol on the value.
    atol = 8 * 2**-17 * reference.pow(2).mean().sqrt().item()
    torch.testing.assert_close(logits, reference, rtol=0.0, atol=atol)


@pytest.mark.parametrize('weight_dtype', [torch.bfloat16, torch.float16])
def test_router_gemm_keeps_existing_paths(weight_dtype):
    """Non-FP32 gates keep their original dispatch."""
    from lmdeploy.pytorch.backends.cuda.moe_router import CudaRouterGemmImpl

    torch.manual_seed(0)
    impl = CudaRouterGemmImpl(out_dtype=torch.float32)
    hidden_states = torch.randn(64, 512, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(32, 512, device='cuda', dtype=weight_dtype)

    logits = impl.forward(hidden_states, weight)

    assert logits.dtype == torch.float32
    reference = F.linear(hidden_states.to(weight_dtype), weight).to(torch.float32)
    torch.testing.assert_close(logits, reference, rtol=1e-2, atol=1e-2)
