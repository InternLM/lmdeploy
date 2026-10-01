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
