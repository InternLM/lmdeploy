from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize('hidden_size', [6144, 7168])
@pytest.mark.parametrize('num_tokens', [1, 2, 4, 8, 16, 24, 32])
@pytest.mark.parametrize('input_dtype', [torch.bfloat16, torch.float32])
def test_fp32_router_gemm(hidden_size, num_tokens, input_dtype):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('The optimized router kernel requires SM90 or newer.')

    from lmdeploy.pytorch.kernels.cuda.fp32_router_gemm import _is_supported, fp32_router_gemm

    torch.manual_seed(num_tokens)
    hidden_states = torch.randn(num_tokens, hidden_size, dtype=input_dtype, device='cuda')
    weight = torch.randn(256, hidden_size, dtype=torch.float32, device='cuda')
    assert _is_supported(hidden_states, weight) == (input_dtype == torch.bfloat16 or num_tokens <= 8)
    output = fp32_router_gemm(hidden_states, weight)
    reference = F.linear(hidden_states.float(), weight)

    torch.testing.assert_close(output, reference, atol=2e-4, rtol=0)
    output_ids = output.topk(8, dim=-1).indices.sort(dim=-1).values
    reference_ids = reference.topk(8, dim=-1).indices.sort(dim=-1).values
    torch.testing.assert_close(output_ids, reference_ids, atol=0, rtol=0)


def test_fp32_router_gemm_fallback():
    from lmdeploy.pytorch.kernels.cuda.fp32_router_gemm import _is_supported, fp32_router_gemm

    hidden_states = torch.randn(33, 6144, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(256, 6144, dtype=torch.float32, device='cuda')
    output = fp32_router_gemm(hidden_states, weight)
    reference = F.linear(hidden_states.float(), weight)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)

    fp32_hidden_states = hidden_states[:16].float()
    assert not _is_supported(fp32_hidden_states, weight)
    output = fp32_router_gemm(fp32_hidden_states, weight)
    reference = F.linear(fp32_hidden_states, weight)
    torch.testing.assert_close(output, reference, atol=0, rtol=0)


def test_fp32_router_gemm_cudagraph():
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('The optimized router kernel requires SM90 or newer.')

    from lmdeploy.pytorch.kernels.cuda.fp32_router_gemm import fp32_router_gemm

    hidden_states = torch.randn(16, 6144, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn(256, 6144, dtype=torch.float32, device='cuda')
    for _ in range(3):
        fp32_router_gemm(hidden_states, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fp32_router_gemm(hidden_states, weight)
    reference = F.linear(hidden_states.float(), weight)
    graph.replay()
    torch.testing.assert_close(output, reference, atol=2e-4, rtol=0)
    output_ids = output.topk(8, dim=-1).indices.sort(dim=-1).values
    reference_ids = reference.topk(8, dim=-1).indices.sort(dim=-1).values
    torch.testing.assert_close(output_ids, reference_ids, atol=0, rtol=0)


@pytest.mark.parametrize(
    'architecture,hidden_size,n_group,topk_group',
    [
        ('GlmMoeDsaForCausalLM', 6144, 1, 1),
        ('DeepseekV32ForCausalLM', 7168, 8, 4),
        ('DeepseekV3ForCausalLM', 7168, 8, 4),
    ],
)
def test_moe_gate_model_contract(architecture, hidden_size, n_group, topk_group):
    from lmdeploy.pytorch.models.deepseek_v2 import MoEGate
    from lmdeploy.pytorch.models.module_map import MODULE_MAP

    assert architecture in MODULE_MAP
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
    )
    torch.manual_seed(hidden_size + n_group)
    gate = MoEGate(config, dtype=torch.bfloat16, device='cuda')
    torch.nn.init.normal_(gate.weight)
    torch.nn.init.uniform_(gate.e_score_correction_bias, -0.05, 0.05)
    hidden_states = torch.randn(16, hidden_size, dtype=torch.bfloat16, device='cuda')

    output_weights, output_ids = gate(hidden_states)
    reference_logits = F.linear(hidden_states.float(), gate.weight)
    reference_weights, reference_ids = gate.noaux_tc_router(reference_logits, gate.e_score_correction_bias)

    torch.testing.assert_close(output_ids, reference_ids, atol=0, rtol=0)
    torch.testing.assert_close(output_weights, reference_weights, atol=5e-7, rtol=0)
