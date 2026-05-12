import pytest
import torch

_FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
    -4.0, -6.0,
],
                           dtype=torch.float32)


def _make_A(M, K, group_size, out_dtype, device='cuda'):
    quant_A = torch.rand(M, K // group_size, group_size, dtype=torch.float32, device=device)
    quant_A = quant_A * 2 - 1
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
    quant_A *= scaling
    quant_A = quant_A.to(out_dtype).to(torch.float32)

    scale = torch.rand(M, K // group_size, dtype=torch.float32, device=device)
    scale /= fmax
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    return A, quant_A, scale


def _make_fp4_B(E, K, N, device='cuda', scale_exp_min=-5, scale_exp_max=-2):
    assert K % 32 == 0
    code = torch.randint(0, 16, (E, N, K), dtype=torch.uint8, device=device)
    low = code[..., 0::2]
    high = code[..., 1::2]
    packed = (low | (high << 4)).to(torch.uint8).view(torch.int8)

    scale_exp = torch.randint(scale_exp_min, scale_exp_max, (E, N, K // 32), device=device)
    scale = torch.pow(torch.full((), 2.0, device=device), scale_exp).to(torch.float8_e8m0fnu)
    dense = _FP4_TABLE.to(device)[code.long()] * scale.float().repeat_interleave(32, dim=-1)
    return dense.to(torch.float16), packed, scale


def _get_sorted_idx(topk_idx: torch.Tensor, num_experts: int):
    flatten_topk_idx = topk_idx.flatten()
    sorted_ids = flatten_topk_idx.argsort()
    exp_range = torch.arange(0, num_experts, device=topk_idx.device)
    exp_tok_cnt = (flatten_topk_idx[None, :] == exp_range[:, None]).sum(1)
    return sorted_ids, exp_tok_cnt


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestGroupedGemmContiguous:

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def num_experts(self):
        yield 4

    @pytest.fixture
    def M(self):
        yield 256

    @pytest.fixture
    def K(self):
        yield 1024

    @pytest.fixture
    def N(self):
        yield 512

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def quant_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def build_A(self, M, K, group_size, quant_dtype, device):
        yield _make_A(M, K, group_size=group_size, out_dtype=quant_dtype, device=device)

    @pytest.fixture
    def A_bf16(self, build_A):
        yield build_A[0].to(torch.bfloat16)

    @pytest.fixture
    def A_quant(self, build_A):
        yield build_A[1]

    @pytest.fixture
    def A_scale(self, build_A):
        yield build_A[2]

    @pytest.fixture
    def build_B(self, num_experts, N, K, device):
        yield _make_fp4_B(num_experts, K, N, device=device)

    @pytest.fixture
    def B_dense(self, build_B):
        yield build_B[0]

    @pytest.fixture
    def B_packed(self, build_B):
        yield build_B[1]

    @pytest.fixture
    def B_scale(self, build_B):
        yield build_B[2]

    @pytest.fixture
    def grouped_layout(self, M, num_experts, device):
        """Create a sorted grouped_layout (contiguous per expert)."""
        # Round-robin: each expert gets M//E tokens.
        per_expert = M // num_experts
        layout = torch.zeros(M, dtype=torch.int32, device=device)
        for e in range(num_experts):
            layout[e * per_expert:(e + 1) * per_expert] = e
        yield layout

    @torch.inference_mode()
    def test_contiguous_vs_launcher(self, A_quant, A_scale, B_packed, B_scale, grouped_layout, M, N, device):
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_fused_moe import fused_moe_v4_fp4_kernel_launcher
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous

        # Run through the new wrapper.
        out_wrapper = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (A_quant, A_scale),
            (B_packed, B_scale),
            out_wrapper,
            grouped_layout,
        )

        # Run through the raw launcher directly (same code path, manual metadata).
        num_experts = B_packed.size(0)
        expert_counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
        valid = grouped_layout >= 0
        expert_counts.scatter_add_(
            0, grouped_layout[valid].long(),
            torch.ones(valid.sum().item(), dtype=torch.int64, device=device))
        exp_end = expert_counts.cumsum(0)
        exp_start = exp_end - expert_counts
        sorted_idx = torch.arange(M, device=device, dtype=torch.int64)

        out_direct = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        fused_moe_v4_fp4_kernel_launcher(
            A_quant, A_scale, B_packed, B_scale, out_direct,
            sorted_idx=sorted_idx, exp_start=exp_start, exp_end=exp_end,
            top_k=1, num_tokens=M, expert_offset=0,
            reindex_a=False, reindex_c=False,
        )

        torch.testing.assert_close(out_wrapper, out_direct, atol=0.0, rtol=0.0)

    @torch.inference_mode()
    def test_contiguous_vs_bf16_reference(self, A_bf16, A_quant, A_scale, B_dense, B_packed, B_scale,
                                          grouped_layout, M, N, device):
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import m_grouped_fp8_fp4_gemm_nt_contiguous

        out = torch.empty(M, N, dtype=torch.bfloat16, device=device)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (A_quant, A_scale),
            (B_packed, B_scale),
            out,
            grouped_layout,
        )

        # BF16 reference: per-expert matmul.
        num_experts = B_packed.size(0)
        per_expert = M // num_experts
        ref = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
        for e in range(num_experts):
            start = e * per_expert
            end = start + per_expert
            ref[start:end] = A_bf16[start:end] @ B_dense[e].to(torch.bfloat16).T

        # Normalize before comparison (FP4 quantization noise).
        out_max = out.abs().max()
        ref_max = ref.abs().max()
        assert (out_max - ref_max).abs() / ref_max < 0.1
        norm_out = out / out_max
        norm_ref = ref / ref_max
        torch.testing.assert_close(norm_out, norm_ref, atol=0.1, rtol=1e-2)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestGroupedGemmMasked:

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def num_experts(self):
        yield 4

    @pytest.fixture
    def max_m(self):
        yield 64

    @pytest.fixture
    def K(self):
        yield 1024

    @pytest.fixture
    def N(self):
        yield 512

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def quant_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def masked_m(self, num_experts, max_m, device):
        """Each expert gets a different number of valid tokens."""
        m = torch.randint(1, max_m + 1, (num_experts,), dtype=torch.int, device=device)
        yield m

    @pytest.fixture
    def build_A(self, num_experts, max_m, K, group_size, quant_dtype, device):
        """Create [E, max_m, K] FP8 input with per-token scales."""
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        A_dense = torch.randn(num_experts, max_m, K, dtype=torch.bfloat16, device=device)
        A_flat = A_dense.reshape(-1, K)
        A_quant_flat, A_scale_flat = quant_fp8(A_flat, group_size, dtype=quant_dtype, scale_fmt='ue8m0')
        A_quant = A_quant_flat.reshape(num_experts, max_m, K)
        A_scale = A_scale_flat.reshape(num_experts, max_m, K // group_size)
        yield A_dense, A_quant, A_scale

    @pytest.fixture
    def A_dense(self, build_A):
        yield build_A[0]

    @pytest.fixture
    def A_quant(self, build_A):
        yield build_A[1]

    @pytest.fixture
    def A_scale(self, build_A):
        yield build_A[2]

    @pytest.fixture
    def build_B(self, num_experts, N, K, device):
        yield _make_fp4_B(num_experts, K, N, device=device)

    @pytest.fixture
    def B_dense(self, build_B):
        yield build_B[0]

    @pytest.fixture
    def B_packed(self, build_B):
        yield build_B[1]

    @pytest.fixture
    def B_scale(self, build_B):
        yield build_B[2]

    @torch.inference_mode()
    def test_masked_vs_launcher(self, A_quant, A_scale, B_packed, B_scale, masked_m, max_m, device):
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_fused_moe import fused_moe_v4_fp4_kernel_launcher
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import m_grouped_fp8_fp4_gemm_nt_masked

        num_experts = B_packed.size(0)
        N = B_packed.size(1)

        # Run through the new wrapper.
        out_wrapper = torch.empty((num_experts, max_m, N), dtype=torch.bfloat16, device=device)
        m_grouped_fp8_fp4_gemm_nt_masked(
            (A_quant, A_scale),
            (B_packed, B_scale),
            out_wrapper,
            masked_m,
            expected_m=max_m,
        )

        # Run through the raw launcher directly.
        A_flat = A_quant.reshape(num_experts * max_m, -1)
        A_scale_flat = A_scale.reshape(num_experts * max_m, -1)
        out_flat = torch.empty(num_experts * max_m, N, dtype=torch.bfloat16, device=device)
        exp_start = torch.arange(num_experts, device=device, dtype=torch.int64) * max_m
        exp_end = exp_start + masked_m.to(torch.int64)
        sorted_idx = torch.arange(num_experts * max_m, device=device, dtype=torch.int64)

        fused_moe_v4_fp4_kernel_launcher(
            A_flat, A_scale_flat, B_packed, B_scale, out_flat,
            sorted_idx=sorted_idx, exp_start=exp_start, exp_end=exp_end,
            top_k=1, num_tokens=num_experts * max_m, expert_offset=0,
            reindex_a=False, reindex_c=False,
        )
        out_direct = out_flat.reshape(num_experts, max_m, N)

        # Only compare valid rows (padding rows contain garbage from the kernel).
        for e in range(num_experts):
            m = masked_m[e].item()
            if m > 0:
                torch.testing.assert_close(out_wrapper[e, :m], out_direct[e, :m], atol=0.0, rtol=0.0)

    @torch.inference_mode()
    def test_masked_vs_bf16_reference(self, A_dense, A_quant, A_scale, B_dense, B_packed, B_scale,
                                      masked_m, max_m, device):
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import m_grouped_fp8_fp4_gemm_nt_masked

        num_experts = B_packed.size(0)
        N = B_packed.size(1)

        out = torch.empty((num_experts, max_m, N), dtype=torch.bfloat16, device=device)
        m_grouped_fp8_fp4_gemm_nt_masked(
            (A_quant, A_scale),
            (B_packed, B_scale),
            out,
            masked_m,
            expected_m=max_m,
        )

        # BF16 reference: per-expert matmul (only valid rows).
        ref = torch.zeros(num_experts, max_m, N, dtype=torch.bfloat16, device=device)
        for e in range(num_experts):
            m = masked_m[e].item()
            if m > 0:
                ref[e, :m] = A_dense[e, :m] @ B_dense[e].to(torch.bfloat16).T

        # Compare only valid rows.
        for e in range(num_experts):
            m = masked_m[e].item()
            if m > 0:
                out_max = out[e, :m].abs().max()
                ref_max = ref[e, :m].abs().max()
                if ref_max > 0:
                    assert (out_max - ref_max).abs() / ref_max < 0.15
                    norm_out = out[e, :m] / out_max
                    norm_ref = ref[e, :m] / ref_max
                    torch.testing.assert_close(norm_out, norm_ref, atol=0.1, rtol=1e-2)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestEPNormal:
    """Test fused_moe_v4_fp4_ep_normal (prefill EP path)."""

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def num_experts(self):
        yield 4

    @pytest.fixture
    def num_tokens(self):
        yield 256

    @pytest.fixture
    def hidden_dim(self):
        yield 1024

    @pytest.fixture
    def ffn_dim(self):
        yield 512

    @pytest.fixture
    def top_k(self):
        yield 2

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def router_logits(self, num_tokens, num_experts, device):
        yield torch.rand(num_tokens, num_experts, dtype=torch.float32, device=device)

    @pytest.fixture
    def topk_info(self, router_logits, top_k):
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        values, indices = torch.topk(routing_weights, top_k, dim=-1)
        # Renormalize.
        values = values / values.sum(dim=-1, keepdim=True)
        yield values, indices

    @pytest.fixture
    def topk_weights(self, topk_info):
        yield topk_info[0]

    @pytest.fixture
    def topk_ids(self, topk_info):
        yield topk_info[1]

    @pytest.fixture
    def permuted_data(self, num_experts, hidden_dim, topk_ids, device):
        """Simulate what DeepEPTokenDispatcher.dispatch returns: tokens
        permuted by expert (contiguous per expert), plus
        recv_tokens_per_expert."""
        flat_ids = topk_ids.flatten()
        counts = torch.zeros(num_experts, dtype=torch.int64, device=device)
        for e in range(num_experts):
            counts[e] = (flat_ids == e).sum().item()
        all_tokens = counts.sum().item()

        # Build permuted recv_x: expert 0 tokens first, then expert 1, etc.
        recv_x = torch.randn(all_tokens, hidden_dim, dtype=torch.bfloat16, device=device)
        yield recv_x, counts

    @pytest.fixture
    def build_w1(self, num_experts, hidden_dim, ffn_dim, device):
        yield _make_fp4_B(num_experts, hidden_dim, ffn_dim * 2, device=device)

    @pytest.fixture
    def w1_packed(self, build_w1):
        yield build_w1[1]

    @pytest.fixture
    def w1_scale(self, build_w1):
        yield build_w1[2]

    @pytest.fixture
    def build_w2(self, num_experts, ffn_dim, hidden_dim, device):
        yield _make_fp4_B(num_experts, ffn_dim, hidden_dim, device=device)

    @pytest.fixture
    def w2_packed(self, build_w2):
        yield build_w2[1]

    @pytest.fixture
    def w2_scale(self, build_w2):
        yield build_w2[2]

    @torch.inference_mode()
    def test_ep_normal_output_shape_and_range(self, permuted_data, topk_ids, topk_weights,
                                               w1_packed, w1_scale,
                                               w2_packed, w2_scale,
                                               num_experts, hidden_dim, ffn_dim, top_k, device):
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import fused_moe_v4_fp4_ep_normal

        recv_x, recv_tokens_per_expert = permuted_data

        out = fused_moe_v4_fp4_ep_normal(
            recv_x, topk_ids, topk_weights, recv_tokens_per_expert,
            w1_packed, w1_scale, w2_packed, w2_scale,
            num_local_experts=num_experts,
            expert_offset=0,
        )

        assert out.shape == (recv_x.size(0), hidden_dim)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all(), 'EP normal output contains non-finite values'

    @torch.inference_mode()
    def test_ep_normal_with_expert_offset(self, w1_packed, w1_scale,
                                           w2_packed, w2_scale,
                                           num_experts, top_k, device):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import (
            fused_moe_v4_fp4_ep_normal,
            m_grouped_fp8_fp4_gemm_nt_contiguous,
        )

        # Simulate EP rank 1 of 2: experts [2, 3] are local.
        # recv_x is already permuted by expert (dispatcher did this).
        half = num_experts // 2
        expert_offset = half
        tokens_per_local_expert = 64
        hidden_dim = 1024

        # Permuted: first 64 rows for local expert 0 (global 2), next 64 for local expert 1 (global 3).
        recv_x = torch.randn(tokens_per_local_expert * half, hidden_dim, dtype=torch.bfloat16, device=device)
        recv_tokens_per_expert = torch.full((half,), tokens_per_local_expert, dtype=torch.int64, device=device)

        # Dummy topk_idx/topk_weights (not used by the new implementation).
        topk_ids = torch.zeros(tokens_per_local_expert * half, top_k, dtype=torch.int64, device=device)
        topk_weights = torch.rand(tokens_per_local_expert * half, top_k, dtype=torch.float32, device=device)

        local_w1_packed = w1_packed[expert_offset:]
        local_w1_scale = w1_scale[expert_offset:]
        local_w2_packed = w2_packed[expert_offset:]
        local_w2_scale = w2_scale[expert_offset:]

        out = fused_moe_v4_fp4_ep_normal(
            recv_x, topk_ids, topk_weights, recv_tokens_per_expert,
            local_w1_packed, local_w1_scale, local_w2_packed, local_w2_scale,
            num_local_experts=half,
            expert_offset=expert_offset,
        )

        # Reference: manually run the same pipeline (quant -> grouped GEMM -> swiglu -> grouped GEMM).
        input_quant, input_scale = quant_fp8(recv_x, 128, dtype=torch.float8_e4m3fn, scale_fmt='ue8m0')
        m_indices = recv_x.new_empty(recv_x.size(0), dtype=torch.int32)
        for e in range(half):
            m_indices[e * tokens_per_local_expert:(e + 1) * tokens_per_local_expert] = e

        gateup_output = recv_x.new_empty((recv_x.size(0), local_w1_packed.size(1)), dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (input_quant, input_scale), (local_w1_packed, local_w1_scale),
            gateup_output, m_indices)

        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import silu_and_mul_moe_ep_v4
        _, act_quant, act_scale = silu_and_mul_moe_ep_v4(gateup_output, group_size=128)

        ref = recv_x.new_empty((recv_x.size(0), local_w2_packed.size(1)), dtype=torch.bfloat16)
        m_grouped_fp8_fp4_gemm_nt_contiguous(
            (act_quant, act_scale), (local_w2_packed, local_w2_scale),
            ref, m_indices)

        torch.testing.assert_close(out, ref, atol=0.0, rtol=0.0)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestEPLowLatency:
    """Test fused_moe_v4_fp4_ep_low_latency (decode EP path)."""

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def num_experts(self):
        yield 4

    @pytest.fixture
    def max_m(self):
        yield 32

    @pytest.fixture
    def hidden_dim(self):
        yield 1024

    @pytest.fixture
    def ffn_dim(self):
        yield 512

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def masked_m(self, num_experts, max_m, device):
        m = torch.randint(1, max_m + 1, (num_experts,), dtype=torch.int, device=device)
        yield m

    @pytest.fixture
    def build_A(self, num_experts, max_m, hidden_dim, group_size, device):
        from lmdeploy.pytorch.kernels.cuda.blocked_gemm_fp8 import quant_fp8
        A_dense = torch.randn(num_experts, max_m, hidden_dim, dtype=torch.bfloat16, device=device)
        A_flat = A_dense.reshape(-1, hidden_dim)
        A_quant_flat, A_scale_flat = quant_fp8(A_flat, group_size, dtype=torch.float8_e4m3fn, scale_fmt='ue8m0')
        A_quant = A_quant_flat.reshape(num_experts, max_m, hidden_dim)
        A_scale = A_scale_flat.reshape(num_experts, max_m, hidden_dim // group_size)
        yield A_dense, A_quant, A_scale

    @pytest.fixture
    def A_quant(self, build_A):
        yield build_A[1]

    @pytest.fixture
    def A_scale(self, build_A):
        yield build_A[2]

    @pytest.fixture
    def build_w1(self, num_experts, hidden_dim, ffn_dim, device):
        yield _make_fp4_B(num_experts, hidden_dim, ffn_dim * 2, device=device)

    @pytest.fixture
    def w1_packed(self, build_w1):
        yield build_w1[1]

    @pytest.fixture
    def w1_scale(self, build_w1):
        yield build_w1[2]

    @pytest.fixture
    def build_w2(self, num_experts, ffn_dim, hidden_dim, device):
        yield _make_fp4_B(num_experts, ffn_dim, hidden_dim, device=device)

    @pytest.fixture
    def w2_packed(self, build_w2):
        yield build_w2[1]

    @pytest.fixture
    def w2_scale(self, build_w2):
        yield build_w2[2]

    @torch.inference_mode()
    def test_ep_low_latency_vs_grouped_gemm(self, A_quant, A_scale,
                                              w1_packed, w1_scale,
                                              w2_packed, w2_scale,
                                              masked_m, max_m, num_experts,
                                              hidden_dim, ffn_dim, device):
        from lmdeploy.pytorch.kernels.cuda.v4_fp4_grouped_gemm import (
            fused_moe_v4_fp4_ep_low_latency,
            m_grouped_fp8_fp4_gemm_nt_masked,
            silu_and_mul_moe_ep_v4,
        )

        out = fused_moe_v4_fp4_ep_low_latency(
            (A_quant, A_scale),
            masked_m,
            expected_m=max_m,
            w1=w1_packed, w1_scale=w1_scale,
            w2=w2_packed, w2_scale=w2_scale,
        )

        # Reference: run the same grouped GEMM pipeline step by step.
        gateup_output = torch.empty((num_experts, max_m, ffn_dim * 2), dtype=torch.bfloat16, device=device)
        m_grouped_fp8_fp4_gemm_nt_masked(
            (A_quant, A_scale),
            (w1_packed, w1_scale),
            gateup_output,
            masked_m,
            expected_m=max_m,
        )

        _, act_quant, act_scale = silu_and_mul_moe_ep_v4(
            gateup_output,
            group_size=128,
        )

        ref = torch.empty((num_experts, max_m, hidden_dim), dtype=torch.bfloat16, device=device)
        m_grouped_fp8_fp4_gemm_nt_masked(
            (act_quant, act_scale),
            (w2_packed, w2_scale),
            ref,
            masked_m,
            expected_m=max_m,
        )

        # Compare only valid rows (same code path, should be exact match).
        for e in range(num_experts):
            m = masked_m[e].item()
            if m > 0:
                torch.testing.assert_close(out[e, :m], ref[e, :m], atol=0.0, rtol=0.0)
