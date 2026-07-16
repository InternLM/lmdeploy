import pytest
import torch
import torch.nn.functional as F


@pytest.mark.parametrize(('num_routes', 'block_m'), [(640, 16), (512 * 24, 32), (512 * 40, 64)])
def test_origin_blocked_fp8_small_m_configs_use_average_routes(num_routes, block_m):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=64,
                                                               num_routes=num_routes,
                                                               num_experts=512,
                                                               local_experts=512)
    assert gate_config == dict(block_m=max(64, block_m), block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=block_m, block_n=128, num_warps=4, num_stages=3)


def test_origin_blocked_fp8_large_m_uses_bm64_down_config():
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=65,
                                                               num_routes=650,
                                                               num_experts=512,
                                                               local_experts=512)
    assert gate_config == dict(block_m=128, block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=64, block_n=128, num_warps=4, num_stages=3)


def test_origin_blocked_fp8_large_m_high_avg_routes_uses_default_down_config():
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=2048,
                                                               num_routes=512 * 40,
                                                               num_experts=512,
                                                               local_experts=512)
    expected = dict(block_m=128, block_n=128, num_warps=4, num_stages=3)
    assert gate_config == expected
    assert down_config == expected


def test_origin_blocked_fp8_uses_average_routes_for_256_experts():
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=64,
                                                               num_routes=256 * 2,
                                                               num_experts=256,
                                                               local_experts=256)
    assert gate_config == dict(block_m=64, block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=16, block_n=128, num_warps=4, num_stages=3)


def test_origin_blocked_fp8_large_m_uses_average_routes_for_256_experts():
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _origin_blocked_fp8_moe_configs

    gate_config, down_config = _origin_blocked_fp8_moe_configs(num_tokens=512,
                                                               num_routes=256 * 16,
                                                               num_experts=256,
                                                               local_experts=256)
    assert gate_config == dict(block_m=128, block_n=128, num_warps=4, num_stages=3)
    assert down_config == dict(block_m=64, block_n=128, num_warps=4, num_stages=3)


@pytest.mark.parametrize(('num_routes', 'block_m'), [(640, 64), (512 * 40, 64), (512 * 64, 128),
                                                     (512 * 160, 128)])
def test_compact_blocked_fp8_configs_use_average_routes(num_routes, block_m):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _compact_blocked_fp8_moe_config

    assert _compact_blocked_fp8_moe_config(num_routes, num_experts=512) == dict(block_m=block_m,
                                                                                block_n=128,
                                                                                num_warps=4,
                                                                                num_stages=3)


@pytest.mark.parametrize(('num_tokens', 'num_routes', 'origin_ctas', 'compact_ctas'), [
    (65, 650, 512 * 2 * 32, 512 * 1 * 32),
    (1024, 512 * 20, 512 * 16 * 32, 512 * 1 * 32),
    (4096, 512 * 80, 512 * 32 * 32, 512 * 1 * 32),
    (8192, 512 * 160, 512 * 64 * 32, 512 * 2 * 32),
])
def test_blocked_fp8_moe_cta_estimates(num_tokens, num_routes, origin_ctas, compact_ctas):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import _blocked_fp8_moe_cta_estimates

    assert _blocked_fp8_moe_cta_estimates(num_tokens,
                                          num_routes,
                                          num_experts=512,
                                          local_experts=512,
                                          out_features=4096) == (origin_ctas, compact_ctas)


@pytest.mark.parametrize(('num_tokens', 'num_routes', 'num_experts', 'local_experts', 'out_features', 'expected'), [
    (64, 640, 512, 512, 4096, False),
    (511, 512 * 10, 512, 512, 4096, False),
    (512, 512 * 10, 512, 512, 4096, True),
    (1024, 512 * 20, 512, 512, 4096, True),
    (1024, 512 * 20, 512, 128, 4096, False),
    (4096, 512 * 80, 512, 256, 128, True),
    (2048, 256 * 64, 256, 256, 2048, False),
    (4096, 256 * 128, 256, 256, 2048, True),
    (512, 256 * 16, 256, 256, 7168, False),
    (1024, 256 * 32, 256, 256, 7168, True),
])
def test_compact_blocked_fp8_down_policy_is_prefill_and_cta_gated(num_tokens, num_routes, num_experts, local_experts,
                                                                  out_features, expected):
    from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import (
        _should_use_compact_blocked_fp8_moe_down_by_shape,
    )

    assert _should_use_compact_blocked_fp8_moe_down_by_shape(num_tokens,
                                                             num_routes,
                                                             num_experts=num_experts,
                                                             local_experts=local_experts,
                                                             out_features=out_features) is expected


def _get_sorted_idx(topk_idx: torch.Tensor, num_experts: int):
    flatten_topk_idx = topk_idx.flatten()
    sorted_ids = flatten_topk_idx.argsort()
    exp_range = torch.arange(0, num_experts, device=topk_idx.device)
    exp_tok_cnt = (flatten_topk_idx[None, :] == exp_range[:, None]).sum(1)
    return sorted_ids, exp_tok_cnt


class TestFusedMoEKernelLauncher:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def N(self):
        yield 128

    @pytest.fixture
    def K(self):
        yield 64

    @pytest.fixture
    def M(self):
        yield 256

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def top_k(self):
        yield 6

    @pytest.fixture
    def A(self, M, K, device, dtype):
        ret = torch.rand(M, K, device=device, dtype=dtype)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def B(self, num_experts, N, K, device, dtype):
        ret = torch.rand(num_experts, N, K, device=device, dtype=dtype)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def bias(self, num_experts, N, device, dtype):
        yield torch.rand(num_experts, N, device=device, dtype=dtype) - 0.5

    @pytest.fixture
    def router_weights(self, M, num_experts, device, dtype):
        yield torch.rand(M, num_experts, device=device, dtype=dtype)

    @pytest.fixture
    def topk_weights(self, router_weights, top_k):
        yield router_weights.topk(top_k, dim=-1)

    @pytest.fixture
    def topk_idx(self, topk_weights):
        yield topk_weights[1]

    @pytest.fixture
    def sort_and_cnt(self, topk_idx, num_experts):
        yield _get_sorted_idx(topk_idx, num_experts)

    @pytest.fixture
    def sorted_idx(self, sort_and_cnt):
        yield sort_and_cnt[0]

    @pytest.fixture
    def exp_tok_cnt(self, sort_and_cnt):
        yield sort_and_cnt[1]

    @pytest.fixture
    def exp_end(self, exp_tok_cnt):
        yield exp_tok_cnt.cumsum(0)

    @pytest.fixture
    def exp_start(self, exp_end, exp_tok_cnt):
        yield exp_end - exp_tok_cnt

    @pytest.fixture
    def gt(self, A, B, bias, top_k, topk_idx):
        M = A.size(0)
        N = B.size(1)
        E = B.size(0)
        C = B.new_empty(M, top_k, N)
        for eid in range(E):
            EB = B[eid].t()
            Ebias = bias[eid]
            token_idx, k_idx = torch.where(topk_idx == eid)
            if len(token_idx) == 0:
                continue
            EC = A[token_idx] @ EB + Ebias
            C[token_idx, k_idx] = EC
        yield C.flatten(0, 1)

    @torch.inference_mode()
    def test_launcher(self, A, B, bias, sorted_idx, exp_start, exp_end, top_k, M, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_moe import fused_moe_kernel_launcher
        N = B.size(1)
        C = B.new_empty(M * top_k, N)

        fused_moe_kernel_launcher(
            A,
            B,
            C,
            sorted_idx,
            exp_start,
            exp_end,
            bias=bias,
            top_k=top_k,
            num_tokens=M,
        )
        torch.testing.assert_close(C, gt, atol=1e-3, rtol=1e-3)


def _mlp_forward(hidden_states, gate_proj, up_proj, down_proj):
    gate = F.linear(hidden_states, gate_proj)
    up = F.linear(hidden_states, up_proj)
    return F.linear(F.silu(gate) * up, down_proj)


class TestFusedMoe:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def in_size(self):
        yield 128

    @pytest.fixture
    def seq_len(seq_len):
        yield 128

    @pytest.fixture
    def hidden_size(self):
        yield 256

    @pytest.fixture
    def out_size(self):
        yield 128

    @pytest.fixture
    def num_experts(self):
        yield 64

    @pytest.fixture
    def top_k(self):
        yield 6

    @pytest.fixture
    def renormalize(self):
        yield True

    @pytest.fixture
    def hidden_states(self, seq_len, in_size, dtype, device):
        ret = torch.rand(seq_len, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w1(self, num_experts, hidden_size, in_size, dtype, device):
        ret = torch.rand(num_experts, hidden_size, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w2(self, num_experts, out_size, hidden_size, dtype, device):
        ret = torch.rand(num_experts, out_size, hidden_size // 2, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def router_logits(self, seq_len, num_experts, dtype, device):
        yield torch.rand(seq_len, num_experts, dtype=dtype, device=device)

    @pytest.fixture
    def topk_logits(self, router_logits, top_k):
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        yield torch.topk(routing_weights, top_k, dim=-1)

    @pytest.fixture
    def topk_weights(self, topk_logits):
        yield topk_logits[0]

    @pytest.fixture
    def topk_idx(self, topk_logits):
        yield topk_logits[1]

    @pytest.fixture
    def gt(self, hidden_states, w1, w2, topk_weights, topk_idx, renormalize):
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        seq_len = hidden_states.size(0)
        out_size = w2.size(1)
        output = hidden_states.new_zeros(seq_len, out_size)
        num_experts = w1.size(0)
        for eid in range(num_experts):
            token_idx, k_idx = torch.where(topk_idx == eid)
            gate_proj, up_proj = w1[eid].chunk(2, dim=0)
            down_proj = w2[eid]
            tmp_out = _mlp_forward(hidden_states[token_idx], gate_proj, up_proj, down_proj)
            tmp_out = tmp_out * topk_weights[token_idx, k_idx, None]
            output.index_add_(0, token_idx, tmp_out.to(output.dtype))
        yield output

    @torch.inference_mode()
    def test_fused_moe(self, hidden_states, w1, w2, topk_weights, topk_idx, top_k, renormalize, gt):
        from lmdeploy.pytorch.kernels.cuda.fused_moe import fused_moe
        output = fused_moe(hidden_states, w1, w2, topk_weights, topk_idx, topk=top_k, renormalize=renormalize)
        torch.testing.assert_close(output, gt, atol=1e-3, rtol=1e-3)


class TestFusedMoeW8A8(TestFusedMoe):

    @pytest.fixture
    def quant_states(self, hidden_states):
        from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_token_quant_int8
        states_i8, states_scale = per_token_quant_int8(hidden_states, 1e-7)
        yield states_i8, states_scale

    def quant_weight(self, w):
        from lmdeploy.pytorch.kernels.cuda.w8a8_triton_kernels import per_channel_quant
        num_experts, num_outs, _ = w.shape
        w = w.flatten(0, -2)
        w_i8, w_scale = per_channel_quant(w, torch.int8)
        w_i8 = w_i8.view(num_experts, num_outs, -1)
        w_scale = w_scale.view(num_experts, num_outs, -1)
        return w_i8, w_scale

    @pytest.fixture
    def quant_w1(self, w1):
        w_i8, w_scale = self.quant_weight(w1)
        yield w_i8, w_scale

    @pytest.fixture
    def quant_w2(self, w2):
        w_i8, w_scale = self.quant_weight(w2)
        yield w_i8, w_scale

    @torch.inference_mode()
    def test_fused_moe(self, quant_states, quant_w1, quant_w2, topk_weights, topk_idx, top_k, renormalize, gt):
        from lmdeploy.pytorch.kernels.cuda.w8a8_fused_moe import fused_moe_w8a8
        state_i8, state_scale = quant_states
        w1_i8, w1_scale = quant_w1
        w2_i8, w2_scale = quant_w2

        output = fused_moe_w8a8(state_i8,
                                state_scale,
                                w1_i8,
                                w1_scale,
                                w2_i8,
                                w2_scale,
                                topk_weights=topk_weights,
                                topk_ids=topk_idx,
                                topk=top_k,
                                out_dtype=torch.float16,
                                renormalize=renormalize)
        torch.testing.assert_close(output, gt, atol=5e-3, rtol=1e-3)
