import pytest
import torch
import torch.nn.functional as F

from lmdeploy.pytorch.kernels.fused_moe import (fused_moe,
                                                fused_moe_kernel_launcher)


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
        yield 4

    @pytest.fixture
    def top_k(self):
        yield 2

    @pytest.fixture
    def A(self, M, K, device, dtype):
        ret = torch.rand(M, K, device=device, dtype=dtype)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def B(self, num_experts, N, K, device, dtype):
        ret = torch.rand(num_experts, N, K, device=device, dtype=dtype)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def router_weights(self, M, num_experts, device, dtype):
        yield torch.rand(M, num_experts, device=device, dtype=dtype)

    @pytest.fixture
    def topk_weights(self, router_weights, top_k):
        yield router_weights.topk(top_k, dim=-1)

    @pytest.fixture
    def weights(self, topk_weights):
        yield topk_weights[0]

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
    def enable_weights(self):
        yield True

    @pytest.fixture
    def gt(self, A, B, top_k, topk_idx, enable_weights, weights):
        M = A.size(0)
        N = B.size(1)
        E = B.size(0)
        C = B.new_empty(M, top_k, N)
        for eid in range(E):
            EB = B[eid].t()
            token_idx, k_idx = torch.where(topk_idx == eid)
            if len(token_idx) == 0:
                continue
            EC = A[token_idx] @ EB
            C[token_idx, k_idx] = EC
        if enable_weights:
            C = C * weights[..., None]
        yield C.flatten(0, 1)

    @torch.inference_mode()
    def test_launcher(self, A, B, sorted_idx, exp_start, exp_end, weights,
                      enable_weights, top_k, M, gt):
        N = B.size(1)
        C = B.new_empty(M * top_k, N)

        fused_moe_kernel_launcher(
            A,
            B,
            C,
            sorted_idx,
            exp_start,
            exp_end,
            weights,
            enable_weights=enable_weights,
            top_k=top_k,
            num_tokens=M,
        )
        torch.testing.assert_close(C, gt)


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
        yield 4

    @pytest.fixture
    def top_k(self):
        yield 2

    @pytest.fixture
    def renormalize(self):
        yield True

    @pytest.fixture
    def hidden_states(self, seq_len, in_size, dtype, device):
        ret = torch.rand(seq_len, in_size, dtype=dtype, device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w1(self, num_experts, hidden_size, in_size, dtype, device):
        ret = torch.rand(num_experts,
                         hidden_size,
                         in_size,
                         dtype=dtype,
                         device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def w2(self, num_experts, out_size, hidden_size, dtype, device):
        ret = torch.rand(num_experts,
                         out_size,
                         hidden_size // 2,
                         dtype=dtype,
                         device=device)
        yield (ret - 0.5) / 2

    @pytest.fixture
    def router_logits(self, seq_len, num_experts, dtype, device):
        yield torch.rand(seq_len, num_experts, dtype=dtype, device=device)

    @pytest.fixture
    def topk_logits(self, router_logits, top_k):
        routing_weights = torch.softmax(router_logits,
                                        dim=-1,
                                        dtype=torch.float32)
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
            topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                           keepdim=True)

        seq_len = hidden_states.size(0)
        out_size = w2.size(1)
        output = hidden_states.new_zeros(seq_len, out_size)
        num_experts = w1.size(0)
        for eid in range(num_experts):
            token_idx, k_idx = torch.where(topk_idx == eid)
            gate_proj, up_proj = w1[eid].chunk(2, dim=0)
            down_proj = w2[eid]
            tmp_out = _mlp_forward(hidden_states[token_idx], gate_proj,
                                   up_proj, down_proj)
            tmp_out = tmp_out * topk_weights[token_idx, k_idx, None]
            output.index_add_(0, token_idx, tmp_out.to(output.dtype))
        yield output

    @torch.inference_mode()
    def test_fused_moe(self, hidden_states, w1, w2, topk_weights, topk_idx,
                       top_k, renormalize, gt):
        output = fused_moe(hidden_states,
                           w1,
                           w2,
                           topk_weights,
                           topk_idx,
                           topk=top_k,
                           renormalize=renormalize)
        torch.testing.assert_close(output, gt, atol=1e-3, rtol=1e-3)
