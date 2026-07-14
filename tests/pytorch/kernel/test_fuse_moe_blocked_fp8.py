import pytest
import torch


def _make_A(M, K, group_size, out_dtype, device='cuda'):
    quant_A = torch.rand(M, K // group_size, group_size, dtype=torch.float32, device=device)
    # -1 ~ 1
    quant_A = quant_A * 2 - 1
    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_A.abs().amax(-1, keepdim=True)
    quant_A *= scaling
    quant_A = quant_A.to(out_dtype).to(torch.float32)

    # create scale and A
    scale = torch.rand(M, K // group_size, dtype=torch.float32, device=device)
    scale /= fmax
    A = quant_A * scale[..., None]

    A = A.reshape(M, K)
    quant_A = quant_A.reshape(M, K).to(out_dtype)
    return A, quant_A, scale


def _make_B(E, K, N, group_size, out_dtype, device='cuda'):
    quant_B = torch.rand(E,
                         N // group_size,
                         group_size,
                         K // group_size,
                         group_size,
                         dtype=torch.float32,
                         device=device)
    quant_B = quant_B * 2 - 1

    # scaling abs max to fmax
    finfo = torch.finfo(out_dtype)
    fmax = finfo.max
    scaling = fmax / quant_B.abs().amax((2, 4), keepdim=True)
    quant_B *= scaling
    quant_B = quant_B.to(out_dtype).to(torch.float32)

    scale = torch.rand(E, N // group_size, 1, K // group_size, 1, dtype=torch.float32, device=device)
    scale /= fmax

    B = quant_B * scale

    B = B.reshape(E, N, K)
    quant_B = quant_B.reshape(E, N, K).to(out_dtype)
    scale = scale.reshape(E, N // group_size, K // group_size)
    bias = torch.rand(E, N, dtype=torch.float32, device=device) - 0.5
    return B, quant_B, scale, bias


def _get_sorted_idx(topk_idx: torch.Tensor, num_experts: int):
    flatten_topk_idx = topk_idx.flatten()
    sorted_ids = flatten_topk_idx.argsort()
    exp_range = torch.arange(0, num_experts, device=topk_idx.device)
    exp_tok_cnt = (flatten_topk_idx[None, :] == exp_range[:, None]).sum(1)
    return sorted_ids, exp_tok_cnt


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestFusedMoEFP8KernelLauncher:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def quant_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def N(self):
        yield 512

    @pytest.fixture
    def K(self):
        yield 1024

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
    def group_size(self):
        yield 128

    @pytest.fixture
    def build_A(self, M, K, group_size, quant_dtype, device):
        yield _make_A(M, K, group_size=group_size, out_dtype=quant_dtype, device=device)

    @pytest.fixture
    def A(self, build_A, dtype):
        yield build_A[0].to(dtype)

    @pytest.fixture
    def A_quant(self, build_A):
        yield build_A[1]

    @pytest.fixture
    def A_scale(self, build_A):
        yield build_A[2]

    @pytest.fixture
    def build_B(self, num_experts, N, K, group_size, quant_dtype, device):
        yield _make_B(num_experts, K, N, group_size=group_size, out_dtype=quant_dtype, device=device)

    @pytest.fixture
    def B(self, build_B, dtype):
        yield build_B[0].to(dtype)

    @pytest.fixture
    def B_quant(self, build_B):
        yield build_B[1]

    @pytest.fixture
    def B_scale(self, build_B):
        yield build_B[2]

    @pytest.fixture
    def bias(self, build_B, dtype):
        yield build_B[3].to(dtype)
        # yield None

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
    def gt(self, A, B, bias, top_k, sorted_idx, exp_start, exp_end, M):
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

        yield C

    @torch.inference_mode()
    def test_launcher(self, A_quant, A_scale, B, B_quant, B_scale, bias, sorted_idx, exp_start, exp_end, top_k, M, gt):
        from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8_kernel_launcher
        N = B.size(1)
        C = B.new_empty(M * top_k, N)
        fused_moe_blocked_fp8_kernel_launcher(
            A=A_quant,
            A_scale=A_scale,
            B=B_quant,
            B_scale=B_scale,
            C=C,
            sorted_idx=sorted_idx,
            exp_start=exp_start,
            exp_end=exp_end,
            bias=bias,
            top_k=top_k,
            num_tokens=M,
        )

        gt_max = gt.abs().max()
        C = C / gt_max
        gt = gt / gt_max
        torch.testing.assert_close(C, gt, atol=4e-3, rtol=1e-3)


@pytest.mark.skipif(torch.cuda.get_device_capability()[0] < 9, reason='require device with cc>=9.0')
class TestFusedMoeBlockedFP8:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def quant_dtype(self):
        yield torch.float8_e4m3fn

    @pytest.fixture
    def device(self):
        yield torch.device('cuda')

    @pytest.fixture
    def in_size(self):
        yield 512

    @pytest.fixture
    def seq_len(seq_len):
        yield 128

    @pytest.fixture
    def hidden_size(self):
        yield 2048

    @pytest.fixture
    def out_size(self):
        yield 1024

    @pytest.fixture
    def num_experts(self):
        yield 4

    @pytest.fixture
    def top_k(self):
        yield 2

    @pytest.fixture
    def group_size(self):
        yield 128

    @pytest.fixture
    def renormalize(self):
        yield True

    @pytest.fixture
    def build_hidden_states(self, seq_len, in_size, group_size, quant_dtype, device):
        yield _make_A(seq_len, in_size, group_size=group_size, out_dtype=quant_dtype, device=device)

    @pytest.fixture
    def hidden_states(self, build_hidden_states, dtype):
        yield build_hidden_states[0].to(dtype)

    @pytest.fixture
    def states_quanted(self, build_hidden_states):
        yield build_hidden_states[1]

    @pytest.fixture
    def states_scale(self, build_hidden_states):
        yield build_hidden_states[2]

    @pytest.fixture
    def build_w1(self, num_experts, hidden_size, in_size, group_size, quant_dtype, device):
        yield _make_B(num_experts, in_size, hidden_size, group_size=group_size, out_dtype=quant_dtype, device=device)

    @pytest.fixture
    def w1(self, build_w1, dtype):
        yield build_w1[0].to(dtype)

    @pytest.fixture
    def w1_quant(self, build_w1):
        yield build_w1[1]

    @pytest.fixture
    def w1_scale(self, build_w1):
        yield build_w1[2]

    @pytest.fixture
    def build_w2(self, num_experts, out_size, hidden_size, group_size, quant_dtype, device):
        yield _make_B(num_experts,
                      hidden_size // 2,
                      out_size,
                      group_size=group_size,
                      out_dtype=quant_dtype,
                      device=device)

    @pytest.fixture
    def w2(self, build_w2, dtype):
        yield build_w2[0].to(dtype)

    @pytest.fixture
    def w2_quant(self, build_w2):
        yield build_w2[1]

    @pytest.fixture
    def w2_scale(self, build_w2):
        yield build_w2[2]

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
    def gt(self, hidden_states, w1, w2, topk_weights, topk_idx, top_k, renormalize):
        from lmdeploy.pytorch.kernels.cuda.fused_moe import fused_moe
        output = fused_moe(hidden_states, w1, w2, topk_weights, topk_idx, topk=top_k, renormalize=renormalize)
        yield output

    @torch.inference_mode()
    def test_fused_moe(self, states_quanted, states_scale, w1_quant, w1_scale, w2_quant, w2_scale, topk_weights,
                       topk_idx, top_k, renormalize, gt):
        from lmdeploy.pytorch.kernels.cuda.blocked_fp8_fused_moe import fused_moe_blocked_fp8
        output = fused_moe_blocked_fp8(states_quanted,
                                       states_scale,
                                       w1_quant,
                                       w1_scale,
                                       w2_quant,
                                       w2_scale,
                                       topk_weights,
                                       topk_idx,
                                       topk=top_k,
                                       renormalize=renormalize)
        out_max = output.abs().max()
        gt_max = gt.abs().max()
        assert (out_max - gt_max).abs() / out_max < 0.05

        norm_out = output / out_max
        norm_gt = gt / gt_max
        torch.testing.assert_close(norm_out, norm_gt, atol=0.05, rtol=1e-3)
