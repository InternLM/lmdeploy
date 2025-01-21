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
    return B, quant_B, scale


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
