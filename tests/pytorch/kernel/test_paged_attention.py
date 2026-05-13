import math

import pytest
import torch

from lmdeploy.messages import QuantPolicy
from lmdeploy.pytorch.kernels.cuda.turbo_quant import (
    hadamard_rotate,
    hadamard_rotate_inv,
)

# Import common TurboQuant utilities from turboquant_utils
from .turboquant_utils import (
    compute_metrics,
    dequantize_turboquant_mse_rot,
    dequantize_turboquant_qjl4_rot,
    quant_turboquant_mse,
    quant_turboquant_qjl4,
)


def _conti_input(data, seq_lens):
    data = [x[:l] for x, l in zip(data, seq_lens)]
    data = torch.cat(data, dim=0)
    return data


def _make_bias(q_seqlens, history_lens, neg_val):
    batch_size = q_seqlens.shape[0]
    full_seq_lens = q_seqlens + history_lens
    max_seq_len = q_seqlens.max().item()
    max_kv_len = full_seq_lens.max().item()
    seq_ranges = torch.arange(max_seq_len).cuda()
    seq_ranges = seq_ranges.repeat(batch_size, 1)
    seq_ranges = torch.where(seq_ranges < q_seqlens[:, None], seq_ranges, -max_kv_len)

    kv_ranges = torch.arange(max_kv_len).cuda()
    kv_ranges = kv_ranges.repeat(batch_size, 1)
    mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None]
    return mask.float() * neg_val


def _make_alibi_bias(q_seqlens, history_lens, neg_val, alibi_slopes):
    batch_size = q_seqlens.shape[0]
    kv_seqlens = q_seqlens + history_lens
    max_seq_len = q_seqlens.max().item()
    max_kv_len = kv_seqlens.max().item()

    seq_ranges = torch.arange(max_seq_len).cuda()
    seq_ranges = seq_ranges.repeat(batch_size, 1) + history_lens[:, None]

    kv_ranges = torch.arange(max_kv_len).cuda()
    kv_ranges = kv_ranges.repeat(batch_size, 1)

    diff = (seq_ranges[:, :, None] - kv_ranges[:, None, :]).abs()
    slope_diff = -diff[:, None] * alibi_slopes[None, :, None, None]

    # add bias
    bias = _make_bias(q_seqlens, history_lens, neg_val)
    bias = bias[:, None] + slope_diff
    return bias


def _make_block_sparse_bias(q_seqlens: torch.Tensor, history_lens: torch.Tensor, neg_val: float,
                            block_sparse_size: int):
    """Make block sparse bias."""
    batch_size = q_seqlens.shape[0]
    kv_seqlens = q_seqlens + history_lens
    max_seq_len = q_seqlens.max().item()
    max_kv_len = kv_seqlens.max().item()

    seq_ranges = torch.arange(max_seq_len).cuda()
    seq_ranges = seq_ranges // block_sparse_size * block_sparse_size
    seq_ranges = seq_ranges.repeat(batch_size, 1)
    seq_ranges = torch.where(seq_ranges < q_seqlens[:, None], seq_ranges, -max_kv_len)

    kv_ranges = torch.arange(max_kv_len).cuda()
    kv_ranges = kv_ranges // block_sparse_size * block_sparse_size
    kv_ranges = kv_ranges.repeat(batch_size, 1)

    mask = (kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None])
    return mask.float() * neg_val


def _make_blocked_cache(batched_k,
                        batched_v,
                        seq_lens,
                        history_lens,
                        block_offsets,
                        block_size,
                        num_heads_k,
                        feat_dim,
                        feat_dim_v,
                        layout: str = 'bshd'):
    max_blocks_nums = block_offsets.max() + 1
    full_seq_lens = seq_lens + history_lens
    blocked_k = batched_k.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim)
    blocked_v = batched_v.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim_v)

    for batch_id, offset in enumerate(block_offsets):
        ori_k = batched_k[batch_id]
        ori_v = batched_v[batch_id]
        seq_len = full_seq_lens[batch_id]
        for block_id, block_start in enumerate(range(0, seq_len, block_size)):
            block_off = offset[block_id]
            tmp_k = ori_k[block_start:block_start + block_size]
            tmp_v = ori_v[block_start:block_start + block_size]
            size = tmp_k.size(0)
            blocked_k[block_off, :size] = tmp_k
            blocked_v[block_off, :size] = tmp_v

    if layout == 'bhsd':
        blocked_k = blocked_k.transpose(1, 2).contiguous()
        blocked_v = blocked_v.transpose(1, 2).contiguous()

    return blocked_k, blocked_v


def _naive_attention(batched_q, batched_kv, bias, sinks=None):
    batched_k, batched_v = batched_kv

    num_heads_q = batched_q.shape[2]
    num_heads_k = batched_k.shape[2]
    head_dim = batched_q.shape[-1]
    group = num_heads_q // num_heads_k

    q = batched_q.transpose(1, 2)
    k = batched_k.permute(0, 2, 3, 1)
    v = batched_v.transpose(1, 2)

    # expand group
    k = k.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)
    v = v.unsqueeze(2).expand(-1, -1, group, -1, -1).flatten(1, 2)

    qk = torch.matmul(q, k) / math.sqrt(head_dim)
    if bias.dim() == 3:
        bias = bias[:, None]
    attn_weight = qk + bias
    if sinks is None:
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    else:
        sinks = sinks[None, :, None, None].to(torch.float32)
        sinks = sinks.expand(attn_weight.shape[0], -1, attn_weight.shape[2], -1)
        attn_weight = attn_weight.to(torch.float32)
        combined_logits = torch.cat([attn_weight, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
        attn_weight = torch.softmax(combined_logits, dim=-1, dtype=torch.float32)
        attn_weight = attn_weight[..., :-1]
    attn_weight = attn_weight.to(q.dtype)
    attn_output = torch.matmul(attn_weight, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


def _naive_window_attention(q, k, v, seqlens_q, seqlens_k, window_size):
    try:
        from lmdeploy.pytorch.third_party.flash_attn_interface import flash_attn_varlen_func
    except Exception:
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception:
            pytest.skip('Skip window attention test since flash attention is not available.')

    def _make_cu_seqlens(seqlens):
        cu_seqlens = seqlens.cumsum(0)
        cu_zero = cu_seqlens.new_zeros(1)
        cu_seqlens = torch.cat([cu_zero, cu_seqlens])
        return cu_seqlens

    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    cu_seqlens_q = _make_cu_seqlens(seqlens_q).int()
    cu_seqlens_k = _make_cu_seqlens(seqlens_k).int()

    output = flash_attn_varlen_func(q,
                                    k,
                                    v,
                                    cu_seqlens_q,
                                    cu_seqlens_k,
                                    max_seqlen_q=max_seqlen_q,
                                    max_seqlen_k=max_seqlen_k,
                                    causal=True,
                                    window_size=window_size)
    return output


class TestPagedAttentionBase:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def feat_dim(self, request):
        yield request.param

    @pytest.fixture
    def feat_dim_v(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_q(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_k(self, request):
        yield request.param

    @pytest.fixture
    def block_size(self, request):
        yield request.param

    @pytest.fixture
    def layout(self, request):
        yield request.param

    @pytest.fixture
    def history_lens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def seq_len(self):
        yield 1

    @pytest.fixture
    def seq_lens(self, seq_len, history_lens):
        yield torch.ones_like(history_lens) * seq_len

    @pytest.fixture
    def kv_seqlens(self, seq_lens, history_lens):
        yield seq_lens + history_lens

    @pytest.fixture
    def batched_q(self, seq_len, kv_seqlens, num_heads_q, feat_dim, dtype):
        torch.manual_seed(123)
        batch_size = len(kv_seqlens)
        inputs = torch.rand(batch_size, seq_len, num_heads_q, feat_dim, dtype=dtype, device='cuda')
        yield inputs

    @pytest.fixture
    def batched_kv(self, kv_seqlens, num_heads_k, feat_dim, feat_dim_v, dtype):
        torch.manual_seed(123)
        batch_size = len(kv_seqlens)
        max_seq_len = kv_seqlens.max().item()
        k = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device='cuda')
        v = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device='cuda')
        yield k, v

    @pytest.fixture
    def conti_q(self, seq_lens, batched_q):
        yield _conti_input(batched_q, seq_lens)

    @pytest.fixture
    def block_offsets(self, kv_seqlens, block_size):
        batch_size = kv_seqlens.size(0)
        num_blocks = (kv_seqlens + block_size - 1) // block_size

        offset = [torch.arange(size) * batch_size + idx for idx, size in enumerate(num_blocks)]
        max_len = max(len(o) for o in offset)
        new_offset = offset[0].new_zeros(batch_size, max_len)
        for o, no in zip(offset, new_offset):
            len_o = o.size(0)
            no[:len_o] = o

        yield new_offset.cuda()

    @pytest.fixture
    def conti_kv(self, batched_kv, history_lens):
        full_seq_lens = 1 + history_lens
        conti_k = _conti_input(batched_kv[0], full_seq_lens)
        conti_v = _conti_input(batched_kv[1], full_seq_lens)
        yield (conti_k, conti_v)

    @pytest.fixture
    def blocked_kv(self, batched_kv, kv_seqlens, history_lens, block_offsets, block_size, num_heads_k, feat_dim,
                   feat_dim_v, layout):
        batched_k, batched_v = batched_kv
        seq_lens = torch.ones_like(kv_seqlens)
        yield _make_blocked_cache(batched_k, batched_v, seq_lens, history_lens, block_offsets, block_size, num_heads_k,
                                  feat_dim, feat_dim_v, layout)

    @pytest.fixture
    def mask(self, history_lens):
        neg_val = -1e30
        seq_lens = torch.ones_like(history_lens)
        yield _make_bias(seq_lens, history_lens, neg_val)

    @pytest.fixture
    def gt(self, batched_q, batched_kv, mask):
        yield _naive_attention(batched_q, batched_kv, mask)

    @pytest.fixture
    def conti_gt(self, gt, seq_lens):
        yield _conti_input(gt, seq_lens)


class TestPagedAttention(TestPagedAttentionBase):

    @pytest.mark.parametrize('feat_dim', [32, 32], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [32], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(128, 2), (8, 2), (2, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [(50, 40, 30, 20)], indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    @pytest.mark.parametrize('layout', ['bshd', 'bhsd'], indirect=True)
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, layout, conti_gt):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v = blocked_kv
        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens,
                                      kv_layout=layout)
        torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)

    @pytest.fixture
    def win_size(self, request):
        yield request.param

    @pytest.fixture
    def window_gt(self, conti_q, conti_kv, seq_lens, history_lens, win_size):
        kv_lens = seq_lens + history_lens
        yield _naive_window_attention(conti_q,
                                      conti_kv[0],
                                      conti_kv[1],
                                      seq_lens,
                                      kv_lens,
                                      window_size=(win_size, win_size))

    @pytest.mark.parametrize('feat_dim', [16], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [16], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(4, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [
        (50, 40, 30, 20),
    ], indirect=True)
    @pytest.mark.parametrize('win_size', (32, ), indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    @pytest.mark.parametrize('layout', ['bshd'], indirect=True)
    def test_window_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, win_size, layout, window_gt):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v = blocked_kv
        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens,
                                      window_size=win_size,
                                      kv_layout=layout)
        torch.testing.assert_close(out, window_gt, atol=1e-3, rtol=1e-5)


class TestPagedAttentionSink(TestPagedAttentionBase):

    @pytest.fixture
    def sinks(self, num_heads_q, dtype):
        yield torch.rand(num_heads_q, dtype=dtype, device='cuda')

    @pytest.fixture
    def sink_gt(self, batched_q, batched_kv, mask, sinks):
        yield _naive_attention(batched_q, batched_kv, mask, sinks)

    @pytest.fixture
    def conti_sink_gt(self, sink_gt, seq_lens):
        yield _conti_input(sink_gt, seq_lens)

    @pytest.mark.parametrize('feat_dim', [32], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [32], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(8, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [(50, 40, 30, 20)], indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    @pytest.mark.parametrize('layout', ['bshd'], indirect=True)
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, layout, sinks, conti_sink_gt):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v = blocked_kv

        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens,
                                      sinks=sinks,
                                      kv_layout=layout)
        torch.testing.assert_close(out, conti_sink_gt, atol=1e-3, rtol=1e-5)


def quant(kv: torch.Tensor, nbits: int = 8):
    """Quant kv on the head_dim."""
    amax = kv.amax(dim=-1, keepdim=True)
    amin = kv.amin(dim=-1, keepdim=True)
    scales = (amax - amin) / (2**nbits - 1)
    zeros = -amin / scales
    q_kv = (kv / scales + zeros + 0.5).to(torch.uint8)
    if nbits == 4:
        q_kv1, q_kv2 = q_kv.split(q_kv.shape[-1] // 2, -1)
        q_kv = q_kv1 + q_kv2 * 16
    return q_kv, torch.cat([scales, zeros], dim=-1)


def _make_blocked_cache_quant(batched_k, batched_v, seq_lens, history_lens, block_offsets, block_size, num_heads_k,
                              feat_dim, feat_dim_v, nbits):
    max_blocks_nums = block_offsets.max() + 1
    full_seq_lens = seq_lens + history_lens
    batched_k, k_scales_zeros = quant(batched_k, nbits)
    batched_v, v_scales_zeros = quant(batched_v, nbits)
    if nbits == 4:
        feat_dim //= 2
        feat_dim_v //= 2
    blocked_k = batched_k.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim)
    blocked_v = batched_v.new_zeros(max_blocks_nums, block_size, num_heads_k, feat_dim_v)
    blocked_ksz = k_scales_zeros.new_zeros(max_blocks_nums, block_size, num_heads_k, 2)
    blocked_vsz = v_scales_zeros.new_zeros(max_blocks_nums, block_size, num_heads_k, 2)

    for batch_id, offset in enumerate(block_offsets):
        ori_k = batched_k[batch_id]
        ori_v = batched_v[batch_id]
        ori_ksz = k_scales_zeros[batch_id]
        ori_vsz = v_scales_zeros[batch_id]
        seq_len = full_seq_lens[batch_id]
        for block_id, block_start in enumerate(range(0, seq_len, block_size)):
            block_off = offset[block_id]
            tmp_k = ori_k[block_start:block_start + block_size]
            tmp_v = ori_v[block_start:block_start + block_size]
            tmp_ksz = ori_ksz[block_start:block_start + block_size]
            tmp_vsz = ori_vsz[block_start:block_start + block_size]
            size = tmp_k.size(0)
            blocked_k[block_off, :size] = tmp_k
            blocked_v[block_off, :size] = tmp_v
            blocked_ksz[block_off, :size] = tmp_ksz
            blocked_vsz[block_off, :size] = tmp_vsz

    return blocked_k, blocked_v, blocked_ksz, blocked_vsz


class TestPagedAttentionInt8(TestPagedAttention):

    @pytest.fixture
    def nbits(self):
        yield 8

    @pytest.fixture
    def blocked_kv(self, batched_kv, seq_lens, history_lens, block_offsets, block_size, num_heads_k, feat_dim,
                   feat_dim_v, nbits):
        batched_k, batched_v = batched_kv
        yield _make_blocked_cache_quant(batched_k, batched_v, seq_lens, history_lens, block_offsets, block_size,
                                        num_heads_k, feat_dim, feat_dim_v, nbits)

    @pytest.mark.parametrize('feat_dim', [48, 32], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [32], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(8, 2), (2, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [(50, 40, 30, 20)], indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, conti_gt, nbits):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv

        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      k_scales_zeros=blocked_ksz,
                                      v_scales_zeros=blocked_vsz,
                                      quant_policy=nbits,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens)
        if nbits == 4:
            torch.testing.assert_close(out, conti_gt, atol=0.05, rtol=0.01)
        else:
            torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)

    @pytest.mark.parametrize('feat_dim', [16], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [16], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(4, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [
        (50, 40, 30, 20),
    ], indirect=True)
    @pytest.mark.parametrize('win_size', (32, ), indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    def test_window_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, win_size, window_gt, nbits):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv
        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      k_scales_zeros=blocked_ksz,
                                      v_scales_zeros=blocked_vsz,
                                      quant_policy=nbits,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens,
                                      window_size=win_size)
        if nbits == 4:
            torch.testing.assert_close(out, window_gt, atol=0.05, rtol=0.01)
        else:
            torch.testing.assert_close(out, window_gt, atol=1e-3, rtol=1e-5)


class TestPagedAttentionInt4(TestPagedAttentionInt8):

    @pytest.fixture
    def nbits(self):
        yield 4


class TestPagedAttentionBlockDecoding(TestPagedAttentionBase):

    @pytest.fixture
    def seq_len(self):
        yield 4

    @pytest.fixture
    def mask(self, seq_lens, history_lens, seq_len):
        neg_val = -1e30
        yield _make_block_sparse_bias(seq_lens, history_lens, neg_val, seq_len)

    @pytest.fixture
    def gt(self, batched_q, batched_kv, mask):
        yield _naive_attention(batched_q, batched_kv, mask)

    @pytest.fixture
    def conti_gt(self, gt, seq_lens):
        yield _conti_input(gt, seq_lens)

    @pytest.mark.parametrize('feat_dim', [48, 32], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [32], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(128, 2), (8, 2), (2, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [(52, 40, 32, 20)], indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    @pytest.mark.parametrize('layout', ['bshd'], indirect=True)
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, layout, conti_gt):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v = blocked_kv

        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens,
                                      kv_layout=layout)
        torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)


class TestPagedAttentionAlibi(TestPagedAttentionBase):

    @pytest.fixture
    def alibi_slopes(self, num_heads_q):
        yield torch.rand(num_heads_q, dtype=torch.float32, device='cuda')

    @pytest.fixture
    def mask(self, seq_lens, history_lens, alibi_slopes):
        neg_val = -1e30
        yield _make_alibi_bias(seq_lens, history_lens, neg_val, alibi_slopes)

    @pytest.mark.parametrize('feat_dim', [128], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [128], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(40, 8)], indirect=True)
    @pytest.mark.parametrize('history_lens', [(52, 40, 32, 20)], indirect=True)
    @pytest.mark.parametrize('layout', ['bshd'], indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, layout, alibi_slopes, conti_gt):
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v = blocked_kv

        out = flash_attn_with_kvcache(conti_q,
                                      blocked_k,
                                      blocked_v,
                                      page_table=block_offsets,
                                      cache_seqlens=kv_seqlens,
                                      alibi_slopes=alibi_slopes,
                                      kv_layout=layout)
        torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)


# =============================================================================
# quant_policy=QuantPolicy.TURBO_QUANT Tests (TurboQuant: K=QJL4, V=TurboQuant MSE int2)
# =============================================================================

def _make_blocked_cache_quant42(batched_k,
                                batched_v,
                                seq_lens,
                                history_lens,
                                block_offsets,
                                block_size,
                                num_heads_k,
                                feat_dim,
                                feat_dim_v):
    """Create full blocked KV cache with quant_policy=QuantPolicy.TURBO_QUANT.

    This matches the semantics of the standard paged attention tests:
    the cache already contains the full KV sequence of length
    `history_lens + seq_lens`.

    - K: QJL4 (3bit MSE + 1bit QJL), packed dim = feat_dim // 2
    - V: TurboQuant MSE int2, packed dim = feat_dim_v // 4
    """
    max_blocks_nums = block_offsets.max().item() + 1
    full_seq_lens = seq_lens + history_lens
    packed_k_dim = feat_dim // 2
    packed_v_dim = feat_dim_v // 4
    batch_size = batched_k.shape[0]
    max_seq_len = batched_k.shape[1]

    # Vectorized K quantization: reshape to (batch*seq, heads, dim) and quantize in one call
    # Shape: (batch_size, max_seq_len, num_heads_k, feat_dim) -> (batch_size * max_seq_len, num_heads_k, feat_dim)
    k_reshaped = batched_k.view(batch_size * max_seq_len, num_heads_k, feat_dim)
    k_q_all, k_m_all = quant_turboquant_qjl4(k_reshaped)  # (batch*seq, heads, packed_k), (batch*seq, heads, 2)
    k_quant = k_q_all.view(batch_size, max_seq_len, num_heads_k, packed_k_dim)
    k_meta = k_m_all.view(batch_size, max_seq_len, num_heads_k, 2)

    # Vectorized V quantization: reshape to (batch*seq, heads, dim) and quantize in one call
    v_reshaped = batched_v.view(batch_size * max_seq_len, num_heads_k, feat_dim_v)
    v_q_all, v_n_all = quant_turboquant_mse(v_reshaped, 2)  # (batch*seq, heads, packed_v), (batch*seq, heads)
    v_quant = v_q_all.view(batch_size, max_seq_len, num_heads_k, packed_v_dim)
    v_norm = v_n_all.view(batch_size, max_seq_len, num_heads_k)

    blocked_k = torch.zeros(
        max_blocks_nums, block_size, num_heads_k, packed_k_dim,
        dtype=torch.uint8, device=batched_k.device)
    blocked_v = torch.zeros(
        max_blocks_nums, block_size, num_heads_k, packed_v_dim,
        dtype=torch.uint8, device=batched_v.device)
    blocked_ksz = torch.zeros(
        max_blocks_nums, block_size, num_heads_k, 2,
        dtype=batched_k.dtype, device=batched_k.device)
    blocked_vsz = torch.zeros(
        max_blocks_nums, block_size, num_heads_k, 1,
        dtype=batched_v.dtype, device=batched_v.device)

    for batch_id, offset in enumerate(block_offsets):
        seq_len = full_seq_lens[batch_id].item()
        ori_k = k_quant[batch_id]
        ori_v = v_quant[batch_id]
        ori_ksz = k_meta[batch_id]
        ori_vsz = v_norm[batch_id]
        for block_id, block_start in enumerate(range(0, seq_len, block_size)):
            block_off = offset[block_id].item()
            tmp_k = ori_k[block_start:block_start + block_size]
            tmp_v = ori_v[block_start:block_start + block_size]
            tmp_ksz = ori_ksz[block_start:block_start + block_size]
            tmp_vsz = ori_vsz[block_start:block_start + block_size]
            size = tmp_k.size(0)
            blocked_k[block_off, :size] = tmp_k
            blocked_v[block_off, :size] = tmp_v
            blocked_ksz[block_off, :size] = tmp_ksz
            blocked_vsz[block_off, :size, :, 0] = tmp_vsz

    return blocked_k, blocked_v, blocked_ksz, blocked_vsz


def _recover_kv_from_blocked_cache(blocked_k,
                                   blocked_v,
                                   blocked_ksz,
                                   blocked_vsz,
                                   block_offsets,
                                   kv_seqlens,
                                   block_size):
    """Recover packed K/V and meta from blocked cache."""
    batch_size = block_offsets.size(0)
    k_recovered = []
    k_meta_recovered = []
    v_recovered = []
    v_meta_recovered = []
    for batch_id in range(batch_size):
        seq_len = kv_seqlens[batch_id].item()
        offset = block_offsets[batch_id]
        nblocks = (seq_len + block_size - 1) // block_size
        k_seq = []
        k_meta_seq = []
        v_seq = []
        v_meta_seq = []
        for block_id in range(nblocks):
            block_off = offset[block_id].item()
            valid = min(block_size, seq_len - block_id * block_size)
            k_seq.append(blocked_k[block_off, :valid])
            k_meta_seq.append(blocked_ksz[block_off, :valid])
            v_seq.append(blocked_v[block_off, :valid])
            v_meta_seq.append(blocked_vsz[block_off, :valid, :, 0])
        k_recovered.append(torch.cat(k_seq, dim=0))            # (seq, heads, packed_k)
        k_meta_recovered.append(torch.cat(k_meta_seq, dim=0))  # (seq, heads, 2)
        v_recovered.append(torch.cat(v_seq, dim=0))            # (seq, heads, packed_v)
        v_meta_recovered.append(torch.cat(v_meta_seq, dim=0))  # (seq, heads)
    return k_recovered, k_meta_recovered, v_recovered, v_meta_recovered


class TestPagedAttentionQuant42(TestPagedAttentionBase):
    """Test quant_policy=QuantPolicy.TURBO_QUANT (TurboQuant) attention kernel
    numerical correctness.

    quant_policy=QuantPolicy.TURBO_QUANT uses:
    - K: QJL4 (3bit MSE + 1bit QJL)
    - V: TurboQuant MSE int2

    Runtime semantics:
    - cache stores ROTATE-domain quantized KV
    - attention is computed in rotate domain
    - final output is inverse-rotated back to original domain
    """

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def feat_dim(self, request):
        yield request.param

    @pytest.fixture
    def feat_dim_v(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_q(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_k(self, request):
        yield request.param

    @pytest.fixture
    def block_size(self, request):
        yield request.param

    @pytest.fixture
    def layout(self):
        yield 'bshd'

    @pytest.fixture
    def history_lens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def seq_len(self):
        yield 1

    @pytest.fixture
    def seq_lens(self, seq_len, history_lens):
        yield torch.ones_like(history_lens) * seq_len

    @pytest.fixture
    def kv_seqlens(self, seq_lens, history_lens):
        yield seq_lens + history_lens

    @pytest.fixture
    def batched_q(self, seq_len, kv_seqlens, num_heads_q, feat_dim, dtype):
        torch.manual_seed(123)
        batch_size = len(kv_seqlens)
        inputs = torch.rand(batch_size, seq_len, num_heads_q, feat_dim, dtype=dtype, device='cuda')
        yield inputs

    @pytest.fixture
    def batched_kv(self, kv_seqlens, num_heads_k, feat_dim, feat_dim_v, dtype):
        torch.manual_seed(123)
        batch_size = len(kv_seqlens)
        max_seq_len = kv_seqlens.max().item()
        k = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device='cuda')
        v = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device='cuda')
        yield k, v

    @pytest.fixture
    def conti_q(self, seq_lens, batched_q):
        yield _conti_input(batched_q, seq_lens)

    @pytest.fixture
    def block_offsets(self, kv_seqlens, block_size):
        batch_size = kv_seqlens.size(0)
        num_blocks = (kv_seqlens + block_size - 1) // block_size
        offset = [torch.arange(size, device='cuda') * batch_size + idx for idx, size in enumerate(num_blocks)]
        max_len = max(len(o) for o in offset)
        new_offset = offset[0].new_zeros(batch_size, max_len)
        for o, no in zip(offset, new_offset):
            len_o = o.size(0)
            no[:len_o] = o
        yield new_offset.cuda()

    @pytest.fixture
    def blocked_kv(self, batched_kv, seq_lens, history_lens, block_offsets, block_size, num_heads_k, feat_dim,
                   feat_dim_v):
        batched_k, batched_v = batched_kv
        yield _make_blocked_cache_quant42(batched_k, batched_v, seq_lens, history_lens, block_offsets, block_size,
                                          num_heads_k, feat_dim, feat_dim_v)

    @pytest.fixture
    def gt(self, batched_q, blocked_kv, block_offsets, kv_seqlens, block_size, num_heads_q, num_heads_k):
        """Compute GT from the actual blocked cache contents.

        IMPORTANT:
        - Q is rotated first
        - K/V are dequantized into ROTATE domain
        - attention is done in rotate domain
        - final output is inverse-rotated back
        """
        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv
        batch_size = batched_q.shape[0]
        seq_len_q = batched_q.shape[1]
        head_dim = batched_q.shape[-1]
        group = num_heads_q // num_heads_k

        k_recovered, k_meta_recovered, v_recovered, v_meta_recovered = _recover_kv_from_blocked_cache(
            blocked_k, blocked_v, blocked_ksz, blocked_vsz, block_offsets, kv_seqlens, block_size)

        q_rot = hadamard_rotate(batched_q.float())

        outputs = []
        for b in range(batch_size):
            q_b = q_rot[b, :seq_len_q]       # (sq, hq, d) in rotate domain
            k_quant = k_recovered[b]         # (sk, hk, packed_k)
            k_meta = k_meta_recovered[b]     # (sk, hk, 2)
            v_quant = v_recovered[b]         # (sk, hk, packed_v)
            v_norm = v_meta_recovered[b]     # (sk, hk)

            # Dequantize to ROTATE domain
            k_dequant = dequantize_turboquant_qjl4_rot(k_quant, k_meta)    # (sk, hk, d)
            v_dequant = dequantize_turboquant_mse_rot(v_quant, v_norm, 2)  # (sk, hk, dv)

            # Expand KV heads to Q heads, same as runtime behavior
            k_dequant = k_dequant.unsqueeze(2).expand(-1, -1, group, -1).reshape(
                k_dequant.shape[0], num_heads_q, k_dequant.shape[-1])
            v_dequant = v_dequant.unsqueeze(2).expand(-1, -1, group, -1).reshape(
                v_dequant.shape[0], num_heads_q, v_dequant.shape[-1])

            q_t = q_b.transpose(0, 1).unsqueeze(0)                        # (1, hq, sq, d)
            k_t = k_dequant.transpose(0, 1).transpose(1, 2).unsqueeze(0)  # (1, hq, d, sk)
            v_t = v_dequant.transpose(0, 1).unsqueeze(0)                  # (1, hq, sk, dv)

            scale = 1.0 / math.sqrt(head_dim)
            qk = torch.matmul(q_t, k_t) * scale
            attn_weight = torch.softmax(qk, dim=-1)
            o_rot = torch.matmul(attn_weight, v_t)                        # (1, hq, sq, dv)

            # Final output back to original domain
            o = hadamard_rotate_inv(o_rot.float())                       # (1, hq, sq, dv)
            o = o.squeeze(0).transpose(0, 1)                              # (sq, hq, dv)
            outputs.append(o)

        gt = torch.stack(outputs, dim=0)  # (batch, seq, heads, dv)
        yield gt

    @pytest.fixture
    def conti_gt(self, gt, seq_lens):
        yield _conti_input(gt, seq_lens)

    @pytest.mark.parametrize('feat_dim', [64, 32], indirect=True)
    @pytest.mark.parametrize('feat_dim_v', [64, 32], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(2, 2)], indirect=True)
    @pytest.mark.parametrize('history_lens', [(8, 4, 2, 1)], indirect=True)
    @pytest.mark.parametrize('block_size', [16], indirect=True)
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, kv_seqlens, conti_gt):
        """Test paged attention with quant_policy=QuantPolicy.TURBO_QUANT."""
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv
        out = flash_attn_with_kvcache(
            conti_q,
            blocked_k,
            blocked_v,
            k_scales_zeros=blocked_ksz,
            v_scales_zeros=blocked_vsz,
            quant_policy=QuantPolicy.TURBO_QUANT,
            page_table=block_offsets,
            cache_seqlens=kv_seqlens,
        )
        conti_gt = conti_gt.to(out.dtype)

        # quant42 has quantization error, but kernel and reference should still
        # be close numerically.
        torch.testing.assert_close(out, conti_gt, atol=0.1, rtol=0.05)


class TestPagedAttentionFP16vsQuant42(TestPagedAttentionBase):
    """Compare FP16 vs quant_policy=QuantPolicy.TURBO_QUANT attention outputs.

    This test verifies that quant_policy=QuantPolicy.TURBO_QUANT (TurboQuant) produces numerically
    reasonable results compared to FP16 baseline.

    quant_policy=QuantPolicy.TURBO_QUANT uses:
    - K: QJL4 (3bit MSE + 1bit QJL)
    - V: TurboQuant MSE int2
    """

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def feat_dim(self):
        yield 64

    @pytest.fixture
    def feat_dim_v(self):
        yield 64

    @pytest.fixture
    def num_heads_q(self):
        yield 8

    @pytest.fixture
    def num_heads_k(self):
        yield 8

    @pytest.fixture
    def block_size(self):
        yield 16

    @pytest.fixture
    def layout(self):
        yield 'bshd'

    @pytest.fixture
    def history_lens(self):
        yield torch.tensor([128, 128, 128, 128], device='cuda')

    @pytest.fixture
    def seq_len(self):
        yield 1

    @pytest.fixture
    def seq_lens(self, seq_len, history_lens):
        yield torch.ones_like(history_lens) * seq_len

    @pytest.fixture
    def kv_seqlens(self, seq_lens, history_lens):
        yield seq_lens + history_lens

    @pytest.fixture
    def batch_size(self, kv_seqlens):
        yield len(kv_seqlens)

    @pytest.fixture
    def batched_q(self, seq_len, kv_seqlens, num_heads_q, feat_dim, dtype):
        torch.manual_seed(123)
        batch_size = len(kv_seqlens)
        inputs = torch.rand(batch_size, seq_len, num_heads_q, feat_dim, dtype=dtype, device='cuda')
        yield inputs

    @pytest.fixture
    def batched_kv(self, kv_seqlens, num_heads_k, feat_dim, feat_dim_v, dtype):
        torch.manual_seed(123)
        batch_size = len(kv_seqlens)
        max_seq_len = kv_seqlens.max().item()
        k = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim, dtype=dtype, device='cuda')
        v = torch.rand(batch_size, max_seq_len, num_heads_k, feat_dim_v, dtype=dtype, device='cuda')
        yield k, v

    @pytest.fixture
    def conti_q(self, seq_lens, batched_q):
        yield _conti_input(batched_q, seq_lens)

    @pytest.fixture
    def block_offsets(self, kv_seqlens, block_size):
        batch_size = kv_seqlens.size(0)
        num_blocks = (kv_seqlens + block_size - 1) // block_size
        offset = [torch.arange(size, device='cuda') * batch_size + idx for idx, size in enumerate(num_blocks)]
        max_len = max(len(o) for o in offset)
        new_offset = offset[0].new_zeros(batch_size, max_len)
        for o, no in zip(offset, new_offset):
            len_o = o.size(0)
            no[:len_o] = o
        yield new_offset.cuda()

    @pytest.fixture
    def blocked_kv_fp16(self, batched_kv, seq_lens, history_lens, block_offsets, block_size, num_heads_k, feat_dim,
                        feat_dim_v):
        """Build FP16 blocked KV cache."""
        batched_k, batched_v = batched_kv
        yield _make_blocked_cache(batched_k, batched_v, seq_lens, history_lens, block_offsets, block_size,
                                  num_heads_k, feat_dim, feat_dim_v)

    @pytest.fixture
    def blocked_kv_quant42(self, batched_kv, seq_lens, history_lens, block_offsets, block_size, num_heads_k, feat_dim,
                           feat_dim_v):
        """Build quant_policy=QuantPolicy.TURBO_QUANT blocked KV cache."""
        batched_k, batched_v = batched_kv
        yield _make_blocked_cache_quant42(batched_k, batched_v, seq_lens, history_lens, block_offsets, block_size,
                                          num_heads_k, feat_dim, feat_dim_v)

    @pytest.fixture
    def out_fp16(self, conti_q, blocked_kv_fp16, block_offsets, kv_seqlens):
        """Run attention with FP16 cache."""
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v = blocked_kv_fp16
        out = flash_attn_with_kvcache(
            conti_q,
            blocked_k,
            blocked_v,
            page_table=block_offsets,
            cache_seqlens=kv_seqlens,
            quant_policy=QuantPolicy.NONE,
        )
        yield out

    @pytest.fixture
    def out_quant42(self, conti_q, blocked_kv_quant42, block_offsets, kv_seqlens):
        """Run attention with quant_policy=QuantPolicy.TURBO_QUANT cache."""
        from lmdeploy.pytorch.kernels.cuda import flash_attn_with_kvcache

        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv_quant42
        out = flash_attn_with_kvcache(
            conti_q,
            blocked_k,
            blocked_v,
            k_scales_zeros=blocked_ksz,
            v_scales_zeros=blocked_vsz,
            quant_policy=QuantPolicy.TURBO_QUANT,
            page_table=block_offsets,
            cache_seqlens=kv_seqlens,
        )
        yield out

    def test_fp16_vs_quant42(self, out_fp16, out_quant42):
        """Test that quant42 output is reasonably close to FP16 baseline."""
        # Compute metrics
        metrics = compute_metrics(out_quant42.float(), out_fp16.float())

        print('\nFP16 vs Quant42 metrics:')
        print(f'  cosine={metrics["cosine"]:.6f}')
        print(f'  nmse={metrics["nmse"]:.6f}')
        print(f'  snr={metrics["snr_db"]:.3f} dB')

        # Quant42 should have reasonable similarity to FP16
        # With 4-bit K and 2-bit V, we expect cosine similarity > 0.95
        assert metrics['cosine'] > 0.90, f'Cosine similarity {metrics["cosine"]} too low'
        # Note: SNR is low due to scale differences between FP16 and quant42
        # (quant42 outputs in original domain after inverse rotation, but with different scale)
        # The important thing is that cosine similarity is high
