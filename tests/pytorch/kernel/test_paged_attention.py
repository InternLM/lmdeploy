import math

import pytest
import torch


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
    attn_weight = qk + bias[:, None]
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
    from flash_attn import flash_attn_varlen_func

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
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, history_lens, feat_dim_v, layout, conti_gt):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd
        kv_seq_lens = 1 + history_lens

        blocked_k, blocked_v = blocked_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

        paged_attention_fwd(conti_q,
                            blocked_k,
                            blocked_v,
                            out,
                            block_offsets=block_offsets,
                            kv_seqlens=kv_seq_lens,
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
    def test_window_attention(self, conti_q, blocked_kv, block_offsets, history_lens, feat_dim_v, win_size, layout,
                              window_gt):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd
        kv_seq_lens = 1 + history_lens

        blocked_k, blocked_v = blocked_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)
        paged_attention_fwd(conti_q,
                            blocked_k,
                            blocked_v,
                            out,
                            block_offsets=block_offsets,
                            kv_seqlens=kv_seq_lens,
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
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, history_lens, feat_dim_v, layout, sinks,
                             conti_sink_gt):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd
        kv_seq_lens = 1 + history_lens

        blocked_k, blocked_v = blocked_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

        paged_attention_fwd(conti_q,
                            blocked_k,
                            blocked_v,
                            out,
                            block_offsets=block_offsets,
                            kv_seqlens=kv_seq_lens,
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
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, seq_lens, history_lens, feat_dim_v, conti_gt,
                             nbits):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd
        kv_seq_lens = 1 + history_lens

        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

        paged_attention_fwd(conti_q,
                            blocked_k,
                            blocked_v,
                            out,
                            k_scales_zeros=blocked_ksz,
                            v_scales_zeros=blocked_vsz,
                            quant_policy=nbits,
                            block_offsets=block_offsets,
                            kv_seqlens=kv_seq_lens)
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
    def test_window_attention(self, conti_q, blocked_kv, block_offsets, history_lens, feat_dim_v, win_size, window_gt,
                              nbits):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd
        kv_seq_lens = 1 + history_lens

        blocked_k, blocked_v, blocked_ksz, blocked_vsz = blocked_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)
        paged_attention_fwd(conti_q,
                            blocked_k,
                            blocked_v,
                            out,
                            k_scales_zeros=blocked_ksz,
                            v_scales_zeros=blocked_vsz,
                            quant_policy=nbits,
                            block_offsets=block_offsets,
                            kv_seqlens=kv_seq_lens,
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
    def test_paged_attention(self, conti_q, blocked_kv, block_offsets, seq_lens, history_lens, feat_dim_v, layout,
                             conti_gt):
        from lmdeploy.pytorch.kernels.cuda import paged_attention_fwd
        kv_seq_lens = seq_lens + history_lens

        blocked_k, blocked_v = blocked_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], feat_dim_v)

        paged_attention_fwd(conti_q,
                            blocked_k,
                            blocked_v,
                            out,
                            block_offsets=block_offsets,
                            kv_seqlens=kv_seq_lens,
                            kv_layout=layout)
        torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)
