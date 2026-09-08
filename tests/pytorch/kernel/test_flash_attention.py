import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_flash_attention_lse_merges_context_partitions():
    from lmdeploy.pytorch.kernels.cuda.dcp import merge_attention_states
    from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func

    torch.manual_seed(9)
    query = torch.randn(7, 4, 32, device='cuda', dtype=torch.float16)
    key = torch.randn(13, 4, 32, device='cuda', dtype=torch.float16)
    value = torch.randn_like(key)
    cu_q = torch.tensor([0, 7], device='cuda', dtype=torch.int32)
    cu_k = torch.tensor([0, 13], device='cuda', dtype=torch.int32)

    full_output, full_lse = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        causal=False,
        kv_layout='shd',
        return_lse=True,
    )
    left_output, left_lse = flash_attn_varlen_func(
        query,
        key[:5],
        value[:5],
        cu_seqlens_q=cu_q,
        cu_seqlens_k=torch.tensor([0, 5], device='cuda', dtype=torch.int32),
        causal=False,
        kv_layout='shd',
        return_lse=True,
    )
    right_output, right_lse = flash_attn_varlen_func(
        query,
        key[5:],
        value[5:],
        cu_seqlens_q=cu_q,
        cu_seqlens_k=torch.tensor([0, 8], device='cuda', dtype=torch.int32),
        causal=False,
        kv_layout='shd',
        return_lse=True,
    )
    padded_output = left_output.new_empty(left_output.size(0),
                                          left_output.size(1) * 2,
                                          left_output.size(2))
    padded_output[:, ::2].copy_(left_output)
    left_output = padded_output[:, ::2]
    padded_lse = left_lse.new_empty(left_lse.size(0), left_lse.size(1) * 2)
    padded_lse[:, ::2].copy_(left_lse)
    left_lse = padded_lse[:, ::2]
    merged_output, merged_lse = merge_attention_states(
        left_output, left_lse, right_output, right_lse)

    expected_lse = torch.logsumexp(
        torch.einsum('qhd,khd->qhk', query.float(), key.float()) /
        query.size(-1)**0.5,
        dim=-1,
    )
    torch.testing.assert_close(full_lse, expected_lse, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(merged_lse, full_lse, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(merged_output, full_output.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_attention_partition_merge_retains_fp32_accumulator(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from lmdeploy.pytorch.kernels.cuda.dcp import merge_attention_states

    # Equal-mass partitions: repeated BF16 rounding previously yielded 0.9414.
    output = -torch.ones(1, 1, 1, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(1, 1, device=device)
    suffix = torch.ones_like(output)
    suffix_lse = torch.zeros_like(lse)
    for _ in range(512):
        output, lse = merge_attention_states(output, lse, suffix, suffix_lse)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.full_like(output, 511 / 513), atol=1e-4, rtol=0)
    torch.testing.assert_close(lse, torch.full_like(lse, math.log(513)), atol=1e-4, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_attention_partition_merge_ignores_empty_nan_outputs(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from lmdeploy.pytorch.kernels.cuda.dcp import merge_attention_states

    output = torch.tensor([[[2.0]], [[float('nan')]]], device=device, dtype=torch.bfloat16)
    lse = torch.tensor([[0.0], [-torch.inf]], device=device)
    empty = torch.full_like(output, torch.nan)
    empty_lse = torch.full_like(lse, -torch.inf)
    merged, merged_lse = merge_attention_states(output, lse, empty, empty_lse)
    torch.testing.assert_close(merged, torch.tensor([[[2.0]], [[0.0]]], device=device))
    torch.testing.assert_close(merged_lse, lse)


def _conti_input(data, q_seqlens):
    data = [x[:l] for x, l in zip(data, q_seqlens)]
    data = torch.cat(data, dim=0)
    return data


def _make_bias(q_seqlens, history_lens, neg_val, causal):
    batch_size = q_seqlens.shape[0]
    kv_seqlens = q_seqlens + history_lens
    max_seq_len = q_seqlens.max().item()
    max_kv_len = kv_seqlens.max().item()
    if causal:
        seq_ranges = torch.arange(max_seq_len).cuda()
        seq_ranges = seq_ranges.repeat(batch_size, 1)
        seq_ranges = torch.where(seq_ranges < q_seqlens[:, None], seq_ranges, -max_kv_len)

        kv_ranges = torch.arange(max_kv_len).cuda()
        kv_ranges = kv_ranges.repeat(batch_size, 1)

        mask = (kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:, None, None])
        return mask.float() * neg_val
    else:
        q_mask = torch.arange(max_seq_len)[None].cuda() < q_seqlens[:, None]
        k_mask = torch.arange(max_kv_len)[None].cuda() < kv_seqlens[:, None]
        mask = q_mask[:, :, None] & k_mask[:, None, :]

        return (~mask).float() * neg_val


def _make_bias_alibi(q_seqlens, history_lens, neg_val, causal, alibi_slopes):

    batch_size = q_seqlens.shape[0]
    kv_seqlens = q_seqlens + history_lens
    max_q_len = q_seqlens.max().item()
    max_kv_len = kv_seqlens.max().item()

    device = 'cuda'
    q_ranges = torch.arange(max_q_len, device=device)
    seq_ranges = q_ranges.repeat(batch_size, 1) + history_lens[:, None]

    kv_ranges = torch.arange(max_kv_len, device=device)
    kv_ranges = kv_ranges.repeat(batch_size, 1)

    diff = (seq_ranges[:, :, None] - kv_ranges[:, None, :]).abs()
    slope_diff = -diff[:, None] * alibi_slopes[None, :, None, None]

    # add bias
    bias = _make_bias(q_seqlens, history_lens, neg_val, causal)
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


class TestFlashAttention:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def head_dim_k(self, request):
        yield request.param

    @pytest.fixture
    def head_dim_v(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_q(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_k(self, request):
        yield request.param

    @pytest.fixture
    def causal(self, request):
        yield request.param

    @pytest.fixture
    def q_seqlens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def cu_seqlens_q(self, q_seqlens):
        cu_seqlens = q_seqlens.cumsum(0)
        cu_zero = cu_seqlens.new_zeros(1)
        yield torch.cat([cu_zero, cu_seqlens]).int()

    @pytest.fixture
    def history_lens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def kv_seqlens(self, q_seqlens, history_lens):
        yield q_seqlens + history_lens

    @pytest.fixture
    def cu_seqlens_k(self, kv_seqlens):
        cu_seqlens = kv_seqlens.cumsum(0)
        cu_zero = cu_seqlens.new_zeros(1)
        yield torch.cat([cu_zero, cu_seqlens]).int()

    @pytest.fixture
    def batched_q(self, q_seqlens, num_heads_q, head_dim_k, dtype):
        torch.manual_seed(123)
        batch_size = len(q_seqlens)
        max_seq_len = q_seqlens.max().item()
        inputs = torch.rand(batch_size, max_seq_len, num_heads_q, head_dim_k, dtype=dtype, device='cuda')
        yield inputs

    @pytest.fixture
    def batched_kv(self, q_seqlens, history_lens, num_heads_k, head_dim_k, head_dim_v, dtype):
        torch.manual_seed(123)
        batch_size = len(q_seqlens)
        kv_seqlens = q_seqlens + history_lens
        max_seq_len = kv_seqlens.max().item()
        k = torch.rand(batch_size, max_seq_len, num_heads_k, head_dim_k, dtype=dtype, device='cuda')
        v = torch.rand(batch_size, max_seq_len, num_heads_k, head_dim_v, dtype=dtype, device='cuda')
        yield k, v

    @pytest.fixture
    def conti_q(self, q_seqlens, batched_q):
        yield _conti_input(batched_q, q_seqlens)

    @pytest.fixture
    def conti_kv(self, kv_seqlens, batched_kv):
        conti_k = _conti_input(batched_kv[0], kv_seqlens)
        conti_k = conti_k.transpose(0, 1).contiguous()
        conti_v = _conti_input(batched_kv[1], kv_seqlens)
        conti_v = conti_v.transpose(0, 1).contiguous()
        yield (conti_k, conti_v)

    @pytest.fixture
    def mask(self, q_seqlens, history_lens, causal):
        neg_val = -1e30
        yield _make_bias(q_seqlens, history_lens, neg_val, causal)

    @pytest.fixture
    def gt(self, batched_q, batched_kv, mask):
        yield _naive_attention(batched_q, batched_kv, mask)

    @pytest.fixture
    def conti_gt(self, gt, q_seqlens):
        yield _conti_input(gt, q_seqlens)

    @pytest.mark.parametrize('head_dim_k', [32, 48], indirect=True)
    @pytest.mark.parametrize('head_dim_v', [32], indirect=True)
    @pytest.mark.parametrize('num_heads_q', [8, 2], indirect=True)
    @pytest.mark.parametrize('num_heads_k', [2], indirect=True)
    @pytest.mark.parametrize('causal', [True, False], indirect=True)
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'], [([30, 50, 70, 90], [50, 40, 30, 20])], indirect=True)
    def test_flash_attention(self, conti_q, conti_kv, q_seqlens, cu_seqlens_q, cu_seqlens_k, causal, conti_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = flash_attn_varlen_func(conti_q,
                                     conti_k,
                                     conti_v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     max_seqlen_q=max_seq_len,
                                     causal=causal)
        torch.testing.assert_close(out, conti_gt, atol=1e-3, rtol=1e-5)

    @pytest.fixture
    def win_size(self, request):
        yield request.param

    @pytest.fixture
    def window_gt(self, conti_q, conti_kv, q_seqlens, kv_seqlens, win_size):
        conti_k, conti_v = conti_kv
        yield _naive_window_attention(conti_q,
                                      conti_k.transpose(0, 1),
                                      conti_v.transpose(0, 1),
                                      q_seqlens,
                                      kv_seqlens,
                                      window_size=(win_size, win_size))

    @pytest.mark.parametrize('head_dim_k', [16], indirect=True)
    @pytest.mark.parametrize('head_dim_v', [16], indirect=True)
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(4, 2)], indirect=True)
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'], [
        ([30, 50, 70, 90], [50, 40, 30, 90]),
    ], indirect=True)
    @pytest.mark.parametrize('win_size', (32, ), indirect=True)
    def test_window_attention(self, conti_q, conti_kv, q_seqlens, cu_seqlens_q, cu_seqlens_k, win_size, window_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = flash_attn_varlen_func(conti_q,
                                     conti_k,
                                     conti_v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     max_seqlen_q=max_seq_len,
                                     window_size=win_size,
                                     causal=True)
        torch.testing.assert_close(out, window_gt, atol=1e-3, rtol=1e-5)

    @pytest.fixture
    def sinks(self, num_heads_q, dtype):
        yield torch.rand(num_heads_q, dtype=dtype, device='cuda')

    @pytest.fixture
    def sink_gt(self, batched_q, batched_kv, mask, sinks):
        yield _naive_attention(batched_q, batched_kv, mask, sinks)

    @pytest.fixture
    def conti_sink_gt(self, sink_gt, q_seqlens):
        yield _conti_input(sink_gt, q_seqlens)

    @pytest.mark.parametrize('head_dim_k', [32], indirect=True)
    @pytest.mark.parametrize('head_dim_v', [32], indirect=True)
    @pytest.mark.parametrize('num_heads_q', [8], indirect=True)
    @pytest.mark.parametrize('num_heads_k', [2], indirect=True)
    @pytest.mark.parametrize('causal', [True], indirect=True)
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'], [([30, 50, 70, 90], [50, 40, 30, 20])], indirect=True)
    def test_sinks(self, conti_q, conti_kv, q_seqlens, cu_seqlens_q, cu_seqlens_k, causal, sinks, conti_sink_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = flash_attn_varlen_func(conti_q,
                                     conti_k,
                                     conti_v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     max_seqlen_q=max_seq_len,
                                     sinks=sinks,
                                     causal=causal)
        torch.testing.assert_close(out, conti_sink_gt, atol=1e-3, rtol=1e-5)

    # block sparse attention
    @pytest.fixture
    def block_sparse_size(self):
        yield 4

    @pytest.fixture
    def block_sparse_mask(self, q_seqlens, history_lens, block_sparse_size):
        neg_val = -1e30
        yield _make_block_sparse_bias(q_seqlens, history_lens, neg_val, block_sparse_size)

    @pytest.fixture
    def block_sparse_gt(self, batched_q, batched_kv, block_sparse_mask):
        yield _naive_attention(batched_q, batched_kv, block_sparse_mask)

    @pytest.mark.parametrize('head_dim_k', [32], indirect=True)
    @pytest.mark.parametrize('head_dim_v', [32], indirect=True)
    @pytest.mark.parametrize('num_heads_q', [8], indirect=True)
    @pytest.mark.parametrize('num_heads_k', [2], indirect=True)
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'], [([16, 32], [64, 8])], indirect=True)
    def test_block_sparse_attention(self, conti_q, conti_kv, q_seqlens, cu_seqlens_q, cu_seqlens_k, block_sparse_size,
                                    block_sparse_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = flash_attn_varlen_func(conti_q,
                                     conti_k,
                                     conti_v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     max_seqlen_q=max_seq_len,
                                     block_sparse_size=block_sparse_size,
                                     causal=True)
        gt = _conti_input(block_sparse_gt, q_seqlens)
        torch.testing.assert_close(out, gt, atol=1e-3, rtol=1e-5)

    @pytest.fixture
    def alibi_slopes(self, num_heads_q):
        yield torch.rand(num_heads_q, dtype=torch.float32, device='cuda')

    @pytest.fixture
    def alibi_bias(self, q_seqlens, history_lens, causal, alibi_slopes):
        neg_val = -1e30
        yield _make_bias_alibi(q_seqlens, history_lens, neg_val, causal, alibi_slopes)

    @pytest.fixture
    def alibi_gt(self, batched_q, batched_kv, alibi_bias):
        yield _naive_attention(batched_q, batched_kv, alibi_bias)

    @pytest.fixture
    def conti_alibi_gt(self, alibi_gt, q_seqlens):
        yield _conti_input(alibi_gt, q_seqlens)

    @pytest.mark.parametrize('head_dim_k', [128], indirect=True)
    @pytest.mark.parametrize('head_dim_v', [128], indirect=True)
    @pytest.mark.parametrize('num_heads_q', [40], indirect=True)
    @pytest.mark.parametrize('num_heads_k', [8], indirect=True)
    @pytest.mark.parametrize('causal', [True], indirect=True)
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'], [
        ([30, 50, 70, 90], [50, 40, 30, 20]),
    ], indirect=True)
    def test_alibi(self, conti_q, conti_kv, q_seqlens, cu_seqlens_q, cu_seqlens_k, causal, alibi_slopes,
                   conti_alibi_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import flash_attn_varlen_func
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = flash_attn_varlen_func(conti_q,
                                     conti_k,
                                     conti_v,
                                     cu_seqlens_q,
                                     cu_seqlens_k,
                                     max_seqlen_q=max_seq_len,
                                     alibi_slopes=alibi_slopes,
                                     causal=causal)
        torch.testing.assert_close(out, conti_alibi_gt, atol=1e-3, rtol=1e-5)
