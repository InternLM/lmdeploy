import math

import pytest
import torch


def _conti_input(data, q_seqlens):
    data = [x[:l] for x, l in zip(data, q_seqlens)]
    data = torch.cat(data, dim=0)
    return data


def _make_bias(q_seqlens, history_lens, neg_val):
    full_seq_lens = q_seqlens + history_lens
    max_seq_len = q_seqlens.max().item()
    max_full_len = full_seq_lens.max().item()
    seq_ranges = [torch.arange(max_seq_len) for _ in q_seqlens]
    for r, l in zip(seq_ranges, q_seqlens):
        r[l:] = -max_full_len
    seq_ranges = torch.stack(seq_ranges, dim=0).cuda()
    kv_ranges = [torch.arange(max_full_len) for _ in full_seq_lens]
    kv_ranges = torch.stack(kv_ranges, 0).cuda()
    mask = kv_ranges[:, None, :] - seq_ranges[:, :, None] > history_lens[:,
                                                                         None,
                                                                         None]
    return mask.float() * neg_val


def _naive_attention(batched_q, batched_kv, bias):
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
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
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
    def q_seqlens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def q_start_loc(self, q_seqlens):
        yield q_seqlens.cumsum(0) - q_seqlens

    @pytest.fixture
    def history_lens(self, request):
        yield torch.tensor(request.param, device='cuda')

    @pytest.fixture
    def kv_seqlens(self, q_seqlens, history_lens):
        yield q_seqlens + history_lens

    @pytest.fixture
    def kv_start_loc(self, kv_seqlens):
        yield kv_seqlens.cumsum(0) - kv_seqlens

    @pytest.fixture
    def batched_q(self, q_seqlens, num_heads_q, head_dim_k, dtype):
        torch.manual_seed(123)
        batch_size = len(q_seqlens)
        max_seq_len = q_seqlens.max().item()
        inputs = torch.rand(batch_size,
                            max_seq_len,
                            num_heads_q,
                            head_dim_k,
                            dtype=dtype,
                            device='cuda')
        yield inputs

    @pytest.fixture
    def batched_kv(self, q_seqlens, history_lens, num_heads_k, head_dim_k,
                   head_dim_v, dtype):
        torch.manual_seed(123)
        batch_size = len(q_seqlens)
        full_seq_lens = q_seqlens + history_lens
        max_seq_len = full_seq_lens.max().item()
        k = torch.rand(batch_size,
                       max_seq_len,
                       num_heads_k,
                       head_dim_k,
                       dtype=dtype,
                       device='cuda')
        v = torch.rand(batch_size,
                       max_seq_len,
                       num_heads_k,
                       head_dim_v,
                       dtype=dtype,
                       device='cuda')
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
    def mask(self, q_seqlens, history_lens):
        neg_val = -1e30
        yield _make_bias(q_seqlens, history_lens, neg_val)

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
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'],
                             [([30, 50, 70, 90], [50, 40, 30, 20])],
                             indirect=True)
    def test_flash_attention(self, conti_q, conti_kv, q_start_loc, q_seqlens,
                             kv_start_loc, kv_seqlens, head_dim_v, conti_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import \
            flash_attention_fwd
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], head_dim_v)
        flash_attention_fwd(conti_q,
                            conti_k,
                            conti_v,
                            out,
                            q_start_loc=q_start_loc,
                            q_seqlens=q_seqlens,
                            kv_start_loc=kv_start_loc,
                            kv_seqlens=kv_seqlens,
                            max_seqlen=max_seq_len)
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
    @pytest.mark.parametrize(['num_heads_q', 'num_heads_k'], [(4, 2)],
                             indirect=True)
    @pytest.mark.parametrize(['q_seqlens', 'history_lens'], [
        ([30, 50, 70, 90], [50, 40, 30, 90]),
    ],
                             indirect=True)
    @pytest.mark.parametrize('win_size', (32, ), indirect=True)
    def test_window_attention(self, conti_q, conti_kv, q_start_loc, q_seqlens,
                              kv_start_loc, kv_seqlens, head_dim_v, win_size,
                              window_gt):
        from lmdeploy.pytorch.kernels.cuda.flashattention import \
            flash_attention_fwd
        max_seq_len = q_seqlens.max().item()

        conti_k, conti_v = conti_kv
        out = conti_q.new_empty(*conti_q.shape[:-1], head_dim_v)
        flash_attention_fwd(conti_q,
                            conti_k,
                            conti_v,
                            out,
                            q_start_loc=q_start_loc,
                            q_seqlens=q_seqlens,
                            kv_start_loc=kv_start_loc,
                            kv_seqlens=kv_seqlens,
                            max_seqlen=max_seq_len,
                            window_size=win_size)
        torch.testing.assert_close(out, window_gt, atol=1e-3, rtol=1e-5)
