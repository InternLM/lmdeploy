import pytest
import torch

from lmdeploy.utils import is_bf16_supported


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_complex(x):
    """Rotates adjacent element pairs for complex-number RoPE."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def _bf16_mark():
    return pytest.mark.skipif(not is_bf16_supported(), reason='bf16 not supported.')


class TestApplyRotary:

    @pytest.fixture
    def dtype(self, request):
        yield request.param

    @pytest.fixture
    def batch_size(self):
        yield 4

    @pytest.fixture
    def num_heads_q(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_k(self, request):
        yield request.param

    @pytest.fixture
    def feature_dim(self):
        yield 128

    @pytest.fixture
    def seq_length(self, batch_size):
        yield torch.randint(8, 16, (batch_size, ), device='cuda')

    @pytest.fixture
    def max_seqlen(self, seq_length):
        yield seq_length.max()

    @pytest.fixture
    def q_states(self, seq_length, num_heads_q, feature_dim, dtype):
        yield torch.randn(seq_length.sum(), num_heads_q, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def k_states(self, seq_length, num_heads_k, feature_dim, dtype):
        yield torch.randn(seq_length.sum(), num_heads_k, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def position_ids_1d(self, seq_length, max_seqlen):
        yield torch.randint(0, max_seqlen.item(), (seq_length.sum().item(), ), device='cuda')

    @pytest.fixture
    def cached_cos(self, max_seqlen, feature_dim, dtype):
        yield torch.randn(max_seqlen, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def cached_sin(self, max_seqlen, feature_dim, dtype):
        yield torch.randn(max_seqlen, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def cos(self, cached_cos, position_ids_1d):
        yield cached_cos[position_ids_1d, None, :]

    @pytest.fixture
    def sin(self, cached_sin, position_ids_1d):
        yield cached_sin[position_ids_1d, None, :]

    @pytest.fixture
    def gt(self, q_states, k_states, cos, sin, position_ids_1d):

        q_embed = q_states * cos + _rotate_half(q_states) * sin
        k_embed = k_states * cos + _rotate_half(k_states) * sin

        yield q_embed, k_embed

    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16, torch.float32],
                             indirect=True)
    @pytest.mark.parametrize(('num_heads_q', 'num_heads_k'), [(8, 8), (8, 4)], indirect=True)
    def test_apply_rotary(self, q_states, k_states, cos, sin, gt):
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb
        q_embed, k_embed = apply_rotary_pos_emb(q_states, k_states, cos, sin)
        q_gt, k_gt = gt

        rtol = None
        atol = None
        torch.testing.assert_close(q_embed, q_gt, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_embed, k_gt, rtol=rtol, atol=atol)


class TestApplyRotaryComplex:

    @pytest.fixture
    def dtype(self, request):
        yield request.param

    @pytest.fixture
    def batch_size(self):
        yield 4

    @pytest.fixture
    def num_heads_q(self, request):
        yield request.param

    @pytest.fixture
    def num_heads_k(self, request):
        yield request.param

    @pytest.fixture
    def feature_dim(self):
        yield 128

    @pytest.fixture
    def seq_length(self, batch_size):
        yield torch.randint(8, 16, (batch_size, ), device='cuda')

    @pytest.fixture
    def max_seqlen(self, seq_length):
        yield seq_length.max()

    @pytest.fixture
    def q_states(self, seq_length, num_heads_q, feature_dim, dtype):
        yield torch.randn(seq_length.sum(), num_heads_q, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def k_states(self, seq_length, num_heads_k, feature_dim, dtype):
        yield torch.randn(seq_length.sum(), num_heads_k, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def position_ids_1d(self, seq_length, max_seqlen):
        yield torch.randint(0, max_seqlen.item(), (seq_length.sum().item(), ), device='cuda')

    @pytest.fixture
    def cached_cos(self, max_seqlen, feature_dim, dtype):
        # complex mode: cos/sin are half the feature dim
        yield torch.randn(max_seqlen, feature_dim // 2, dtype=dtype, device='cuda')

    @pytest.fixture
    def cached_sin(self, max_seqlen, feature_dim, dtype):
        yield torch.randn(max_seqlen, feature_dim // 2, dtype=dtype, device='cuda')

    @pytest.fixture
    def cos(self, cached_cos, position_ids_1d):
        yield cached_cos[position_ids_1d, None, :]

    @pytest.fixture
    def sin(self, cached_sin, position_ids_1d):
        yield cached_sin[position_ids_1d, None, :]

    @pytest.fixture
    def gt(self, q_states, k_states, cos, sin, position_ids_1d):
        # complex mode: expand cos/sin from (seq, dim//2) to (seq, 1, dim) by
        # duplicating each value for its adjacent pair: [c0,c0,c1,c1,...]
        cos_full = cos.squeeze(1).repeat_interleave(2, dim=-1).unsqueeze(-2)  # [seq, 1, dim]
        sin_full = sin.squeeze(1).repeat_interleave(2, dim=-1).unsqueeze(-2)  # [seq, 1, dim]
        q_embed = q_states * cos_full + _rotate_complex(q_states) * sin_full
        k_embed = k_states * cos_full + _rotate_complex(k_states) * sin_full
        yield q_embed, k_embed

    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16, torch.float32],
                             indirect=True)
    @pytest.mark.parametrize(('num_heads_q', 'num_heads_k'), [(8, 8), (8, 4)], indirect=True)
    def test_apply_rotary_complex(self, q_states, k_states, cos, sin, gt):
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb
        q_embed, k_embed = apply_rotary_pos_emb(q_states, k_states, cos, sin, complex_mode=True)
        q_gt, k_gt = gt

        rtol = None
        atol = None
        torch.testing.assert_close(q_embed, q_gt, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_embed, k_gt, rtol=rtol, atol=atol)
