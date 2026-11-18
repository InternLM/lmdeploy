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

    @pytest.mark.parametrize('tokens', [1, 33, 257])
    def test_apply_rotary_complex_non_contiguous(self, tokens):
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

        query = torch.randn(1, tokens, 32, 128, device='cuda', dtype=torch.bfloat16)[..., :64]
        key = torch.randn(1, tokens, 128, device='cuda', dtype=torch.bfloat16)[..., :64][..., None, :]
        positions = torch.linspace(0, 1048575, tokens, device='cuda')
        inv_freq = 1 / (8000000**(torch.arange(0, 64, 2, device='cuda').float() / 64))
        angles = positions[:, None] * inv_freq[None, :]
        cos = angles.cos().to(torch.bfloat16)
        sin = angles.sin().to(torch.bfloat16)
        pair_cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
        pair_sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        q_gt = query * pair_cos + _rotate_complex(query) * pair_sin
        k_gt = key * pair_cos + _rotate_complex(key) * pair_sin

        q_embed, k_embed = apply_rotary_pos_emb(query, key, cos, sin, complex_mode=True)

        assert torch.equal(q_embed, q_gt)
        assert torch.equal(k_embed, k_gt)

    @pytest.mark.parametrize('inplace', [False, True])
    @pytest.mark.parametrize('complex_mode', [False, True])
    def test_apply_rotary_partial_rotation(self, inplace, complex_mode):
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

        S, H, D, R = 4, 2, 64, 32
        q = torch.randn(S, H, D, dtype=torch.float16, device='cuda')
        k = torch.randn(S, 1, D, dtype=torch.float16, device='cuda')
        cos_dim = R // 2 if complex_mode else R
        cos = torch.randn(S, cos_dim, dtype=torch.float16, device='cuda')
        sin = torch.randn(S, cos_dim, dtype=torch.float16, device='cuda')

        q_rot = q[..., :R]
        k_rot = k[..., :R]
        if complex_mode:
            cos_full = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin_full = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
            q_rot_embed = q_rot * cos_full + _rotate_complex(q_rot) * sin_full
            k_rot_embed = k_rot * cos_full + _rotate_complex(k_rot) * sin_full
        else:
            cos_full = cos.unsqueeze(-2)
            sin_full = sin.unsqueeze(-2)
            q_rot_embed = q_rot * cos_full + _rotate_half(q_rot) * sin_full
            k_rot_embed = k_rot * cos_full + _rotate_half(k_rot) * sin_full

        q_gt = torch.cat([q_rot_embed, q[..., R:]], dim=-1)
        k_gt = torch.cat([k_rot_embed, k[..., R:]], dim=-1)

        if inplace:
            q_in, k_in = q.clone(), k.clone()
            qo, ko = apply_rotary_pos_emb(q_in, k_in, cos, sin, q_in, k_in, complex_mode=complex_mode)
        else:
            qo_buf = torch.full_like(q, float('nan'))
            ko_buf = torch.full_like(k, float('nan'))
            qo, ko = apply_rotary_pos_emb(q, k, cos, sin, qo_buf, ko_buf, complex_mode=complex_mode)

        torch.testing.assert_close(qo, q_gt)
        torch.testing.assert_close(ko, k_gt)
        assert torch.equal(qo[..., R:], q[..., R:])
        assert torch.equal(ko[..., R:], k[..., R:])
