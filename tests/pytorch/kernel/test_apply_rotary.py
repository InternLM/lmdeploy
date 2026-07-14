import pytest
import torch

from lmdeploy.utils import is_bf16_supported


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_interleaved(x):
    """Rotate adjacent pairs of hidden dimensions."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


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

    @pytest.mark.parametrize('dtype', [pytest.param(torch.bfloat16, marks=_bf16_mark()), torch.float16, torch.float32],
                             indirect=True)
    @pytest.mark.parametrize(('num_heads_q', 'num_heads_k'), [(8, 8), (8, 4)], indirect=True)
    @pytest.mark.parametrize('inplace', [False, True])
    def test_apply_rotary_interleaved(self, q_states, k_states, cos, sin, inplace):
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

        half_size = cos.size(-1) // 2
        cos = torch.cat([cos[..., :half_size], cos[..., :half_size]], dim=-1)
        sin = torch.cat([sin[..., :half_size], sin[..., :half_size]], dim=-1)
        pair_cos = cos[..., :half_size].repeat_interleave(2, dim=-1)
        pair_sin = sin[..., :half_size].repeat_interleave(2, dim=-1)
        q_gt = q_states * pair_cos + _rotate_interleaved(q_states) * pair_sin
        k_gt = k_states * pair_cos + _rotate_interleaved(k_states) * pair_sin

        if inplace:
            q_states = q_states.clone()
            k_states = k_states.clone()
            q_embed, k_embed = apply_rotary_pos_emb(
                q_states, k_states, cos, sin, q_embed=q_states, k_embed=k_states, interleaved=True)
        else:
            q_embed, k_embed = apply_rotary_pos_emb(q_states, k_states, cos, sin, interleaved=True)

        assert torch.equal(q_embed, q_gt)
        assert torch.equal(k_embed, k_gt)

    @pytest.mark.parametrize('tokens', [1, 33, 257])
    def test_apply_rotary_interleaved_non_contiguous(self, tokens):
        from lmdeploy.pytorch.kernels.cuda import apply_rotary_pos_emb

        query = torch.randn(1, tokens, 32, 128, device='cuda', dtype=torch.bfloat16)[..., :64]
        key = torch.randn(1, tokens, 128, device='cuda', dtype=torch.bfloat16)[..., :64][..., None, :]
        positions = torch.linspace(0, 1048575, tokens, device='cuda')
        inv_freq = 1 / (8000000**(torch.arange(0, 64, 2, device='cuda').float() / 64))
        angles = positions[:, None] * inv_freq[None, :]
        cos = torch.cat([angles.cos(), angles.cos()], dim=-1).to(torch.bfloat16)
        sin = torch.cat([angles.sin(), angles.sin()], dim=-1).to(torch.bfloat16)
        pair_cos = cos[..., :32].repeat_interleave(2, dim=-1).unsqueeze(-2)
        pair_sin = sin[..., :32].repeat_interleave(2, dim=-1).unsqueeze(-2)
        q_gt = query * pair_cos + _rotate_interleaved(query) * pair_sin
        k_gt = key * pair_cos + _rotate_interleaved(key) * pair_sin

        q_embed, k_embed = apply_rotary_pos_emb(query, key, cos, sin, interleaved=True)

        assert torch.equal(q_embed, q_gt)
        assert torch.equal(k_embed, k_gt)
