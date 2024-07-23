import pytest
import torch

from lmdeploy.pytorch.kernels import apply_rotary_pos_emb


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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
        yield 16

    @pytest.fixture
    def seq_length(self, batch_size):
        yield torch.randint(8, 16, (batch_size, ), device='cuda')

    @pytest.fixture
    def max_seqlen(self, seq_length):
        yield seq_length.max()

    @pytest.fixture
    def q_states(self, seq_length, num_heads_q, feature_dim, dtype):
        yield torch.rand(seq_length.sum(),
                         num_heads_q,
                         feature_dim,
                         dtype=dtype,
                         device='cuda')

    @pytest.fixture
    def k_states(self, seq_length, num_heads_k, feature_dim, dtype):
        yield torch.rand(seq_length.sum(),
                         num_heads_k,
                         feature_dim,
                         dtype=dtype,
                         device='cuda')

    @pytest.fixture
    def position_ids_1d(self, seq_length, max_seqlen):
        yield torch.randint(0,
                            max_seqlen.item(), (seq_length.sum().item(), ),
                            device='cuda')

    @pytest.fixture
    def cached_cos(self, max_seqlen, feature_dim, dtype):
        yield torch.rand(max_seqlen, feature_dim, dtype=dtype, device='cuda')

    @pytest.fixture
    def cached_sin(self, max_seqlen, feature_dim, dtype):
        yield torch.rand(max_seqlen, feature_dim, dtype=dtype, device='cuda')

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

    @pytest.mark.parametrize('dtype',
                             [torch.bfloat16, torch.float16, torch.float32],
                             indirect=True)
    @pytest.mark.parametrize(('num_heads_q', 'num_heads_k'), [(8, 8), (8, 4)],
                             indirect=True)
    def test_apply_rotary(self, q_states, k_states, cos, sin, gt):
        q_embed, k_embed = apply_rotary_pos_emb(q_states, k_states, cos, sin)
        q_gt, k_gt = gt

        rtol = None
        atol = None
        if q_states.dtype == torch.float16:
            rtol = 1e-5
            atol = 1e-3
        torch.testing.assert_close(q_embed, q_gt, rtol=rtol, atol=atol)
        torch.testing.assert_close(k_embed, k_gt, rtol=rtol, atol=atol)
