import pytest
import torch

from lmdeploy.pytorch_poc.kernels import apply_rotary_pos_emb


def _rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class TestApplyRotary:

    @pytest.fixture(scope='class')
    def batch_size(self):
        yield 4

    @pytest.fixture(scope='class')
    def num_heads(self):
        yield 8

    @pytest.fixture(scope='class')
    def feature_dim(self):
        yield 16

    @pytest.fixture(scope='class')
    def seq_length(self, batch_size):
        yield torch.randint(8, 16, (batch_size, ), device='cuda')

    @pytest.fixture(scope='class')
    def max_seqlen(self, seq_length):
        yield seq_length.max()

    @pytest.fixture(scope='class')
    def q_states(self, seq_length, num_heads, feature_dim):
        yield torch.rand(seq_length.sum(),
                         num_heads,
                         feature_dim,
                         device='cuda')

    @pytest.fixture(scope='class')
    def k_states(self, seq_length, num_heads, feature_dim):
        yield torch.rand(seq_length.sum(),
                         num_heads,
                         feature_dim,
                         device='cuda')

    @pytest.fixture(scope='class')
    def position_ids_1d(self, seq_length, max_seqlen):
        yield torch.randint(0,
                            max_seqlen.item(), (seq_length.sum().item(), ),
                            device='cuda')

    @pytest.fixture(scope='class')
    def cached_cos(self, max_seqlen, feature_dim):
        yield torch.rand(max_seqlen, feature_dim, device='cuda')

    @pytest.fixture(scope='class')
    def cached_sin(self, max_seqlen, feature_dim):
        yield torch.rand(max_seqlen, feature_dim, device='cuda')

    @pytest.fixture(scope='class')
    def gt(self, q_states, k_states, cached_cos, cached_sin, position_ids_1d):
        cos = cached_cos[position_ids_1d, None, :]
        sin = cached_sin[position_ids_1d, None, :]

        q_embed = q_states * cos + _rotate_half(q_states) * sin
        k_embed = k_states * cos + _rotate_half(k_states) * sin

        yield q_embed, k_embed

    def test_apply_rotary(self, q_states, k_states, cached_cos, cached_sin,
                          position_ids_1d, gt):
        q_embed, k_embed = apply_rotary_pos_emb(q_states, k_states, cached_cos,
                                                cached_sin, None,
                                                position_ids_1d)
        q_gt, k_gt = gt

        torch.testing.assert_close(q_embed, q_gt)
        torch.testing.assert_close(k_embed, k_gt)
