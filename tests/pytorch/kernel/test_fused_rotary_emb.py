import pytest
import torch
from torch import nn

from lmdeploy.pytorch.kernels.fused_rotary_emb import fused_rotary_emb


class DummyRotaryEmbedding(nn.Module):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base**(torch.arange(
            0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, x, position_ids, seq_len=None):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        # backwards compatibility
        return cos, sin


class DummyLinearScalingRotaryEmbedding(DummyRotaryEmbedding):

    def __init__(self,
                 dim,
                 max_position_embeddings=2048,
                 base=10000,
                 device=None,
                 scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def forward(self, x, position_ids, seq_len=None):
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids, seq_len)
        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TestFusedRotaryEmb:

    @pytest.fixture
    def dtype(self):
        yield torch.float16

    @pytest.fixture
    def batch_size(self):
        yield 2

    @pytest.fixture
    def head_dim(self):
        yield 64

    @pytest.fixture
    def q_num_heads(self):
        yield 4

    @pytest.fixture
    def k_num_heads(self):
        yield 2

    @pytest.fixture
    def seq_len(self):
        yield 100

    @pytest.fixture
    def q(self, batch_size, seq_len, q_num_heads, head_dim, dtype):
        yield torch.rand(batch_size,
                         seq_len,
                         q_num_heads,
                         head_dim,
                         dtype=dtype).to('cuda')

    @pytest.fixture
    def k(self, batch_size, seq_len, k_num_heads, head_dim, dtype):
        yield torch.rand(batch_size,
                         seq_len,
                         k_num_heads,
                         head_dim,
                         dtype=dtype).to('cuda')

    @pytest.fixture
    def position_ids(self, batch_size, seq_len):
        yield torch.randint(0, seq_len + 100, (batch_size, seq_len)).cuda()

    @pytest.fixture
    def rotary_emb(self, head_dim):
        yield DummyLinearScalingRotaryEmbedding(head_dim,
                                                scaling_factor=1.0).to('cuda')

    @pytest.fixture
    def gt(self, q, k, position_ids, rotary_emb):
        with torch.inference_mode():
            cos, sin = rotary_emb(q, position_ids)
            yield apply_rotary_pos_emb(q,
                                       k,
                                       cos,
                                       sin,
                                       position_ids=position_ids)

    def test_fused_rotary_emb(self, q, k, position_ids, rotary_emb, gt):
        inv_freq = rotary_emb.inv_freq
        scaling_factor = rotary_emb.scaling_factor

        with torch.inference_mode():
            outq, outk = fused_rotary_emb(q,
                                          k,
                                          position_ids,
                                          inv_freq,
                                          scaling_factor=scaling_factor)

        gtq, gtk = gt
        torch.testing.assert_close(outq, gtq, atol=1e-3, rtol=1e-5)
        torch.testing.assert_close(outk, gtk, atol=1e-3, rtol=1e-5)
