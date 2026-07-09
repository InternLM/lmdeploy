from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.models.deepseek_v32 import (
    DeepseekV32Attention,
    DSATopKIndicesBuffer,
    Indexer,
    apply_interleaved_rotary_pos_emb,
    get_layer_indexer_type,
    rotate_interleaved,
)


def _patch_minimal_attention(attn, monkeypatch):
    """Patch a DSA attention module down to the top-k routing contract."""

    def fake_qkv_proj(hidden_states, num_heads):
        q_len = hidden_states.size(1)
        return (
            torch.zeros(q_len, 1, 2),
            torch.zeros(q_len, 1, 2),
            torch.zeros(q_len, 1, 1),
            torch.zeros(q_len, 1, 1),
            torch.zeros(q_len, 1, 1),
            torch.zeros(q_len, 1, 1),
        )

    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v32.get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
                            dist_config=SimpleNamespace(attn_tp=1))))
    attn.num_heads = 1
    attn.kv_lora_rank = 1
    attn.qk_rope_head_dim = 1
    attn._qkv_proj = fake_qkv_proj
    attn.apply_rotary_pos_emb = lambda q, k, cos, sin, inplace=False: (q, k)
    attn.vc = lambda attn_output, out: out.copy_(attn_output)
    attn.v_head_dim = 1
    attn.o_proj = nn.Identity()


def test_interleaved_rotary_uses_pairwise_layout_with_half_split_freqs():
    query = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    key = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])
    cos = torch.tensor([[10.0, 20.0, 10.0, 20.0]])
    sin = torch.tensor([[1.0, 2.0, 1.0, 2.0]])

    query_out, key_out = apply_interleaved_rotary_pos_emb(query, key, cos, sin)

    interleaved_cos = torch.tensor([[[10.0, 10.0, 20.0, 20.0]]])
    interleaved_sin = torch.tensor([[[1.0, 1.0, 2.0, 2.0]]])
    expected_query = query * interleaved_cos + rotate_interleaved(query) * interleaved_sin
    expected_key = key * interleaved_cos + rotate_interleaved(key) * interleaved_sin
    assert torch.equal(query_out, expected_query)
    assert torch.equal(key_out, expected_key)


def test_indexer_rope_interleave_uses_interleaved_layout():
    indexer = Indexer.__new__(Indexer)
    indexer.rope_interleave = True

    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    key = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    cos = torch.tensor([[10.0, 20.0, 10.0, 20.0]])
    sin = torch.tensor([[1.0, 2.0, 1.0, 2.0]])

    query_out, key_out = indexer._apply_rotary_pos_emb(query, key, (cos, sin))

    key = key[..., None, :]
    interleaved_cos = torch.tensor([[10.0, 10.0, 20.0, 20.0]]).unsqueeze(-2)
    interleaved_sin = torch.tensor([[1.0, 1.0, 2.0, 2.0]]).unsqueeze(-2)
    expected_query = query * interleaved_cos + rotate_interleaved(query) * interleaved_sin
    expected_key = key * interleaved_cos + rotate_interleaved(key) * interleaved_sin
    assert torch.equal(query_out, expected_query)
    assert torch.equal(key_out, expected_key)


def test_indexer_rope_interleave_can_fall_back_to_half_split_layout():
    indexer = Indexer.__new__(Indexer)
    indexer.rope_interleave = False
    indexer.apply_rotary_pos_emb = lambda q, k, cos, sin, inplace=False: ('half-split', q, k, cos, sin, inplace)

    query = torch.zeros(1, 1, 4)
    key = torch.zeros(1, 4)
    cos = torch.zeros(1, 4)
    sin = torch.zeros(1, 4)

    out = indexer._apply_rotary_pos_emb(query, key, (cos, sin))

    assert out[0] == 'half-split'
    assert out[2].shape == (1, 1, 4)
    assert out[-1] is False


def test_full_indexer_layer_writes_shared_topk_buffer(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 0
    _patch_minimal_attention(attn, monkeypatch)
    computed_topk = torch.tensor([[4, 2, 1], [3, 2, 0]], dtype=torch.int32)
    attn.indexer = lambda *args, **kwargs: computed_topk
    seen = {}

    def fake_attn_fwd(*args, **kwargs):
        seen['nsa_indices'] = kwargs['nsa_indices']
        return torch.zeros(args[0].size(0), 1, 1)

    attn.attn_fwd = fake_attn_fwd
    topk_buffer = DSATopKIndicesBuffer(topk=3)

    output = attn(
        hidden_states=torch.zeros(1, 2, 2),
        rotary_pos_emb=(torch.zeros(2, 1), torch.zeros(2, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        topk_indices_buffer=topk_buffer,
    )

    assert output.shape == (1, 2, 1)
    assert torch.equal(topk_buffer.indices[:2], computed_topk)
    assert seen['nsa_indices'].data_ptr() == topk_buffer.indices[:2].data_ptr()


def test_dsa_topk_buffer_compacts_selected_rows():
    topk_buffer = DSATopKIndicesBuffer(topk=3)
    rows = torch.tensor([[0, 1, 2], [10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    topk_buffer.write(rows)

    topk_buffer.compact(torch.tensor([2, 0]))

    assert torch.equal(topk_buffer.indices[:2], rows[[2, 0]])


def test_shared_indexer_layer_reuses_shared_topk_buffer(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 3
    attn.indexer = None
    _patch_minimal_attention(attn, monkeypatch)
    seen = {}

    def fake_attn_fwd(*args, **kwargs):
        seen['nsa_indices'] = kwargs['nsa_indices']
        return torch.zeros(args[0].size(0), 1, 1)

    attn.attn_fwd = fake_attn_fwd
    topk_buffer = DSATopKIndicesBuffer(topk=3)
    topk_buffer.write(torch.tensor([[5, 4, 3]], dtype=torch.int32))

    output = attn(
        hidden_states=torch.zeros(1, 1, 2),
        rotary_pos_emb=(torch.zeros(1, 1), torch.zeros(1, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        topk_indices_buffer=topk_buffer,
    )

    assert output.shape == (1, 1, 1)
    assert seen['nsa_indices'].data_ptr() == topk_buffer.indices[:1].data_ptr()


def test_shared_indexer_layer_requires_shared_topk_buffer(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 3
    attn.indexer = None
    attn.num_heads = 1
    attn.kv_lora_rank = 1
    attn.qk_rope_head_dim = 1
    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v32.get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
                            dist_config=SimpleNamespace(attn_tp=1))))
    attn._qkv_proj = lambda *args, **kwargs: (
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
        torch.zeros(1, 1, 1),
    )
    attn.apply_rotary_pos_emb = lambda q, k, cos, sin, inplace=False: (q, k)

    with pytest.raises(RuntimeError, match='reuses DSA top-k indices'):
        attn(
            hidden_states=torch.zeros(1, 1, 2),
            rotary_pos_emb=(torch.zeros(1, 1), torch.zeros(1, 1)),
            past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        )


def test_layer_indexer_type_defaults_to_full_and_reads_shared_entries():
    config = SimpleNamespace(indexer_types=['full', 'shared'])

    assert get_layer_indexer_type(config, 0) == 'full'
    assert get_layer_indexer_type(config, 1) == 'shared'
    assert get_layer_indexer_type(config, 2) == 'full'
    assert get_layer_indexer_type(SimpleNamespace(indexer_types=None), 1) == 'full'
