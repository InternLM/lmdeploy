from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lmdeploy.pytorch.models.deepseek_v32 import (DeepseekV32Attention, Indexer, apply_interleaved_rotary_pos_emb,
                                                 get_layer_indexer_type, rotate_interleaved)


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


def test_shared_indexer_layer_reuses_previous_topk_indices(monkeypatch):
    attn = DeepseekV32Attention.__new__(DeepseekV32Attention)
    nn.Module.__init__(attn)
    attn.layer_idx = 3
    attn.indexer = None
    prev_topk_indices = torch.tensor([[3, 1, 0]], dtype=torch.int32)
    seen = {}

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

    def fake_rotary(q, k, cos, sin, inplace=False):
        return q, k

    def fake_attn_fwd(*args, **kwargs):
        seen['nsa_indices'] = kwargs['nsa_indices']
        return torch.zeros(1, 1, 1)

    monkeypatch.setattr('lmdeploy.pytorch.models.deepseek_v32.get_dist_manager',
                        lambda: SimpleNamespace(current_context=lambda: SimpleNamespace(
                            dist_config=SimpleNamespace(attn_tp=1))))
    attn.num_heads = 1
    attn.kv_lora_rank = 1
    attn.qk_rope_head_dim = 1
    attn._qkv_proj = fake_qkv_proj
    attn.apply_rotary_pos_emb = fake_rotary
    attn.attn_fwd = fake_attn_fwd
    attn.vc = lambda attn_output, out: out.copy_(attn_output)
    attn.v_head_dim = 1
    attn.o_proj = nn.Identity()

    output, returned_topk = attn(
        hidden_states=torch.zeros(1, 1, 2),
        rotary_pos_emb=(torch.zeros(1, 1), torch.zeros(1, 1)),
        past_key_value=[torch.zeros(1, 1, 2), torch.zeros(1, 1, 1)],
        prev_topk_indices=prev_topk_indices,
    )

    assert output.shape == (1, 1, 1)
    assert returned_topk is prev_topk_indices
    assert seen['nsa_indices'] is prev_topk_indices


def test_shared_indexer_layer_requires_previous_topk_indices(monkeypatch):
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
            prev_topk_indices=None,
        )


def test_layer_indexer_type_defaults_to_full_and_reads_shared_entries():
    config = SimpleNamespace(indexer_types=['full', 'shared'])

    assert get_layer_indexer_type(config, 0) == 'full'
    assert get_layer_indexer_type(config, 1) == 'shared'
    assert get_layer_indexer_type(config, 2) == 'full'
    assert get_layer_indexer_type(SimpleNamespace(indexer_types=None), 1) == 'full'
