import torch
from transformers import PretrainedConfig

from lmdeploy.pytorch.nn import build_rotary_embedding_from_config


def _make_config(*, mrope_interleaved: bool = False):
    return PretrainedConfig(
        hidden_size=16,
        num_attention_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        rope_theta=10000,
        rope_scaling=dict(
            rope_type='default',
            mrope_section=[2, 3, 3],
            mrope_interleaved=mrope_interleaved,
        ),
    )


def test_mrope_uses_rope_parameters_partial_rotary_factor():
    config = PretrainedConfig(
        hidden_size=16,
        num_attention_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        rope_parameters=dict(
            rope_type='default',
            rope_theta=10000,
            partial_rotary_factor=0.5,
            mrope_section=[1, 1, 2],
            mrope_interleaved=True,
        ),
    )
    rotary_emb = build_rotary_embedding_from_config(config)
    hidden_states = torch.empty(5, 16)
    position_ids = torch.stack([
        torch.arange(5),
        torch.arange(10, 15),
        torch.arange(20, 25),
    ])

    cos, sin = rotary_emb(hidden_states, position_ids)

    assert cos.shape == (5, 8)
    assert sin.shape == (5, 8)


def test_chunked_mrope_matches_legacy_selection():
    rotary_emb = build_rotary_embedding_from_config(_make_config())
    hidden_states = torch.empty(5, 16)
    position_ids = torch.stack([
        torch.arange(5),
        torch.arange(10, 15),
        torch.arange(20, 25),
    ])

    cos, sin = rotary_emb(hidden_states, position_ids)
    base_cos, base_sin = rotary_emb.impl(hidden_states, position_ids)
    mrope_section = [2, 3, 3] * 2
    expected_cos = torch.cat([m[i % 3] for i, m in enumerate(base_cos.split(mrope_section, dim=-1))], dim=-1)
    expected_sin = torch.cat([m[i % 3] for i, m in enumerate(base_sin.split(mrope_section, dim=-1))], dim=-1)

    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_interleaved_mrope_matches_qwen3_selection():
    rotary_emb = build_rotary_embedding_from_config(_make_config(mrope_interleaved=True))
    hidden_states = torch.empty(5, 16)
    position_ids = torch.stack([
        torch.arange(5),
        torch.arange(10, 15),
        torch.arange(20, 25),
    ])

    cos, sin = rotary_emb(hidden_states, position_ids)
    base_cos, base_sin = rotary_emb.impl(hidden_states, position_ids)

    def apply_interleaved(freqs):
        half_dim = freqs.size(-1) // 2
        out = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            half_dim = freqs.size(-1) // 2
            length = min([2, 3, 3][dim] * 3, half_dim)
            out[..., offset:length:3] = freqs[dim, ..., offset:length:3]
            out[..., half_dim + offset:half_dim + length:3] = \
                freqs[dim, ..., half_dim + offset:half_dim + length:3]
        return out

    torch.testing.assert_close(cos, apply_interleaved(base_cos))
    torch.testing.assert_close(sin, apply_interleaved(base_sin))


def test_interleaved_mrope_matches_legacy_qwen3_formula_tightly():
    rotary_emb = build_rotary_embedding_from_config(_make_config(mrope_interleaved=True))
    hidden_states = torch.empty(1, 17, 16)
    position_ids = torch.stack([
        torch.arange(17),
        torch.arange(101, 118),
        torch.arange(1001, 1018),
    ]).unsqueeze(1)

    cos, sin = rotary_emb(hidden_states, position_ids)
    inv_freq = rotary_emb.impl.inv_freq
    inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
    position_ids_expanded = position_ids[:, :, None, :].float()

    freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
    freqs_t = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = [2, 3, 3][dim] * 3
        freqs_t[..., offset:length:3] = freqs[dim, ..., offset:length:3]

    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    expected_cos = emb.cos()
    expected_sin = emb.sin()

    assert (cos - expected_cos).abs().max().item() <= 2e-7
    assert (sin - expected_sin).abs().max().item() <= 2e-7


def test_mrope_config_keeps_text_positions_as_regular_rope():
    rotary_emb = build_rotary_embedding_from_config(_make_config(mrope_interleaved=True))
    hidden_states = torch.empty(5, 16)
    position_ids = torch.arange(5).unsqueeze(0)

    cos, sin = rotary_emb(hidden_states, position_ids)
    expected_cos, expected_sin = rotary_emb.impl(hidden_states, position_ids)

    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_mrope_position_ids_pad_to_hidden_state_length():
    rotary_emb = build_rotary_embedding_from_config(_make_config())
    hidden_states = torch.empty(8, 16)
    position_ids = torch.stack([
        torch.arange(5),
        torch.arange(10, 15),
        torch.arange(20, 25),
    ])

    cos, sin = rotary_emb(hidden_states, position_ids)
    padded_position_ids = torch.zeros(3, 8, dtype=position_ids.dtype)
    padded_position_ids[:, :5] = position_ids
    expected_cos, expected_sin = rotary_emb(hidden_states, padded_position_ids)

    assert cos.shape == (8, 16)
    assert sin.shape == (8, 16)
    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)
