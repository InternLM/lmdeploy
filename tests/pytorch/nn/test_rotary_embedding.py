import pytest
import torch
from transformers import PretrainedConfig

from lmdeploy.pytorch.nn import build_rotary_embedding_from_config


def _rotate_complex(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def _complex_rope_reference(x, cos, sin):
    cos_full = cos.repeat_interleave(2, dim=-1)
    sin_full = sin.repeat_interleave(2, dim=-1)
    if cos_full.dim() == x.dim() - 1:
        cos_full = cos_full.unsqueeze(-2)
    if sin_full.dim() == x.dim() - 1:
        sin_full = sin_full.unsqueeze(-2)
    return x * cos_full + _rotate_complex(x) * sin_full


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

    assert rotary_emb._uses_static_inv_freq_rope()
    torch.testing.assert_close(cos, expected_cos, rtol=0, atol=1e-7)
    torch.testing.assert_close(sin, expected_sin, rtol=0, atol=1e-7)


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


def test_interleaved_mrope_yarn_long_context_override_uses_backend_path():
    config = PretrainedConfig(
        hidden_size=2048,
        num_attention_heads=8,
        head_dim=256,
        max_position_embeddings=512000,
        rope_theta=10000000,
        rope_parameters=dict(
            mrope_interleaved=True,
            mrope_section=[11, 11, 10],
            rope_type='yarn',
            rope_theta=10000000,
            partial_rotary_factor=0.25,
            factor=4.0,
            original_max_position_embeddings=262144,
        ),
    )
    rotary_emb = build_rotary_embedding_from_config(config)
    position_ids = torch.tensor([
        [0, 1, 1024, 262143, 262144, 400000, 511998, 511999],
        [3, 5, 2048, 262140, 262150, 399900, 510000, 511999],
        [7, 11, 4096, 262130, 262160, 399800, 509000, 511999],
    ]).unsqueeze(1)
    hidden_states = torch.empty(1, position_ids.shape[-1], config.head_dim)

    cos, sin = rotary_emb(hidden_states, position_ids)
    leading_shape = position_ids.shape[:-1]
    base_cos, base_sin = rotary_emb.impl(hidden_states, position_ids.flatten(0, -2))
    base_cos = base_cos.reshape(*leading_shape, *base_cos.shape[1:])
    base_sin = base_sin.reshape(*leading_shape, *base_sin.shape[1:])

    def apply_interleaved_reference(freqs):
        half_dim = freqs.size(-1) // 2
        out = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = min(config.rope_parameters['mrope_section'][dim] * 3, half_dim)
            out[..., offset:length:3] = freqs[dim, ..., offset:length:3]
            out[..., half_dim + offset:half_dim + length:3] = \
                freqs[dim, ..., half_dim + offset:half_dim + length:3]
        return out

    assert not rotary_emb._uses_static_inv_freq_rope()
    assert cos.shape == (1, position_ids.shape[-1], 64)
    assert sin.shape == (1, position_ids.shape[-1], 64)
    torch.testing.assert_close(cos, apply_interleaved_reference(base_cos), rtol=0, atol=0)
    torch.testing.assert_close(sin, apply_interleaved_reference(base_sin), rtol=0, atol=0)


def test_mrope_config_keeps_text_positions_as_regular_rope():
    rotary_emb = build_rotary_embedding_from_config(_make_config(mrope_interleaved=True))
    hidden_states = torch.empty(5, 16)
    position_ids = torch.arange(5).unsqueeze(0)

    cos, sin = rotary_emb(hidden_states, position_ids)
    expected_cos, expected_sin = rotary_emb.impl(hidden_states, position_ids)

    torch.testing.assert_close(cos, expected_cos)
    torch.testing.assert_close(sin, expected_sin)


def test_default_apply_rotary_complex_accepts_half_width_tables():
    from lmdeploy.pytorch.backends.default.apply_rotary_emb import DefaultApplyRotaryEmbImpl

    q_states = torch.randn(5, 3, 8)
    k_states = torch.randn(5, 2, 8)
    cos = torch.randn(5, 4)
    sin = torch.randn(5, 4)

    q_embed, k_embed = DefaultApplyRotaryEmbImpl().forward(q_states, k_states, cos, sin, inplace=False,
                                                           complex_mode=True)

    torch.testing.assert_close(q_embed, _complex_rope_reference(q_states, cos, sin))
    torch.testing.assert_close(k_embed, _complex_rope_reference(k_states, cos, sin))


def test_default_apply_rotary_complex_accepts_half_width_tables_with_empty_key():
    from lmdeploy.pytorch.backends.default.apply_rotary_emb import DefaultApplyRotaryEmbImpl

    q_states = torch.randn(5, 3, 8)
    k_states = torch.empty(5, 0, 8)
    cos = torch.randn(5, 4)
    sin = torch.randn(5, 4)

    q_embed, k_embed = DefaultApplyRotaryEmbImpl().forward(q_states, k_states, cos, sin, inplace=False,
                                                           complex_mode=True)

    torch.testing.assert_close(q_embed, _complex_rope_reference(q_states, cos, sin))
    assert k_embed.shape == k_states.shape


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason='requires CUDA')),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('enable_fp32_compute', [False, True])
@pytest.mark.parametrize('inplace', [False, True])
@pytest.mark.parametrize('complex_mode', [False, True])
def test_apply_rotary_compute_precision(monkeypatch, dtype, device, enable_fp32_compute, inplace, complex_mode):
    from lmdeploy.pytorch.backends.default.op_backend import DefaultOpsBackend
    from lmdeploy.pytorch.nn import rotary_embedding

    if device == 'cpu':
        monkeypatch.setattr(rotary_embedding, 'get_backend', lambda: DefaultOpsBackend)
    module = rotary_embedding.ApplyRotaryEmb(enable_fp32_compute=enable_fp32_compute)

    generator = torch.Generator().manual_seed(123)
    # Unequal head counts and strided Q/K exercise the fused CUDA kernel.
    query = torch.randn(33, 3, 32, generator=generator).to(device=device, dtype=dtype)[..., ::2]
    key = torch.randn(33, 2, 32, generator=generator).to(device=device, dtype=dtype)[..., ::2]
    # FP32 tables also catch accidental downcasting before FP32 arithmetic.
    cos_value = 0.625 + (2**-12 if enable_fp32_compute else 0)
    sin_value = 0.375 + (2**-13 if enable_fp32_compute else 0)
    table_dtype = torch.float32 if enable_fp32_compute else dtype
    table_dim = 8 if complex_mode else 16
    cos = torch.full((33, table_dim), cos_value, dtype=table_dtype, device=device)
    sin = torch.full((33, table_dim), sin_value, dtype=table_dtype, device=device)
    original = (query.clone(), key.clone())
    outputs = module(query, key, cos, sin, inplace=inplace, complex_mode=complex_mode)

    for value, saved, actual in zip((query, key), original, outputs):
        inputs = saved.float() if enable_fp32_compute else saved
        if complex_mode:
            rotated = _rotate_complex(inputs)
        else:
            left, right = inputs.chunk(2, dim=-1)
            rotated = torch.cat((-right, left), dim=-1)
        expected = (inputs * cos_value + rotated * sin_value).to(dtype)
        assert actual.dtype == dtype
        torch.testing.assert_close(actual, expected,
                                   rtol=1e-6 if dtype == torch.float32 else 0,
                                   atol=1e-7 if dtype == torch.float32 else 0)
        if inplace:
            assert actual is value
        else:
            torch.testing.assert_close(value, saved, rtol=0, atol=0)


def test_dlinfer_rejects_fp32_rotary():
    from lmdeploy.pytorch.backends.apply_rotary_emb import ApplyRotaryEmbBuildSpec
    from lmdeploy.pytorch.backends.dlinfer.op_backend import DlinferOpsBackend

    with pytest.raises(NotImplementedError, match='enable_fp32_compute=True'):
        DlinferOpsBackend.build_op(ApplyRotaryEmbBuildSpec(enable_fp32_compute=True))


def test_glm_vision_uses_common_fp32_rotary():
    from lmdeploy.pytorch.models.glm5_next import Glm5NextVisionAttention
    from lmdeploy.pytorch.nn import ApplyRotaryEmb

    config = PretrainedConfig(hidden_size=128, num_heads=2, attention_bias=False)
    module = Glm5NextVisionAttention(config, dtype=torch.bfloat16, device='cpu')
    assert type(module.apply_rotary_pos_emb) is ApplyRotaryEmb
    assert module.apply_rotary_pos_emb.impl.enable_fp32_compute
