from types import SimpleNamespace

import pytest

from lmdeploy.pytorch.configurations.deepseek_v32 import normalize_glm_moe_dsa_config


def _make_config(**kwargs):
    values = dict(
        model_type='glm_moe_dsa',
        num_hidden_layers=4,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
    )
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_glm_moe_dsa_normalizes_pattern_indexer_types():
    cfg = _make_config(index_topk_pattern='FSSF')

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.qk_head_dim == 192
    assert cfg.head_dim == 64
    assert cfg.indexer_types == ['full', 'shared', 'shared', 'full']


def test_glm_moe_dsa_normalizes_freq_offset_indexer_types():
    cfg = _make_config(num_hidden_layers=6, index_topk_freq=2, index_skip_topk_offset=2)

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.indexer_types == ['full', 'full', 'shared', 'full', 'shared', 'full']


def test_glm_moe_dsa_recovers_corrupted_rope_head_dim():
    cfg = _make_config(qk_nope_head_dim=192, qk_rope_head_dim=192, qk_head_dim=256)

    normalize_glm_moe_dsa_config(cfg)

    assert cfg.qk_nope_head_dim == 192
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_head_dim == 256
    assert cfg.head_dim == 64


def test_glm_moe_dsa_rejects_shared_first_layer():
    cfg = _make_config(index_topk_pattern='SFFF')

    with pytest.raises(ValueError, match='first GLM-MoE-DSA layer'):
        normalize_glm_moe_dsa_config(cfg)


def test_glm_moe_dsa_fallback_config_keeps_rope_head_dim():
    from lmdeploy.pytorch.transformers.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(qk_nope_head_dim=128,
                          qk_rope_head_dim=64,
                          hidden_size=4096,
                          intermediate_size=8192,
                          num_attention_heads=32,
                          num_hidden_layers=2,
                          vocab_size=32000,
                          bos_token_id=1,
                          eos_token_id=2)

    assert cfg.qk_nope_head_dim == 128
    assert cfg.qk_rope_head_dim == 64
    assert cfg.qk_head_dim == 192
    assert cfg.head_dim == 64
