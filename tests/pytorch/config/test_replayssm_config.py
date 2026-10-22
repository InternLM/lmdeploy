from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import KVTransferConfig, PytorchEngineConfig
from lmdeploy.pytorch.config import SpecDecodeConfig
from lmdeploy.pytorch.configurations.qwen3_next import configure_gdn_replayssm
from lmdeploy.pytorch.consts import (
    GDN_CONV_STATE_CACHE_NAME,
    GDN_REPLAY_BETA_CACHE_NAME,
    GDN_REPLAY_CHECKPOINT_CACHE_NAME,
    GDN_REPLAY_CURSOR_CACHE_NAME,
    GDN_REPLAY_D_CACHE_NAME,
    GDN_REPLAY_G_CACHE_NAME,
    GDN_REPLAY_K_CACHE_NAME,
    GDN_REPLAY_RAW_K_CACHE_NAME,
    GDN_REPLAY_RAW_V_CACHE_NAME,
)
from lmdeploy.pytorch.engine.config_builder import ConfigBuilder
from lmdeploy.pytorch.engine.executor import (
    _validate_replayssm_preallocation,
    build_executor,
)


def _hf_config():
    return SimpleNamespace(
        layer_types=['linear_attention', 'full_attention', 'linear_attention'],
        linear_key_head_dim=64,
        linear_value_head_dim=128,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_conv_kernel_dim=4,
    )


def _cfg():
    return SimpleNamespace(states_shapes=[('legacy', torch.bfloat16)],
                           state_cache_specs=[],
                           enable_replayssm=False,
                           replayssm_buffer_len=16)


def test_public_replayssm_config_and_cache_propagation():
    engine_config = PytorchEngineConfig(enable_replayssm=True, replayssm_buffer_len=32)
    cache_config = ConfigBuilder.build_cache_config(engine_config)

    assert cache_config.enable_replayssm
    assert cache_config.replayssm_buffer_len == 32


def test_public_standard_replayssm_accepts_supported_preallocation_config():
    cache_config = ConfigBuilder.build_cache_config(
        PytorchEngineConfig(enable_replayssm=True))
    _validate_replayssm_preallocation(cache_config)

    disabled = ConfigBuilder.build_cache_config(
        PytorchEngineConfig(enable_replayssm=False))
    _validate_replayssm_preallocation(disabled)


def test_public_mtp_replayssm_accepts_supported_preallocation_config():
    cache_config = ConfigBuilder.build_cache_config(
        PytorchEngineConfig(enable_replayssm=True, replayssm_buffer_len=8))
    spec_config = SpecDecodeConfig(model='', method='qwen3_5_mtp',
                                   num_speculative_tokens=3)

    _validate_replayssm_preallocation(cache_config, spec_config)


@pytest.mark.parametrize('connector', ['CustomConnector'])
@pytest.mark.parametrize('role', ['kv_producer', 'kv_consumer', 'kv_both'])
@pytest.mark.parametrize('method', [None, 'qwen3_5_mtp', 'deepseek_mtp'])
def test_replayssm_rejects_external_connector_before_model_loading(monkeypatch, connector, role, method):
    cache_config = ConfigBuilder.build_cache_config(PytorchEngineConfig(
        enable_replayssm=True,
        kv_transfer_config=KVTransferConfig(kv_connector=connector, kv_role=role)))
    spec_config = None if method is None else SpecDecodeConfig(
        model='', method=method, num_speculative_tokens=3)

    def unexpected_model_load(*args, **kwargs):
        pytest.fail('connector must be rejected before reading model config or allocating caches')

    monkeypatch.setattr('lmdeploy.pytorch.engine.executor.ModelConfig.from_pretrained', unexpected_model_load)
    with pytest.raises(ValueError, match='ReplaySSM does not support external KV-cache connectors'):
        build_executor('model-must-not-be-read', cache_config, None,
                       SimpleNamespace(dp=1, world_size=1), None, specdecode_config=spec_config)


@pytest.mark.parametrize('connector', ['MooncakeStoreConnector', 'CustomConnector'])
@pytest.mark.parametrize('role', ['kv_producer', 'kv_consumer', 'kv_both'])
def test_flag_off_does_not_restrict_external_connectors(connector, role):
    cache_config = ConfigBuilder.build_cache_config(PytorchEngineConfig(
        enable_replayssm=False,
        kv_transfer_config=KVTransferConfig(kv_connector=connector, kv_role=role)))
    _validate_replayssm_preallocation(cache_config)
    assert cache_config.kv_transfer_config.is_kv_transfer_instance


@pytest.mark.parametrize('method', [None, 'qwen3_5_mtp', 'deepseek_mtp'])
def test_replayssm_accepts_empty_transfer_config(method):
    cache_config = ConfigBuilder.build_cache_config(PytorchEngineConfig(
        enable_replayssm=True, kv_transfer_config=KVTransferConfig()))
    spec_config = None if method is None else SpecDecodeConfig(
        model='', method=method, num_speculative_tokens=3)
    _validate_replayssm_preallocation(cache_config, spec_config)


@pytest.mark.parametrize('method', [None, 'deepseek_mtp'])
def test_replayssm_rejects_model_without_gdn_or_kda(monkeypatch, method):
    cache_config = ConfigBuilder.build_cache_config(PytorchEngineConfig(enable_replayssm=True))
    spec_config = None if method is None else SpecDecodeConfig(
        model='', method=method, num_speculative_tokens=3)
    # DSV4 and other unsupported families must not be admitted just because
    # their cache has state or their speculative method is deepseek_mtp.
    monkeypatch.setattr('lmdeploy.pytorch.engine.executor.ModelConfig.from_pretrained',
                        lambda *args, **kwargs: SimpleNamespace(enable_replayssm=False))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(ValueError, match='supported Gated DeltaNet or GLM-5.3 KDA'):
        build_executor('unsupported-model', cache_config, None,
                       SimpleNamespace(dp=1, world_size=1),
                       SimpleNamespace(hf_overrides=None, model_format=None), specdecode_config=spec_config)


@pytest.mark.parametrize('method', ['hy3_mtp', 'medusa'])
def test_mtp_replayssm_rejects_unsupported_speculative_method(method):
    cache_config = ConfigBuilder.build_cache_config(
        PytorchEngineConfig(enable_replayssm=True, replayssm_buffer_len=16))
    spec_config = SpecDecodeConfig(model='', method=method,
                                   num_speculative_tokens=3)

    with pytest.raises(ValueError, match='supports only qwen3_5_mtp'):
        build_executor('model-must-not-be-read', cache_config, None,
                       SimpleNamespace(dp=1, world_size=1), None,
                       specdecode_config=spec_config)


@pytest.mark.parametrize(('buffer_len', 'num_speculative_tokens'), [
    (8, 4),
    (8, 15),
    (16, 8),
    (32, 16),
])
def test_mtp_replayssm_rejects_short_buffer_before_allocation(
        buffer_len, num_speculative_tokens):
    cache_config = ConfigBuilder.build_cache_config(
        PytorchEngineConfig(enable_replayssm=True,
                            replayssm_buffer_len=buffer_len))
    spec_config = SpecDecodeConfig(
        model='', method='qwen3_5_mtp',
        num_speculative_tokens=num_speculative_tokens)

    with pytest.raises(ValueError, match='L=.*Tmax='):
        build_executor('model-must-not-be-read', cache_config, None,
                       SimpleNamespace(dp=1, world_size=1), None,
                       specdecode_config=spec_config)


@pytest.mark.parametrize('buffer_len', [0, 7, 9, 64])
def test_public_replayssm_config_rejects_untuned_lengths(buffer_len):
    with pytest.raises(AssertionError, match='replayssm_buffer_len'):
        PytorchEngineConfig(replayssm_buffer_len=buffer_len)


def test_standard_replayssm_named_schema():
    cfg = _cfg()
    configure_gdn_replayssm(cfg,
                            _hf_config(),
                            tp=2,
                            activation_dtype=torch.bfloat16,
                            ssm_dtype=torch.bfloat16,
                            num_spec_tokens=0,
                            enable_replayssm=True,
                            replayssm_buffer_len=16)

    specs = {spec.name: spec for spec in cfg.state_cache_specs}
    assert len(cfg.states_shapes) == 6
    assert cfg.enable_replayssm
    assert list(specs) == [
        GDN_CONV_STATE_CACHE_NAME,
        GDN_REPLAY_CHECKPOINT_CACHE_NAME,
        GDN_REPLAY_D_CACHE_NAME,
        GDN_REPLAY_K_CACHE_NAME,
        GDN_REPLAY_G_CACHE_NAME,
        GDN_REPLAY_CURSOR_CACHE_NAME,
    ]
    assert specs[GDN_CONV_STATE_CACHE_NAME].shape == (768, 4)
    assert specs[GDN_REPLAY_CHECKPOINT_CACHE_NAME].shape == (4, 128, 64)
    assert specs[GDN_REPLAY_D_CACHE_NAME].shape == (4, 16, 128)
    assert specs[GDN_REPLAY_K_CACHE_NAME].shape == (2, 16, 64)
    assert specs[GDN_REPLAY_G_CACHE_NAME].dtype == torch.float32
    assert specs[GDN_REPLAY_CURSOR_CACHE_NAME].shape == (3, )
    assert specs[GDN_REPLAY_CURSOR_CACHE_NAME].layer_ids is None
    for name, spec in specs.items():
        if name != GDN_REPLAY_CURSOR_CACHE_NAME:
            assert spec.layer_ids == [0, 2]


def test_mtp_replayssm_named_schema_uses_fp32_checkpoint_and_raw_rings():
    cfg = _cfg()
    configure_gdn_replayssm(cfg,
                            _hf_config(),
                            tp=1,
                            activation_dtype=torch.bfloat16,
                            ssm_dtype=torch.bfloat16,
                            num_spec_tokens=3,
                            enable_replayssm=True,
                            replayssm_buffer_len=16)

    specs = {spec.name: spec for spec in cfg.state_cache_specs}
    assert cfg.enable_replayssm
    assert specs[GDN_REPLAY_CHECKPOINT_CACHE_NAME].dtype == torch.float32
    assert specs[GDN_REPLAY_RAW_V_CACHE_NAME].shape == (8, 16, 128)
    assert specs[GDN_REPLAY_RAW_K_CACHE_NAME].shape == (4, 16, 64)
    assert specs[GDN_REPLAY_BETA_CACHE_NAME].dtype == torch.float32
    assert specs[GDN_REPLAY_D_CACHE_NAME].dtype == torch.bfloat16
    assert specs[GDN_REPLAY_K_CACHE_NAME].dtype == torch.bfloat16


def test_mtp_replayssm_model_config_rejects_short_buffer():
    with pytest.raises(ValueError, match='L=8, Tmax=5'):
        configure_gdn_replayssm(_cfg(),
                                _hf_config(),
                                tp=1,
                                activation_dtype=torch.bfloat16,
                                ssm_dtype=torch.bfloat16,
                                num_spec_tokens=4,
                                enable_replayssm=True,
                                replayssm_buffer_len=8)


def test_disabled_replayssm_preserves_legacy_shapes():
    cfg = _cfg()
    configure_gdn_replayssm(cfg,
                            _hf_config(),
                            tp=1,
                            activation_dtype=torch.bfloat16,
                            ssm_dtype=torch.float32,
                            num_spec_tokens=0,
                            enable_replayssm=False,
                            replayssm_buffer_len=16)
    assert cfg.states_shapes == [('legacy', torch.bfloat16)]
    assert cfg.state_cache_specs == []
    assert not cfg.enable_replayssm


@pytest.mark.parametrize('num_spec_tokens', [0, 3, 5])
def test_glm_replayssm_preserves_kpool_and_replaces_only_kda_banks(monkeypatch, num_spec_tokens):
    from lmdeploy.hf_configs.configuration_glm5_next import Glm5NextConfig
    from lmdeploy.pytorch.configurations.glm5_next import Glm5NextModelConfigBuilder
    from lmdeploy.pytorch.consts import GLM5_KPOOL_TAIL_K_STATE, GLM5_KPOOL_TAIL_SCORE_STATE

    monkeypatch.setattr('lmdeploy.pytorch.configurations.deepseek_v2.flash_mla_available', lambda: True)
    hf = Glm5NextConfig(text_config=dict(num_hidden_layers=3,
        layer_types=['linear_attention', 'full_attention', 'linear_attention'],
        linear_num_heads=4, linear_head_dim=128, index_head_dim=128))
    kwargs = dict(num_spec_tokens=num_spec_tokens, tp=2,
                  spec_method='deepseek_mtp' if num_spec_tokens else None)
    baseline = Glm5NextModelConfigBuilder.build(hf, **kwargs)
    config = Glm5NextModelConfigBuilder.build(hf, enable_replayssm=True, **kwargs)
    specs = {spec.name: spec for spec in config.state_cache_specs}
    for tail in (GLM5_KPOOL_TAIL_K_STATE, GLM5_KPOOL_TAIL_SCORE_STATE):
        assert specs[tail] == next(s for s in baseline.state_cache_specs if s.name == tail)
        assert not specs[tail].store_cacheable
    assert specs[GDN_REPLAY_CHECKPOINT_CACHE_NAME].shape == (2, 128, 128)
    assert tuple(specs[GDN_REPLAY_CHECKPOINT_CACHE_NAME].layer_ids) == (0, 2)
    assert specs[GDN_REPLAY_G_CACHE_NAME].shape == (2, 16, 128)
    assert specs[GDN_REPLAY_D_CACHE_NAME].dtype == torch.float32
    assert specs[GDN_REPLAY_K_CACHE_NAME].dtype == torch.float32
    assert config.enable_replayssm
    assert (GDN_REPLAY_RAW_K_CACHE_NAME in specs) == bool(num_spec_tokens)
    assert not baseline.enable_replayssm
    draft = Glm5NextModelConfigBuilder.build(hf, enable_replayssm=True, is_draft_model=True, **kwargs)
    assert not draft.enable_replayssm
    assert not draft.state_cache_specs
    with pytest.raises(ValueError, match='deepseek_mtp only'):
        Glm5NextModelConfigBuilder.build(hf, enable_replayssm=True, spec_method='qwen3_5_mtp')


@pytest.mark.parametrize('length', [0, 3, 7, 64])
@pytest.mark.parametrize('num_spec_tokens', [0, 3])
def test_direct_gdn_builder_rejects_invalid_length(length, num_spec_tokens):
    cfg = _cfg()
    with pytest.raises(ValueError, match='one of 8, 16, or 32'):
        configure_gdn_replayssm(cfg, _hf_config(), tp=1, activation_dtype=torch.bfloat16,
                                ssm_dtype=torch.float32, num_spec_tokens=num_spec_tokens,
                                enable_replayssm=True, replayssm_buffer_len=length)
    assert not cfg.enable_replayssm
    assert not cfg.state_cache_specs


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('num_spec_tokens', [0, 3])
@pytest.mark.parametrize('draft', [False, True])
def test_qwen_replayssm_target_draft_protocol(monkeypatch, enabled, num_spec_tokens, draft):
    from transformers import Qwen3_5Config

    from lmdeploy.pytorch.configurations.qwen3_5 import Qwen3_5ModelConfigBuilder

    monkeypatch.setattr('lmdeploy.pytorch.configurations.qwen3_5.is_bf16_supported', lambda _: True)
    hf = Qwen3_5Config(text_config=dict(num_hidden_layers=3,
        layer_types=['linear_attention', 'full_attention', 'linear_attention'],
        linear_key_head_dim=64, linear_value_head_dim=128,
        linear_num_key_heads=4, linear_num_value_heads=8, mtp_num_hidden_layers=1))
    hf.architectures = ['Qwen3_5ForConditionalGeneration']
    cfg = Qwen3_5ModelConfigBuilder.build(hf, enable_replayssm=enabled,
        num_spec_tokens=num_spec_tokens, spec_method='qwen3_5_mtp' if num_spec_tokens else None,
        is_draft_model=draft)
    assert cfg.enable_replayssm == (enabled and not draft)
    assert bool(cfg.state_cache_specs) == (enabled and not draft)
    assert not hasattr(cfg, 'replayssm_mode')
    if draft:
        assert cfg.states_shapes == []
    if cfg.enable_replayssm:
        specs = {s.name: s for s in cfg.state_cache_specs}
        assert (GDN_REPLAY_RAW_K_CACHE_NAME in specs) == (num_spec_tokens > 0)
        assert specs[GDN_CONV_STATE_CACHE_NAME].shape[-1] == hf.text_config.linear_conv_kernel_dim + num_spec_tokens


@pytest.mark.parametrize('num_spec_tokens,paradigm,raw,expected_error', [
    (0, 'ar', False, None), (3, 'ar_spec', True, None),
    (0, 'ar_spec', False, 'target schema'), (3, 'ar', True, 'target schema'),
    (0, 'ar', True, 'raw-history schema'), (3, 'ar_spec', False, 'raw-history schema'),
])
def test_replayssm_build_boundary_matches_configured_protocol(monkeypatch, num_spec_tokens, paradigm, raw,
                                                               expected_error):
    from lmdeploy.pytorch.config import ModelConfig, StateCacheSpec

    names = [GDN_REPLAY_CURSOR_CACHE_NAME]
    if raw:
        names += [GDN_REPLAY_RAW_K_CACHE_NAME, GDN_REPLAY_RAW_V_CACHE_NAME, GDN_REPLAY_BETA_CACHE_NAME]
    cfg = ModelConfig(hidden_size=128, num_layers=1, num_attention_heads=1, num_key_value_heads=1,
                      bos_token_id=0, eos_token_id=[1], head_dim=128, enable_replayssm=True,
                      model_paradigm=paradigm,
                      state_cache_specs=[StateCacheSpec(name, (1,), torch.float32) for name in names])
    monkeypatch.setattr('lmdeploy.pytorch.configurations.AutoModelConfigBuilder.build', lambda *a, **kw: cfg)
    monkeypatch.setattr('lmdeploy.pytorch.config._update_torch_dtype', lambda config, *a, **kw: config)
    if expected_error:
        with pytest.raises(ValueError, match=expected_error):
            ModelConfig.from_hf_config(object(), num_spec_tokens=num_spec_tokens, enable_replayssm=True)
    else:
        assert ModelConfig.from_hf_config(object(), num_spec_tokens=num_spec_tokens, enable_replayssm=True) is cfg


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('draft', [False, True])
def test_qwen_next_replayssm_is_target_only(enabled, draft):
    from transformers import Qwen3NextConfig

    from lmdeploy.pytorch.configurations.qwen3_next import Qwen3NextModelConfigBuilder

    hf = Qwen3NextConfig(num_hidden_layers=4, full_attention_interval=4,
                         layer_types=['linear_attention'] * 3 + ['full_attention'],
                         linear_key_head_dim=64, linear_value_head_dim=128,
                         linear_num_key_heads=4, linear_num_value_heads=8)
    baseline = Qwen3NextModelConfigBuilder.build(hf, enable_replayssm=False, is_draft_model=draft)
    cfg = Qwen3NextModelConfigBuilder.build(hf, enable_replayssm=enabled, is_draft_model=draft)
    assert cfg.enable_replayssm == (enabled and not draft)
    assert bool(cfg.state_cache_specs) == (enabled and not draft)
    if not enabled or draft:
        # Unlike the Qwen3.5 predictor, this builder retains its legacy SSM shapes.
        assert cfg.states_shapes == baseline.states_shapes
        assert cfg.state_cache_specs == baseline.state_cache_specs
    else:
        assert GDN_REPLAY_CHECKPOINT_CACHE_NAME in {spec.name for spec in cfg.state_cache_specs}


@pytest.mark.parametrize('role', ['kv_producer', 'kv_consumer', 'kv_both'])
@pytest.mark.parametrize('method', [None, 'qwen3_5_mtp', 'deepseek_mtp'])
@pytest.mark.parametrize('prefix', [False, True])
def test_replayssm_accepts_mooncake_full_state_protocol(role, method, prefix):
    cache_config = ConfigBuilder.build_cache_config(PytorchEngineConfig(
        enable_replayssm=True, enable_prefix_caching=prefix,
        kv_transfer_config=KVTransferConfig(kv_connector='MooncakeStoreConnector', kv_role=role)))
    spec_config = None if method is None else SpecDecodeConfig(
        model='', method=method, num_speculative_tokens=3)
    _validate_replayssm_preallocation(cache_config, spec_config)
