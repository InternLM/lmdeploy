from types import SimpleNamespace

import pytest
import torch

from lmdeploy.messages import PytorchEngineConfig
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
from lmdeploy.pytorch.kernels.cuda.gated_delta_replayssm_config import (
    MTP_COMMIT_CURSOR_SM90_CONFIG,
    RESET_CURSOR_SM90_CONFIG,
    STANDARD_CURSOR_SM90_CONFIG,
    get_mtp_exact_fold_config,
    get_standard_replayssm_config,
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
                           replayssm_buffer_len=16,
                           replayssm_mode=None)


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


@pytest.mark.parametrize('method', ['deepseek_mtp', 'hy3_mtp', 'medusa'])
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
    assert cfg.replayssm_mode == 'standard'
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
    assert cfg.replayssm_mode == 'mtp'
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
    assert cfg.replayssm_mode is None


@pytest.mark.parametrize(
    ('batch', 'expected'),
    [
        (1, (32, 1, 1)),
        (8, (32, 2, 4)),
        (32, (128, 4, 4)),
        (64, (128, 4, 3)),
        (256, (128, 4, 3)),
        (512, (128, 4, 4)),
    ],
)
def test_standard_replayssm_uses_offline_batch_table(batch, expected):
    config = get_standard_replayssm_config(batch)
    assert (config['BV'], config['NK'], config['num_stages']) == expected
    assert config['num_warps'] == 1
    assert config['dot_precision'] == 'ieee'


def test_cursor_operations_keep_independently_tuned_blocks():
    assert STANDARD_CURSOR_SM90_CONFIG['BLOCK'] == 128
    assert MTP_COMMIT_CURSOR_SM90_CONFIG['BLOCK'] == 256
    assert RESET_CURSOR_SM90_CONFIG['BLOCK'] == 128


def test_mtp_exact_fold_uses_single_row_specialization():
    assert get_mtp_exact_fold_config(1) == {'BV': 32, 'num_warps': 2, 'num_stages': 3}
    assert get_mtp_exact_fold_config(2) == {'BV': 16, 'num_warps': 1, 'num_stages': 1}
