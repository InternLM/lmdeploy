# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.config import StateCacheSpec
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

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder


def _check_env_qwen3_next(device: str):
    """Check env for qwen3 next."""
    if device != 'cuda':
        return

    try:
        import fla  # noqa: F401
    except ImportError:
        raise ImportError('Qwen3-Next cuda support requires https://github.com/fla-org/flash-linear-attention.')


def configure_gdn_replayssm(cfg,
                            hf_config,
                            *,
                            tp: int,
                            activation_dtype: torch.dtype,
                            ssm_dtype: torch.dtype,
                            num_spec_tokens: int,
                            enable_replayssm: bool,
                            replayssm_buffer_len: int) -> None:
    """Replace legacy GDN state shapes with the named ReplaySSM schema."""
    cfg.enable_replayssm = enable_replayssm
    cfg.replayssm_buffer_len = replayssm_buffer_len
    if not enable_replayssm:
        return
    tmax = num_spec_tokens + 1
    if num_spec_tokens > 0 and replayssm_buffer_len < 2 * tmax:
        raise ValueError(
            'MTP ReplaySSM requires replayssm_buffer_len >= '
            '2 * (num_speculative_tokens + 1), got '
            f'L={replayssm_buffer_len}, Tmax={tmax}.')
    cfg.replayssm_baseline_state_shapes = list(cfg.states_shapes)

    layer_ids = [idx for idx, layer_type in enumerate(hf_config.layer_types)
                 if layer_type == 'linear_attention']
    if not layer_ids:
        raise ValueError('ReplaySSM requires at least one linear_attention layer.')

    head_k_dim = hf_config.linear_key_head_dim
    head_v_dim = hf_config.linear_value_head_dim
    num_v_heads = hf_config.linear_num_value_heads // tp
    num_k_heads = hf_config.linear_num_key_heads // tp
    key_dim = head_k_dim * num_k_heads
    value_dim = head_v_dim * num_v_heads
    conv_dim = key_dim * 2 + value_dim
    conv_kernel_size = hf_config.linear_conv_kernel_dim + num_spec_tokens
    mode = 'mtp' if num_spec_tokens > 0 else 'standard'
    checkpoint_dtype = torch.float32 if mode == 'mtp' else ssm_dtype
    ring_len = replayssm_buffer_len

    specs = [
        StateCacheSpec(GDN_CONV_STATE_CACHE_NAME,
                       (conv_dim, conv_kernel_size), activation_dtype,
                       layer_ids=layer_ids),
        StateCacheSpec(GDN_REPLAY_CHECKPOINT_CACHE_NAME,
                       (num_v_heads, head_v_dim, head_k_dim), checkpoint_dtype,
                       layer_ids=layer_ids),
        StateCacheSpec(GDN_REPLAY_D_CACHE_NAME,
                       (num_v_heads, ring_len, head_v_dim),
                       activation_dtype if mode == 'mtp' else ssm_dtype,
                       layer_ids=layer_ids),
        StateCacheSpec(GDN_REPLAY_K_CACHE_NAME,
                       (num_k_heads, ring_len, head_k_dim),
                       activation_dtype if mode == 'mtp' else ssm_dtype,
                       layer_ids=layer_ids),
        StateCacheSpec(GDN_REPLAY_G_CACHE_NAME,
                       (num_v_heads, ring_len), torch.float32,
                       layer_ids=layer_ids),
    ]
    if mode == 'mtp':
        specs.extend([
            StateCacheSpec(GDN_REPLAY_RAW_V_CACHE_NAME,
                           (num_v_heads, ring_len, head_v_dim), activation_dtype,
                           layer_ids=layer_ids),
            StateCacheSpec(GDN_REPLAY_RAW_K_CACHE_NAME,
                           (num_k_heads, ring_len, head_k_dim), activation_dtype,
                           layer_ids=layer_ids),
            StateCacheSpec(GDN_REPLAY_BETA_CACHE_NAME,
                           (num_v_heads, ring_len), torch.float32,
                           layer_ids=layer_ids),
        ])
    specs.append(StateCacheSpec(GDN_REPLAY_CURSOR_CACHE_NAME, (3, ), torch.int32))
    cfg.state_cache_specs = specs
    # Compatibility bridge for scheduler/executor SSM feature gates. Named
    # specs remain the allocation authority.
    cfg.states_shapes = [(tuple(spec.shape), spec.dtype) for spec in specs]
    cfg.replayssm_mode = mode


class Qwen3NextModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type == 'qwen3_next'

    @classmethod
    def build(cls, hf_config, model_path: str = None, tp: int = 1, **kwargs):
        """build."""
        cfg = DefaultModelConfigBuilder.build(hf_config, model_path, tp=tp, **kwargs)

        # update num layers
        num_layers = cfg.num_layers
        num_full_layers = num_layers // hf_config.full_attention_interval
        num_delta_layers = num_full_layers * (hf_config.full_attention_interval - 1)
        cfg.num_layers = num_full_layers

        # set state shapes
        head_k_dim = hf_config.linear_key_head_dim
        head_v_dim = hf_config.linear_value_head_dim
        num_v_heads = hf_config.linear_num_value_heads // tp
        num_k_heads = hf_config.linear_num_key_heads // tp
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = hf_config.linear_conv_kernel_dim

        conv_state_shape = (num_delta_layers, conv_dim, conv_kernel_size)
        recurrent_state_shape = (num_delta_layers, num_v_heads, head_v_dim, head_k_dim)
        dtype = torch.bfloat16
        cfg.states_shapes = [(conv_state_shape, dtype), (recurrent_state_shape, dtype)]
        cfg.is_gated_delta = True
        cfg.check_env_func = _check_env_qwen3_next
        configure_gdn_replayssm(
            cfg,
            hf_config,
            tp=tp,
            activation_dtype=dtype,
            ssm_dtype=dtype,
            num_spec_tokens=kwargs.get('num_spec_tokens', 0),
            enable_replayssm=kwargs.get('enable_replayssm', False),
            replayssm_buffer_len=kwargs.get('replayssm_buffer_len', 16),
        )
        return cfg
