# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .builder import AutoModelConfigBuilder
from .default import DefaultModelConfigBuilder
from .qwen3_next import _check_env_qwen3_next


class Qwen3_5ModelConfigBuilder(AutoModelConfigBuilder):

    @classmethod
    def condition(cls, hf_config):
        """config."""
        return hf_config.model_type in ['qwen3_5', 'qwen3_5_moe']

    @classmethod
    def build(cls,
              hf_config,
              model_path: str = None,
              tp: int = 1,
              is_draft_model: bool = False,
              spec_method: str = None,
              **kwargs):
        """build."""
        text_config = hf_config.text_config
        # propagate quantization_config from top-level hf_config into text_config
        quantization_config = getattr(hf_config, 'quantization_config', None)
        if quantization_config is not None and not hasattr(text_config, 'quantization_config'):
            text_config.quantization_config = quantization_config
        cfg = DefaultModelConfigBuilder.build(text_config, model_path, tp=tp, **kwargs)

        # update num layers
        num_layers = cfg.num_layers
        layer_types = text_config.layer_types
        num_delta_layers = sum([1 for lt in layer_types if lt == 'linear_attention'])
        num_full_layers = num_layers - num_delta_layers
        cfg.num_layers = num_full_layers

        # set state shapes
        head_k_dim = text_config.linear_key_head_dim
        head_v_dim = text_config.linear_value_head_dim
        num_v_heads = text_config.linear_num_value_heads // tp
        num_k_heads = text_config.linear_num_key_heads // tp
        key_dim = head_k_dim * num_k_heads
        value_dim = head_v_dim * num_v_heads
        conv_dim = key_dim * 2 + value_dim
        conv_kernel_size = text_config.linear_conv_kernel_dim

        conv_state_shape = (num_delta_layers, conv_dim, conv_kernel_size)
        recurrent_state_shape = (num_delta_layers, num_v_heads, head_k_dim, head_v_dim)
        dtype = torch.bfloat16
        cfg.states_shapes = [(conv_state_shape, dtype), (recurrent_state_shape, dtype)]
        cfg.check_env_func = _check_env_qwen3_next

        # for spec
        if spec_method is not None:
            assert spec_method == 'deepseek_mtp'
            cfg.model_paradigm = 'ar_spec'

        # draft model cfg
        if is_draft_model:
            hf_config.architectures[0] = 'Qwen3_5MTPModel'
            # remove for correct mapping when building the patched model
            if hasattr(hf_config, 'auto_map'):
                del hf_config.auto_map

            cfg.model_paradigm = 'ar_spec'
            cfg.num_layers = text_config.mtp_num_hidden_layers
            cfg.states_shapes = []

        return cfg
