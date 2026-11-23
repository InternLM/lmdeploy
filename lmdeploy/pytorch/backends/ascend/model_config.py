# Copyright (c) 2024, DeepLink. All rights reserved.
"""Ascend model configuration adapters, invoked explicitly by shared
builders."""

import json
from pathlib import Path

import torch


def load_modelslim_config(hf_config, model_path):
    if model_path is None:
        return
    path = Path(model_path) / 'quant_model_description.json'
    if not path.is_file():
        return
    description = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(description, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    hf_config.quantization_config = dict(quant_method='modelslim', quant_dtype='int8', quant_description=description)


def configure_qwen3_5_states(cfg, text_config, tp, num_spec_tokens, dtype):
    num_layers = sum(lt == 'linear_attention' for lt in text_config.layer_types)
    num_v_heads = text_config.linear_num_value_heads // tp
    num_k_heads = text_config.linear_num_key_heads // tp
    head_k_dim = text_config.linear_key_head_dim
    head_v_dim = text_config.linear_value_head_dim
    conv_dim = 2 * head_k_dim * num_k_heads + head_v_dim * num_v_heads
    conv_shape = (text_config.linear_conv_kernel_dim + num_spec_tokens, conv_dim)
    recurrent_shape = (num_v_heads, head_k_dim, head_v_dim)
    if num_spec_tokens:
        recurrent_shape = (1 + num_spec_tokens, *recurrent_shape)
    cfg.state_cache_is_indexed = False
    cfg.state_cache_specs = []
    cfg.states_shapes = [(conv_shape, dtype)] * num_layers + [(recurrent_shape, torch.float32)] * num_layers
