# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

from ..config import RopeParam
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class DeepSeekVLReader(LlamaReader):
    """DeepSeekVL model reader."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg['language_config']
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'language_model.model.layers.{i}.input_layernorm.weight']

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'language_model.model.layers.{i}.post_attention_layernorm.weight']


@INPUT_MODELS.register_module(name='deepseekvl')
class DeepSeekVLModel(LlamaModel):
    """DeepSeekVL model in hf format."""

    Reader = DeepSeekVLReader

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            if 'language_config' in model_arg and model_arg['language_config'].get('model_type', None) == 'llama':
                model_arg = model_arg['language_config']  # depseek-vl
            num_layer = model_arg['num_hidden_layers']
            hidden_units = model_arg.get('hidden_size', 4096)
            inter_size = model_arg.get('intermediate_size', 11008)
            vocab_size = model_arg.get('vocab_size', 102400)
            norm_eps = model_arg.get('rms_norm_eps', 1e-06)
            attn_head_num = model_arg.get('num_attention_heads', 32)
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg.get('num_attention_heads', 32)
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            scaling_type = 'default'
            if isinstance(rope_scaling, dict):
                scaling_type = model_arg['rope_scaling'].get('type', 'default')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')
            head_dim = model_arg.get('head_dim', hidden_units // attn_head_num)
            rope_param = RopeParam(type=scaling_type,
                                   base=rope_theta,
                                   dim=head_dim,
                                   max_position_embeddings=max_position_embeddings,
                                   factor=scaling_factor)

        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    hidden_units=hidden_units,
                    inter_size=inter_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=max_position_embeddings,
                    rope_param=rope_param)
