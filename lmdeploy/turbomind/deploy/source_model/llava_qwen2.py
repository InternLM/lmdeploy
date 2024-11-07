# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class LlavaQwen2Reader(LlamaReader):
    """LlavaQwen2Reader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict, policy):
        model_cfg = model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg,
                         policy)


@INPUT_MODELS.register_module(name='llava_qwen2')
class LlavaQwen2Model(LlamaModel):
    """LlavaQwen2Model model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        self.Reader = LlavaQwen2Reader

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)['text_config']
            num_layer = model_arg.get('num_hidden_layers', 32)
            norm_eps = model_arg.get('rms_norm_eps', 1e-6)
            attn_head_num = model_arg.get('num_attention_heads', 32)
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg.get('num_key_value_heads', 32)
            else:
                kv_head_num = model_arg.get('num_attention_heads', 32)
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(
                model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            use_dynamic_ntk = 0

            # special for the model: llava-hf/llava-interleave-qwen-7b-hf
            hidden_units = model_arg.get('hidden_size', 4096)
            vocab_size = model_arg.get('vocab_size', 152000)
            intermediate_size = model_arg.get('intermediate_size', 11008)
            attn_bias = int(model_arg.get('attn_bias', 1))
            use_logn_attn = int(model_arg.get('use_logn_attn', 0))

            if isinstance(rope_scaling, dict):
                scaling_type = model_arg['rope_scaling'].get('type', '')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')
                if scaling_type == 'dynamic':
                    use_dynamic_ntk = 1

        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    head_num=attn_head_num,
                    hidden_units=hidden_units,
                    kv_head_num=kv_head_num,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    use_dynamic_ntk=use_dynamic_ntk,
                    rope_scaling_factor=scaling_factor,
                    inter_size=intermediate_size,
                    use_logn_attn=use_logn_attn,
                    attn_bias=attn_bias,
                    vocab_size=vocab_size)