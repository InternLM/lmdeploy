# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

from ..config import RopeParam
from .base import INPUT_MODELS
from .internlm2 import InternLM2Reader
from .llama import LlamaModel, LlamaReader


class InternVLReader(LlamaReader):
    """InternVLReader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config') or model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


# Note the subtle difference in keys
class InternVL2Reader(InternLM2Reader):
    """InternVLReader for InternLM2 model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.tok_embeddings.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, **kwargs):
        model_cfg = model_cfg.get('llm_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, **kwargs)


@INPUT_MODELS.register_module(name='internvl')
class InternVLModel(LlamaModel):
    """InternVL model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        llm_config = getattr(config, 'llm_config', None) or getattr(config, 'text_config', None)
        arch = llm_config.architectures[0]
        _readers = dict(InternLM2ForCausalLM=InternVL2Reader,
                        LlamaForCausalLM=InternVLReader,
                        Qwen2ForCausalLM=InternVLReader)
        self.Reader = _readers[arch]

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            file_content = json.load(f)
            model_arg = file_content.get('llm_config') or file_content.get('text_config')
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            hidden_units = model_arg['hidden_size']
            attn_head_num = model_arg['num_attention_heads']
            vocab_size = model_arg['vocab_size']
            inter_size = model_arg['intermediate_size']
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            scaling_type = 'default'
            if isinstance(rope_scaling, dict):
                scaling_type = model_arg['rope_scaling'].get('type', 'default')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')
            attn_bias = 1 if model_arg['architectures'][0] == 'Qwen2ForCausalLM' else 0
            rotary_embedding = hidden_units // attn_head_num
            rope_param = RopeParam(type=scaling_type,
                                   base=rope_theta,
                                   dim=rotary_embedding,
                                   max_position_embeddings=max_position_embeddings,
                                   factor=scaling_factor)

        return dict(num_layer=num_layer,
                    size_per_head=hidden_units // attn_head_num,
                    attn_bias=attn_bias,
                    norm_eps=norm_eps,
                    hidden_units=hidden_units,
                    inter_size=inter_size,
                    vocab_size=vocab_size,
                    head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    max_position_embeddings=max_position_embeddings,
                    rope_param=rope_param)
