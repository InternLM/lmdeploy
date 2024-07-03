# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

from .base import INPUT_MODELS
from .internlm2 import InternLM2AwqReader, InternLM2Reader
from .llama import LlamaModel, LlamaReader
from .llama_awq import LlamaAwqReader


class InternVLReader(LlamaReader):
    """InternVLReader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'


class InternVL2Reader(InternLM2Reader):
    """InternVLReader for InternLM2 model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.tok_embeddings.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        model_cfg = model_cfg.get('llm_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg)


@INPUT_MODELS.register_module(name='internvl')
class InternVLModel(LlamaModel):
    """InternVL model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = config.llm_config.architectures[0]
        _readers = dict(InternLM2ForCausalLM=InternVL2Reader,
                        LlamaForCausalLM=InternVLReader)
        self.Reader = _readers[arch]

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)['llm_config']
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg['rms_norm_eps']
            attn_head_num = model_arg['num_attention_heads']
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg['num_attention_heads']
            rope_theta = float(model_arg.get('rope_theta', 10000.0))
            max_position_embeddings = int(
                model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            use_dynamic_ntk = 0
            if isinstance(rope_scaling, dict):
                scaling_type = model_arg['rope_scaling'].get('type', '')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')
                if scaling_type == 'dynamic':
                    use_dynamic_ntk = 1

        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    attn_head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    rope_theta=rope_theta,
                    max_position_embeddings=max_position_embeddings,
                    use_dynamic_ntk=use_dynamic_ntk,
                    rope_scaling_factor=scaling_factor)


class InternVLAwqReader(LlamaAwqReader):
    """InternVLReader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'


class InternVL2AwqReader(InternLM2AwqReader):
    """InternVLReader for InternLM2 model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.tok_embeddings.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        model_cfg = model_cfg.get('llm_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg)


@INPUT_MODELS.register_module(name='internvl-awq')
class InternVLAwqModel(InternVLModel):
    """InternVL model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = config.llm_config.architectures[0]
        _readers = dict(InternLM2ForCausalLM=InternVL2AwqReader,
                        LlamaForCausalLM=InternVLAwqReader)
        self.Reader = _readers[arch]
