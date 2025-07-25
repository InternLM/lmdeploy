# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

from ..config import RopeParam
from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class LlavaReader(LlamaReader):
    """LlavaReader for llama model."""

    attn_layer_prefix = 'language_model.model.layers'
    attn_layer_patten = r'language_model\.model\.layers\.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, policy):
        model_cfg = model_cfg.get('text_config')
        super().__init__(new_params, unused_params, last_bin, model_cfg, policy)


@INPUT_MODELS.register_module(name='llava')
class LlavaModel(LlamaModel):
    """LlavaModel model in hf format."""

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config = getattr(config, 'text_config', config)
        arch = config.architectures[0]
        _readers = dict(Qwen2ForCausalLM=LlavaReader, LlamaForCausalLM=LlavaReader)
        self.Reader = _readers[arch]
        self.arch = arch

    def model_info(self):
        """Read model info for LlavaForConditionalGeneration.

        https://huggingface.co/llava-hf/llava-interleave-qwen-7b-hf
        """
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
            max_position_embeddings = int(model_arg.get('max_position_embeddings', 0))
            rope_scaling = model_arg.get('rope_scaling', None)
            scaling_factor = 0.0
            scaling_type = 'default'

            # special for the model: llava-hf/llava-interleave-qwen-7b-hf
            hidden_units = model_arg.get('hidden_size', 4096)
            vocab_size = model_arg.get('vocab_size', 152000)
            intermediate_size = model_arg.get('intermediate_size', 11008)
            attn_bias = 1 if model_arg['architectures'][0] \
                == 'Qwen2ForCausalLM' else 0
            attn_bias = int(model_arg.get('attn_bias', attn_bias))
            use_logn_attn = int(model_arg.get('use_logn_attn', 0))

            if isinstance(rope_scaling, dict):
                scaling_type = model_arg['rope_scaling'].get('type', '')
                scaling_factor = model_arg['rope_scaling'].get('factor', '')

            rope_param = RopeParam(type=scaling_type,
                                   base=rope_theta,
                                   dim=hidden_units // attn_head_num,
                                   max_position_embeddings=max_position_embeddings,
                                   factor=scaling_factor)

        return dict(num_layer=num_layer,
                    norm_eps=norm_eps,
                    head_num=attn_head_num,
                    hidden_units=hidden_units,
                    kv_head_num=kv_head_num,
                    rope_param=rope_param,
                    max_position_embeddings=max_position_embeddings,
                    inter_size=intermediate_size,
                    use_logn_attn=use_logn_attn,
                    attn_bias=attn_bias,
                    vocab_size=vocab_size)
