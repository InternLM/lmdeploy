# Copyright (c) OpenMMLab. All rights reserved.
import math
import re

import torch

from lmdeploy.archs import get_model_arch

from ..config import RopeParam
from ..loader import create_loader
from .base import INPUT_MODELS, BaseInputModel, BaseReader


class LlamaReader(BaseReader):
    """LlamaReader."""

    attn_layer_prefix = 'model.layers'
    attn_layer_patten = r'model\.layers\.([0-9]+).'
    tok_embeddings_key = 'model.embed_tokens.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'lm_head.weight'

    attn_pattern = r'self_attn'
    ffn_pattern = r'mlp'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool, model_cfg: dict, policy):
        super().__init__()
        self.params = unused_params
        self.params.update(new_params)
        self.last_bin = last_bin
        self.model_cfg = model_cfg
        tie_word_embeddings = self.model_cfg.get('tie_word_embeddings', False)
        if tie_word_embeddings:
            self.output_weight_key = self.tok_embeddings_key
        self.processor = policy

    def filter(self, pattern: str):
        params = []
        for k in self.params.keys():
            if re.search(pattern, k):
                params.append(k)
        return params

    def tok_embeddings(self):
        """Get embeddings."""
        return self.transform(self.params.get(self.tok_embeddings_key, None), 'weight')

    def norm_weight(self):
        """Get norm."""
        return self.transform(self.params.get(self.norm_weight_key, None), 'weight')

    def output_weight(self):
        """Get output."""
        return self.transform(self.params.get(self.output_weight_key, None), 'weight')

    def _transform(self, x: torch.Tensor, kind: str):
        return self.processor(x, kind)

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(f'{self.attn_layer_prefix}.{i}.self_attn.{key}_proj.{kind}')
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def attn(self, i: int, kind: str):
        if not kind:
            return self.filter(self.attn_pattern)
        return self._attn(i, kind)

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.transform(self.params[f'{self.attn_layer_prefix}.{i}.input_layernorm.weight'], 'weight')

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        if not kind:
            return self.filter(self.ffn_pattern)
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[f'{self.attn_layer_prefix}.{i}.mlp.{key}_proj.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int, kind: str):
        if not kind:
            return self.filter(self.ffn_pattern)
        return self._ffn(i, kind)

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.transform(self.params[f'{self.attn_layer_prefix}.{i}.post_attention_layernorm.weight'], 'weight')


@INPUT_MODELS.register_module(name='llama')
class LlamaModel(BaseInputModel):
    """Llama model in hf format."""

    Reader = LlamaReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path)
        self.policy = kwargs.get('input_policy')
        _, self.model_config = get_model_arch(model_path)
        self.model_config = self.model_config.to_dict()

    def readers(self):
        mappings = getattr(self.Reader, 'mappings', [])
        loader = create_loader(self.model_path, self.Reader.attn_layer_patten, mappings)
        for i, param in loader.items():
            reader = self.Reader(param, {}, False, self.model_config, policy=self.policy)
            yield i, reader
        torch.cuda.empty_cache()

    def model_info(self):
        """Read model info."""
        model_arg = self.model_config
        num_layer = model_arg['num_hidden_layers']
        norm_eps = model_arg['rms_norm_eps']
        attn_head_num = model_arg['num_attention_heads']
        vocab_size = model_arg['vocab_size']
        inter_size = model_arg['intermediate_size']
        if 'num_key_value_heads' in model_arg:
            kv_head_num = model_arg['num_key_value_heads']
        else:
            kv_head_num = model_arg['num_attention_heads']
        hidden_units = model_arg['hidden_size']
        # head_dim could be none in config
        head_dim = model_arg.get('head_dim', None)
        head_dim = head_dim or hidden_units // attn_head_num
        # compute rope param
        rope_theta = float(model_arg.get('rope_theta', 10000.0))
        max_position_embeddings = int(model_arg.get('max_position_embeddings', 0))
        rope_param = RopeParam(type='default', base=rope_theta, dim=head_dim)
        rope_scaling = model_arg.get('rope_scaling', None)
        if isinstance(rope_scaling, dict):
            llama2_scaling_type = rope_scaling.get('type', '')
            llama3_scaling_type = rope_scaling.get('rope_type', '')
            if llama2_scaling_type and llama3_scaling_type \
                    and llama2_scaling_type != llama3_scaling_type:
                raise ValueError(f'Ambiguous rope_scaling in config: {model_arg}')
            scaling_type = llama2_scaling_type if llama2_scaling_type \
                else llama3_scaling_type
            if rope_scaling.get('mrope_section') is not None:
                # TODO: treat mrope as an option to the common rope functions
                scaling_type = 'mrope'
            scaling_factor = rope_scaling.get('factor', 0.0)
            if scaling_type == 'default':
                pass
            elif scaling_type == 'dynamic':
                rope_param.type = 'dynamic'
                rope_param.factor = scaling_factor
                rope_param.max_position_embeddings = max_position_embeddings
            elif scaling_type == 'linear':
                rope_param.type = 'linear'
                rope_param.factor = scaling_factor
            elif scaling_type == 'llama3':
                low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
                high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
                original_max_position_embeddings = model_arg['rope_scaling'].get('original_max_position_embeddings', 0)
                rope_param.type = 'llama3'
                rope_param.factor = scaling_factor
                rope_param.low_freq_factor = low_freq_factor
                rope_param.high_freq_factor = high_freq_factor
                rope_param.original_max_position_embeddings = original_max_position_embeddings
            elif scaling_type == 'yarn':
                attention_factor = rope_scaling.get('attention_factor', None)
                if attention_factor is None:
                    attention_factor = 0.1 * math.log(scaling_factor) + 1.0
                beta_fast = rope_scaling.get('beta_fast', 32.0)
                beta_slow = rope_scaling.get('beta_slow', 1.0)
                rope_param.type = 'yarn'
                if 'original_max_position_embeddings' in rope_scaling:
                    original_max_position_embeddings = rope_scaling['original_max_position_embeddings']
                    scaling_factor = max_position_embeddings / original_max_position_embeddings
                else:
                    original_max_position_embeddings = max_position_embeddings
                rope_param.factor = scaling_factor
                rope_param.max_position_embeddings = original_max_position_embeddings
                rope_param.attention_factor = attention_factor
                rope_param.beta_fast = beta_fast
                rope_param.beta_slow = beta_slow
            elif scaling_type == 'mrope':
                mrope_section = rope_scaling.get('mrope_section')
                rope_param.type = 'mrope'
                rope_param.mrope_section = mrope_section
            else:
                raise RuntimeError(f'Unsupported rope type: {scaling_type}')

        return dict(size_per_head=head_dim,
                    num_layer=num_layer,
                    norm_eps=norm_eps,
                    head_num=attn_head_num,
                    kv_head_num=kv_head_num,
                    hidden_units=hidden_units,
                    inter_size=inter_size,
                    vocab_size=vocab_size,
                    max_position_embeddings=max_position_embeddings,
                    rope_param=rope_param)
