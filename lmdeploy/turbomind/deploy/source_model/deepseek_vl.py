# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class DeepSeekVLReader(LlamaReader):
    """DeepSeekVL model reader."""

    attn_layer_patten = r'language_model.model.layers.([0-9]+).'
    tok_embeddings_key = 'language_model.model.embed_tokens.weight'
    norm_weight_key = 'language_model.model.norm.weight'
    output_weight_key = 'language_model.lm_head.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def init_layer_id(self):
        """Get start/end transformer layer id."""
        super().init_layer_id()

    def clean_up(self, last: bool) -> None:
        """Clean up unused params."""
        super().clean_up(last)

    @property
    def start_layer_id(self):
        """Get start transformer layer id."""
        return self._start_layer_id

    @property
    def end_layer_id(self):
        """Get end transformer layer id."""
        return self._end_layer_id

    def tok_embeddings(self):
        """Get embeddings."""
        return self.params.get(self.tok_embeddings_key, None)

    def norm_weight(self):
        """Get norm."""
        return self.params.get(self.norm_weight_key, None)

    def output_weight(self):
        """Get output."""
        return self.params.get(self.output_weight_key, None)

    def _attn(self, i: int, kind: str, allow_none=False):
        """Get q, k, v, o kind for layer i."""
        result = []
        for key in ['q', 'k', 'v', 'o']:
            tensor = self.params.get(
                f'language_model.model.layers.{i}.self_attn.{key}_proj.{kind}')
            if not allow_none:
                assert tensor is not None
            result.append(tensor)
        return (*result, )

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        return self._attn(i, 'weight')

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        return self._attn(i, 'bias', allow_none=True)

    def attn_zero(self, i: int):
        """Get q, k, v, o zero point for layer i."""
        return (None, ) * 4

    def attn_scale(self, i: int):
        """Get q, k, v, o scale for layer i."""
        return (None, ) * 4

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[
            f'language_model.model.layers.{i}.input_layernorm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['gate', 'down', 'up']:
            tensor = self.params[
                f'language_model.model.layers.{i}.mlp.{key}_proj.{kind}']
            result.append(tensor)
        return (*result, )

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        return self._ffn(i, 'weight')

    def ffn_zero(self, i: int):
        """Get ffn zero point for layer i."""
        return (None, ) * 3

    def ffn_scale(self, i: int):
        """Get ffn scale for layer i."""
        return (None, ) * 3

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[
            f'language_model.model.layers.{i}.post_attention_layernorm.weight']


@INPUT_MODELS.register_module(name='deepseekvl')
class DeepSeekVLModel(LlamaModel):
    """DeepSeekVL model in hf format."""

    Reader = DeepSeekVLReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)

    def model_info(self):
        """Read model info."""
        params_path = osp.join(self.model_path, 'config.json')
        with open(params_path) as f:
            model_arg = json.load(f)
            if 'language_config' in model_arg and model_arg[
                    'language_config'].get('model_type', None) == 'llama':
                model_arg = model_arg['language_config']  # depseek-vl
            num_layer = model_arg['num_hidden_layers']
            norm_eps = model_arg.get('rms_norm_eps', 1e-06)
            attn_head_num = model_arg.get('num_attention_heads', 32)
            if 'num_key_value_heads' in model_arg:
                kv_head_num = model_arg['num_key_value_heads']
            else:
                kv_head_num = model_arg.get('num_attention_heads', 32)
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
