# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch

from .base import INPUT_MODELS
from .glm4 import Glm4Model, Glm4Reader


class Glm4AwqReader(Glm4Reader):
    """Glm4AwqReader."""

    attn_layer_patten = r'transformer.encoder.layers.([0-9]+).'
    tok_embeddings_key = 'transformer.embedding.word_embeddings.weight'
    norm_weight_key = 'transformer.encoder.final_layernorm.weight'
    output_weight_key = 'transformer.output_layer.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        qkv = self.params[f'transformer.encoder.layers.{i}'
                          f'.self_attention.query_key_value.{kind}']
        attn_head_num = self.model_cfg['num_attention_heads']
        kv_head_num = attn_head_num
        if self.model_cfg.get('multi_query_attention', False):
            kv_head_num = self.model_cfg['multi_query_group_num']
        HEAD_DIM = int(qkv.shape[size_dim] / (attn_head_num + kv_head_num * 2))
        q, k, v = torch.split(qkv, [
            attn_head_num * HEAD_DIM, kv_head_num * HEAD_DIM,
            kv_head_num * HEAD_DIM
        ],
                              dim=size_dim)
        o = self.params.get(
            f'transformer.encoder.layers.{i}.self_attention.dense.{kind}',
            None)
        if o is None:  # handle the case when qkv has bias but o doesn't
            o = torch.zeros_like(q)
        return q, k, v, o

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        return self._attn(i, 'qweight', -1, -1)

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        return self._attn(i, 'qzeros', -1, -1)

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        return self._attn(i, 'scales', -1, -1)

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        return self._attn(i, 'bias', -1, 0)

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        up_and_gate = self.params[
            f'transformer.encoder.layers.{i}.mlp.dense_h_to_4h.{kind}']
        up, gate = up_and_gate.chunk(2, dim=-1)
        down = self.params[
            f'transformer.encoder.layers.{i}.mlp.dense_4h_to_h.{kind}']

        return (up, down, gate)

    def ffn(self, i: int):
        """Get ffn weight for layer i."""
        return self._ffn(i, 'qweight')

    def ffn_zero(self, i: int):
        """Get ffn zero point for layer i."""
        return self._ffn(i, 'qzeros')

    def ffn_scale(self, i: int):
        """Get ffn scale for layer i."""
        return self._ffn(i, 'scales')


@INPUT_MODELS.register_module(name='glm4-awq')
class Glm4AwqModel(Glm4Model):
    """Glm2/3/4 model in hf format."""

    Reader = Glm4AwqReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
        config_path = osp.join(self.model_path, 'config.json')
        with open(config_path) as f:
            self.config = json.load(f)
