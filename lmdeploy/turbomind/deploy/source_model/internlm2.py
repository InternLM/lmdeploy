# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class InternLM2Reader(LlamaReader):
    """InternLM2 model reader."""

    attn_layer_prefix = 'model.layers'
    attn_layer_patten = r'model.layers.([0-9]+).'
    tok_embeddings_key = 'model.tok_embeddings.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'output.weight'

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        q, k, v = (None, ) * 3
        kv_head_num = self.model_cfg['num_key_value_heads']
        gs = int(self.model_cfg['num_attention_heads'] / kv_head_num)
        qkv = self.params.get(
            f'{self.attn_layer_prefix}.{i}.attention.wqkv.{kind}')
        qkv = self.transform(qkv, kind)
        if qkv is not None:
            qkv = qkv.view(kv_head_num, gs + 2, 128, -1)
            hidden_dim = qkv.shape[-1]
            q, k, v = torch.split(qkv, [gs, 1, 1], dim=1)
            q = q.reshape(-1, hidden_dim)
            k = k.reshape(-1, hidden_dim)
            v = v.reshape(-1, hidden_dim)
        o = self.params.get(
            f'{self.attn_layer_prefix}.{i}.attention.wo.{kind}')
        o = self.transform(o, kind)
        return (q, k, v, o)

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[
            f'{self.attn_layer_prefix}.{i}.attention_norm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[
                f'{self.attn_layer_prefix}.{i}.feed_forward.{key}.{kind}']
            tensor = self.transform(tensor, kind)
            result.append(tensor)
        return (*result, )

    def ffn_norm(self, i: int):
        """Get ffn norm for layer i."""
        return self.params[f'{self.attn_layer_prefix}.{i}.ffn_norm.weight']


@INPUT_MODELS.register_module(name='internlm2')
class InternLM2Model(LlamaModel):
    """InternLM2 model in hf format."""

    Reader = InternLM2Reader
