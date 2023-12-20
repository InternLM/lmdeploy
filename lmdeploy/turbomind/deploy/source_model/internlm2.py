# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class InternLM2Reader(LlamaReader):
    """InternLM2 model reader."""

    attn_layer_patten = r'model.layers.([0-9]+).'
    tok_embeddings_key = 'model.tok_embeddings.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        qkv = self.params[f'model.layers.{i}.attention.wqkv.{kind}']
        gs = 4
        qkv = qkv.view(8, gs + 2, 128, -1)
        hidden_dim = qkv.shape[-1]
        q, k, v = torch.split(qkv, [gs, 1, 1], dim=1)
        q = q.reshape(-1, hidden_dim)
        k = k.reshape(-1, hidden_dim)
        v = v.reshape(-1, hidden_dim)
        o = self.params.get(f'model.layers.{i}.attention.wo.{kind}')
        return q, k, v, o

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        return self._attn(i, 'weight', 0, 0)

    def attn_bias(self, i: int):
        return (None, ) * 4

    def attn_zero(self, i: int):
        """Get q, k, v, o zero point for layer i."""
        return (None, ) * 4

    def attn_scale(self, i: int):
        """Get q, k, v, o scale for layer i."""
        return (None, ) * 4

    def attn_norm(self, i: int):
        """Get attn norm for layer i."""
        return self.params[f'model.layers.{i}.attention_norm.weight']

    def _ffn(self, i: int, kind: str):
        """Get ffn kind for layer i."""
        result = []
        for key in ['w1', 'w2', 'w3']:
            tensor = self.params[f'model.layers.{i}.feed_forward.{key}.{kind}']
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
        return self.params[f'model.layers.{i}.ffn_norm.weight']


@INPUT_MODELS.register_module(name='internlm2')
class InternLM2Model(LlamaModel):
    """InternLM2 model in hf format."""

    Reader = InternLM2Reader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
