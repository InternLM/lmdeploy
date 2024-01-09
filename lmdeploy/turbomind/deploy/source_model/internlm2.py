# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader
from .llama_awq import ensure_fp16orint32


class InternLM2Reader(LlamaReader):
    """InternLM2 model reader."""

    attn_layer_patten = r'model.layers.([0-9]+).'
    tok_embeddings_key = 'model.tok_embeddings.weight'
    norm_weight_key = 'model.norm.weight'
    output_weight_key = 'output.weight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        kv_head_num = self.model_cfg['kv_head_num']
        gs = int(self.model_cfg['attn_head_num'] / kv_head_num)
        qkv = self.params[f'model.layers.{i}.attention.wqkv.{kind}']
        qkv = qkv.view(kv_head_num, gs + 2, 128, -1)
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


class InternLM2AwqReader(InternLM2Reader):
    """read weights from internlm2 awq model."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o qweight for layer i."""
        kv_head_num = self.model_cfg['kv_head_num']
        gs = int(self.model_cfg['attn_head_num'] / kv_head_num)
        qkv = self.params[f'model.layers.{i}.attention.wqkv.{kind}']
        hidden_dim = qkv.shape[0]
        qkv = qkv.view(hidden_dim, kv_head_num, gs + 2, -1)
        q, k, v = torch.split(qkv, [gs, 1, 1], dim=-2)
        q = q.reshape(hidden_dim, -1)
        k = k.reshape(hidden_dim, -1)
        v = v.reshape(hidden_dim, -1)
        o = self.params.get(f'model.layers.{i}.attention.wo.{kind}')
        return ensure_fp16orint32((q, k, v, o))

    def attn(self, i: int):
        """Get q, k, v, o qweight for layer i."""
        return self._attn(i, 'qweight')

    def attn_zero(self, i: int):
        """Get q, k, v, o qzeros for layer i."""
        return self._attn(i, 'qzeros')

    def attn_scale(self, i: int):
        """Get q, k, v, o scales for layer i."""
        return self._attn(i, 'scales')

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qweight'))

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qzeros'))

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'scales'))


@INPUT_MODELS.register_module(name='internlm2-awq')
class InternLM2AwqModel(InternLM2Model):
    """InternLM2 awq model."""

    Reader = InternLM2AwqReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs):
        super().__init__(model_path, tokenizer_path, **kwargs)
