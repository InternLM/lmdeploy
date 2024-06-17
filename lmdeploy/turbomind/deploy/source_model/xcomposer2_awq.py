# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .base import INPUT_MODELS
from .llama_awq import ensure_fp16orint32
from .xcomposer2 import Xcomposer2Model, Xcomposer2Reader


class Xcomposer2AwqReader(Xcomposer2Reader):
    """LlamaAwqReader."""

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
        o = self.params[f'model.layers.{i}.attention.wo.{kind}']
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

    def attn_lora_a(self, i):
        """Get attn lora_a."""
        qkv = self.params[f'model.layers.{i}.attention.wqkv.Plora_A.weight']
        o = self.params[f'model.layers.{i}.attention.wo.Plora_A.weight']
        return qkv, o

    def attn_lora_b(self, i):
        """Get attn lora_b."""
        return super()._attn(i, 'Plora_B.weight', 0, 0)

    def ffn(self, i: int):
        """Get ffn qweight for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qweight'))

    def ffn_zero(self, i: int):
        """Get ffn qzeros for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'qzeros'))

    def ffn_scale(self, i: int):
        """Get ffn scales for layer i."""
        return ensure_fp16orint32(self._ffn(i, 'scales'))


@INPUT_MODELS.register_module(name='xcomposer2-awq')
class Xcomposer2AwqModel(Xcomposer2Model):
    """Llama Awq model in hf format."""

    Reader = Xcomposer2AwqReader

    def __init__(self,
                 model_path: str,
                 tokenizer_path: str,
                 ckpt_path: str = None,
                 **kwargs):
        super().__init__(model_path,
                         tokenizer_path,
                         ckpt_path=ckpt_path,
                         **kwargs)
