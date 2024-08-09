# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class BaichuanReader(LlamaReader):
    """BaichuanReader."""

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        q, k, v, o = (None, ) * 4
        pack_key = f'model.layers.{i}.self_attn.W_pack.{kind}'
        qkv = self.transform(self.params.get(pack_key), kind)
        if qkv is not None:
            q, k, v = torch.split(qkv, qkv.shape[0] // 3, dim=0)
        o = self.params.get(f'model.layers.{i}.self_attn.o_proj.{kind}')
        o = self.transform(o, kind)
        return q, k, v, o


@INPUT_MODELS.register_module(name='baichuan')
class BaichuanModel(LlamaModel):
    """Llama model in baichuan format."""

    Reader = BaichuanReader


class Baichuan2Reader(BaichuanReader):
    """Baichuan2Reader."""

    def output_weight(self):
        """Get output."""
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        tensor = self.params.get('lm_head.weight', None)
        if tensor is not None:
            tensor = tensor.cuda()
            tensor = torch.nn.functional.normalize(tensor)
        return tensor


@INPUT_MODELS.register_module(name='baichuan2')
class Baichuan2Model(LlamaModel):
    """Llama model in baichuan format."""

    Reader = Baichuan2Reader
