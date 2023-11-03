# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


class BaichuanReader(LlamaReader):
    """BaichuanReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def _attn(self, i: int, kind: str, size_dim: int, dim: int = 0):
        """Get q, k, v, o kind for layer i."""
        result = []
        pack_key = f'model.layers.{i}.self_attn.W_pack.{kind}'
        qkv = self.params[pack_key]
        result.extend(torch.split(qkv, qkv.shape[size_dim] // 3, dim=dim))
        o = self.params[f'model.layers.{i}.self_attn.o_proj.{kind}']
        result.append(o)
        return (*result, )

    def attn(self, i: int):
        """Get q, k, v, o weight for layer i."""
        return self._attn(i, 'weight', 0, 0)

    def attn_bias(self, i: int):
        """Get q, k, v, o bias for layer i."""
        return (None, ) * 4


class Baichuan2Reader(BaichuanReader):
    """Baichuan2Reader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool):
        super().__init__(new_params, unused_params, last_bin)

    def output_weight(self):
        """Get output."""
        # https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/blob/main/modeling_baichuan.py#L507
        tensor = self.params.get('lm_head.weight', None)
        if tensor is not None:
            tensor = tensor.cuda()
            tensor = torch.nn.functional.normalize(tensor)
        return tensor


@INPUT_MODELS.register_module(name='baichuan')
class BaichuanModel(LlamaModel):
    """Llama model in baichuan format."""

    Reader = BaichuanReader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path, **kwargs)


@INPUT_MODELS.register_module(name='baichuan2')
class Baichuan2Model(LlamaModel):
    """Llama model in baichuan format."""

    Reader = Baichuan2Reader

    def __init__(self, model_path: str, tokenizer_path: str, **kwargs: dict):
        super().__init__(model_path, tokenizer_path, **kwargs)
