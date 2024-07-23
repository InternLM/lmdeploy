# Copyright (c) OpenMMLab. All rights reserved.

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader
from .llama_awq import process_awq_gemm


class BaichuanReader(LlamaReader):
    """BaichuanReader."""

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _attn(self, i: int, kind: str):
        """Get q, k, v, o kind for layer i."""
        q, k, v, o = (None, ) * 4
        pack_key = f'model.layers.{i}.self_attn.W_pack.{kind}'
        qkv = self.transform(self.params.get(pack_key), kind)
        if qkv:
            q, k, v = torch.split(qkv, qkv.shape[0] // 3, dim=0)
        o = self.params.get(f'model.layers.{i}.self_attn.o_proj.{kind}')
        o = self.transform(o, kind)
        return q, k, v, o


@INPUT_MODELS.register_module(name='baichuan')
class BaichuanModel(LlamaModel):
    """Llama model in baichuan format."""

    Reader = BaichuanReader


class BaichuanAwqReader(BaichuanReader):
    """BaichuanAwqReader."""

    weight_suffix = 'qweight'

    def _transform(self, x: torch.Tensor, kind: str):
        return process_awq_gemm(x)


@INPUT_MODELS.register_module(name='baichuan-awq')
class BaichuanAwqModel(BaichuanModel):
    """BaichuanAwqModel."""

    Reader = BaichuanAwqReader
