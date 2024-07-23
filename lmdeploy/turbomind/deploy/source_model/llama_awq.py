# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from .base import INPUT_MODELS
from .llama import LlamaModel, LlamaReader


def get_u4_slices(x: torch.Tensor, dtype: torch.dtype) -> List[torch.Tensor]:
    assert x.dtype == torch.int32
    xs = []
    for _ in range(8):
        xs.append((x & 15).to(dtype))
        x = x >> 4
    return xs


def unpack_awq_gemm(x: torch.Tensor) -> torch.Tensor:
    xs = get_u4_slices(x, torch.uint8)
    order = [0, 4, 1, 5, 2, 6, 3, 7]
    ys = [xs[i] for i in order]
    return torch.stack(ys, dim=-1).view(*x.shape[:-1], -1)


def process_awq_gemm(x: torch.Tensor):
    x = x.cuda()
    if x.dtype == torch.int32:
        x = unpack_awq_gemm(x)
    if len(x.shape) > 1:
        x = x.t()
    return x


class LlamaAwqReader(LlamaReader):
    """LlamaAwqReader."""

    weight_suffix = 'qweight'

    def __init__(self, new_params: dict, unused_params: dict, last_bin: bool,
                 model_cfg: dict):
        super().__init__(new_params, unused_params, last_bin, model_cfg)

    def _transform(self, x: torch.Tensor, kind: str):
        return process_awq_gemm(x)


@INPUT_MODELS.register_module(name='llama-awq')
class LlamaAwqModel(LlamaModel):
    """Llama Awq model in hf format."""

    Reader = LlamaAwqReader
