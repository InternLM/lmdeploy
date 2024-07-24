# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.cuda

def to_cuda(x: torch.Tensor):
    return x.cuda()


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
    return x.t()


def get_input_policy(model_format):
    if model_format == 'awq':
        return ('qweight', process_awq_gemm)
    else:
        return ('weight', to_cuda)
