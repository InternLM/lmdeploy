# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.cuda


def to_cuda(x: torch.Tensor, *args):
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


def process_awq_gemm(x: torch.Tensor, kind: str):
    x = x.cuda()
    if x.dtype == torch.int32:
        x = unpack_awq_gemm(x)
    if kind in ['qweight', 'qzeros', 'scales']:
        x = x.t()
    return x


def process_gptq(x: torch.Tensor, kind: str):
    x = x.cuda()
    if x.dtype == torch.int32:
        xs = get_u4_slices(x, torch.uint8)
        if kind == 'qweight':  # (k/8,n)
            x = torch.stack(xs, dim=1).view(-1, x.size(-1))
        else:  # 'qzeros' (k/g,n/8)
            x = torch.stack(xs, dim=-1).view(x.size(0), -1) + 1
    if kind in ['qweight', 'qzeros', 'scales']:
        x = x.t()
    return x


def get_input_policy(model_format):
    if model_format == 'awq':
        return ('qweight', process_awq_gemm)
    elif model_format == 'gptq':
        return ('qweight', process_gptq)
    else:
        return ('weight', to_cuda)
