# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import contextlib
import functools
from typing import Callable

import torch
import triton
import triton.language as tl

def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(cu_seqlens: torch.LongTensor,
                          chunk_size: int) -> torch.LongTensor:
    indices = torch.cat([
        torch.arange(n)
        for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
    ])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices],
                       1).to(cu_seqlens)


def prepare_chunk_offsets(cu_seqlens: torch.LongTensor,
                          chunk_size: int) -> torch.LongTensor:
    return torch.cat([
        cu_seqlens.new_tensor([0]),
        triton.cdiv(prepare_lens(cu_seqlens), chunk_size)
    ]).cumsum(-1)


def input_guard(
        fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator to make sure all input tensors are contiguous and set the device based on input tensors.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        contiguous_args = (i if not isinstance(i, torch.Tensor) else
                           i.contiguous() for i in args)
        contiguous_kwargs = {
            k: (v if not isinstance(v, torch.Tensor) else v.contiguous())
            for k, v in kwargs.items()
        }

        tensor = None
        for arg in args:
            if isinstance(arg, torch.Tensor):
                tensor = arg
                break
        if tensor is None:
            for value in kwargs.values():
                if isinstance(value, torch.Tensor):
                    tensor = value
                    break

        if tensor is not None:
            ctx = torch.npu.device(tensor.device.index)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))
