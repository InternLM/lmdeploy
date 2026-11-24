# Copyright © 2025 Huawei Technologies Co., Ltd.
# Based on vLLM: https://github.com/vllm-project/vllm
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch
import triton
import triton.language as tl


def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


@triton.jit
def safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float('-inf')))
