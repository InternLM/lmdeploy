# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(A, B, C, size, BLOCK: tl.constexpr):
    """add kernel."""
    prog_id = tl.program_id(0)
    offs = prog_id * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + offs, mask=offs < size)
    b = tl.load(B + offs, mask=offs < size)
    tl.store(C + offs, a + b, mask=offs < size)


def custom_add(a, b):
    """custom add one."""
    c = torch.empty_like(a)
    size = c.size(0)
    BLOCK = 16

    grid = [triton.cdiv(size, BLOCK)]
    _add_kernel[grid](a, b, c, size, BLOCK=BLOCK)
    return c
