# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor

from .triton_utils import get_kernel_meta, wrap_jit_func


@triton.jit
def _compute_rms_norm(x, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    """compute rms norm."""
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf * tl.math.rsqrt(var + eps)
    out = (w * out).to(x.dtype)
    return out


@wrap_jit_func(type_hint=dict(
    input=Tensor,
    weight=Tensor,
    output=Tensor,
    input_row_stride=int,
    eps=float,
    N_COLS=torch.int32,
    BLOCK_N=torch.int32,
))
@triton.jit
def rms_norm_kernel(input, weight, output, input_row_stride: tl.constexpr,
                    eps: tl.constexpr, N_COLS: tl.constexpr,
                    BLOCK_N: tl.constexpr):
    """rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    out = _compute_rms_norm(x, w, eps, N_COLS)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


@wrap_jit_func(type_hint=dict(
    input=Tensor,
    weight=Tensor,
    residual=Tensor,
    output=Tensor,
    out_residual=Tensor,
    input_row_stride=int,
    residual_row_stride=int,
    eps=float,
    N_COLS=torch.int32,
    BLOCK_N=torch.int32,
))
@triton.jit
def add_rms_norm_kernel(input, weight, residual, output, out_residual,
                        input_row_stride: tl.constexpr,
                        residual_row_stride: tl.constexpr, eps: tl.constexpr,
                        N_COLS: tl.constexpr, BLOCK_N: tl.constexpr):
    """rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)

    res_ptr = residual + prog_id * residual_row_stride
    res = tl.load(res_ptr + offsets, mask=offsets < N_COLS)

    new_x = x + res
    out_res_ptr = out_residual + prog_id * residual_row_stride
    tl.store(out_res_ptr + offsets, new_x, mask=offsets < N_COLS)

    out = _compute_rms_norm(new_x, w, eps, N_COLS)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def rms_norm(hidden_states: Tensor,
             weight: Tensor,
             eps: float = 1e-6,
             residual: Tensor = None,
             out: Tensor = None,
             out_residual: Tensor = None):
    """rms norm."""
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()

    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)

    if out is None:
        out = torch.empty_like(hidden_states)

    kernel_meta = get_kernel_meta(hidden_states)
    grid = (seq_len, )

    if residual is None:
        rms_norm_kernel[grid](hidden_states,
                              weight,
                              out,
                              input_row_stride=input_stride,
                              eps=eps,
                              N_COLS=feat_size,
                              BLOCK_N=BLOCK_N,
                              num_warps=4,
                              num_stages=2,
                              **kernel_meta)
        return out
    else:
        if out_residual is None:
            out_residual = torch.empty_like(hidden_states)

        res_stride = residual.stride(-2)
        add_rms_norm_kernel[grid](hidden_states,
                                  weight,
                                  residual,
                                  out,
                                  out_residual,
                                  input_row_stride=input_stride,
                                  residual_row_stride=res_stride,
                                  eps=eps,
                                  N_COLS=feat_size,
                                  BLOCK_N=BLOCK_N,
                                  num_warps=4,
                                  num_stages=2,
                                  **kernel_meta)
        return out, out_residual


if __name__ == '__main__':
    import time

    def torch_forward(hidden_states, weight, variance_epsilon=1e-6):
        """pytorch forward."""
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance +
                                                    variance_epsilon)
        return weight * hidden_states.to(input_dtype)

    def test_rms_norm(bsz, ctx_len, feat_len, dtype):
        """test rms norm."""
        input = torch.empty((bsz, ctx_len, feat_len),
                            dtype=dtype,
                            device='cuda').normal_(mean=0.,
                                                   std=0.5).contiguous()
        weight = torch.empty((feat_len), dtype=dtype,
                             device='cuda').normal_(mean=0.,
                                                    std=0.5).contiguous()
        triton_output = rms_norm(hidden_states=input, weight=weight)
        torch_output = torch_forward(hidden_states=input, weight=weight)
        assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=0)

        N_REPEATS = 20

        t0 = time.time()
        for _ in range(N_REPEATS):
            torch_forward(hidden_states=input, weight=weight)

        t1 = time.time()
        for _ in range(N_REPEATS):
            rms_norm(hidden_states=input, weight=weight)
        t2 = time.time()

        torch_cost = (t1 - t0) / N_REPEATS * 1000
        triton_cost = (t2 - t1) / N_REPEATS * 1000
        print(
            'input {} weight {} dtype {}\n  torch {:.3f} triton {:.3f} (ms)\n'.
            format(input.shape, weight.shape, dtype, torch_cost, triton_cost))

    test_rms_norm(1, 8128, 5120, torch.float16)
    test_rms_norm(1, 8128, 5120, torch.float32)
    test_rms_norm(1, 992, 128, torch.float16)
    test_rms_norm(1, 65537, 128, torch.float32)
