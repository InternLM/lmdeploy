# Copyright (c) OpenMMLab. All rights reserved.
import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream


@triton.jit
def rms_norm_kernel(input, weight, output, input_row_stride, n_cols, eps,
                    N_COLS: tl.constexpr, BLOCK_N: tl.constexpr):
    """rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < n_cols)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < n_cols)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < n_cols)


@torch.inference_mode()
def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-6):
    """rms norm."""

    def _kernel_meta():
        device = hidden_states.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)

    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)

    out = torch.empty_like(hidden_states)

    kernel_meta = _kernel_meta()
    grid = (seq_len, )
    rms_norm_kernel[grid](hidden_states,
                          weight,
                          out,
                          input_stride,
                          feat_size,
                          eps,
                          feat_size,
                          BLOCK_N,
                          num_warps=4,
                          num_stages=2,
                          **kernel_meta)

    return out


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
