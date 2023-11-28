# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def per_channel_quant(x, n_bits, dtype):
    """Quantize the input tensor 'x' channel-wise using the given number of
    bits.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be a
            2-dimensional tensor.
        n_bits (int): The number of bits to use for quantization.
        dtype (torch.dtype): The data type to which the quantized tensor should
            be converted.

    Returns:
        tuple: A tuple containing two items -- the quantized tensor and
            the scale used for quantization.
    """
    assert x.ndim == 2
    x = x.to(torch.float32)
    x_absmax = x.view(x.shape[0], -1).abs().max(dim=1, keepdim=True)[0]
    q_max = 2**(n_bits - 1) - 1
    q_min = -2**(n_bits - 1)
    scale = x_absmax / (2**(n_bits - 1) - 1)
    x_q = torch.round(x / scale).clamp(q_min, q_max).to(dtype)
    return x_q, scale


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M': 16,
            'BLOCK_N': 128,
            'BLOCK_K': 256,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 32,
            'BLOCK_N': 64,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rms_scale_ptr,
    linear_scale_ptr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M': 16,
            'BLOCK_N': 128,
            'BLOCK_K': 256,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 32,
            'BLOCK_N': 64,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 64,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 64,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4),
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        },
                      num_stages=4,
                      num_warps=4)
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_add(
    A,
    B,
    C,
    residual_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    rms_scale_ptr,
    linear_scale_ptr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale
    c = c.to(residual_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    residual_ptrs = (residual_ptr + stride_cm * offs_cm[:, None] +
                     stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    residual = tl.load(residual_ptrs, mask=c_mask, other=0.)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c + residual, mask=c_mask)


def matmul_kernel_dynamic_quant(a,
                                b,
                                rms_scale,
                                linear_scale,
                                residual=None,
                                bias=None,
                                output_dtype=torch.float16):
    assert a.shape[-1] == b.shape[-1]
    assert b.ndim == 2 and b.is_contiguous()
    b = b.t()  # (K, N)
    M = a.numel() // a.shape[-1]
    K, N = b.shape
    c_shape = a.shape[:-1] + (N, )
    if residual is not None:
        assert residual.shape == c_shape
        assert residual.is_contiguous()
    c = torch.empty(c_shape, device=a.device, dtype=output_dtype)

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) *
                triton.cdiv(N, META['BLOCK_N']), )

    if residual is not None:
        _linear_add[grid](a,
                          b,
                          c,
                          residual,
                          M,
                          N,
                          K,
                          a.stride(-2),
                          a.stride(-1),
                          b.stride(0),
                          b.stride(1),
                          c.stride(-2),
                          c.stride(-1),
                          GROUP_SIZE_M=8,
                          rms_scale_ptr=rms_scale,
                          linear_scale_ptr=linear_scale)
    else:
        _linear[grid](a,
                      b,
                      c,
                      M,
                      N,
                      K,
                      a.stride(-2),
                      a.stride(-1),
                      b.stride(0),
                      b.stride(1),
                      c.stride(-2),
                      c.stride(-1),
                      GROUP_SIZE_M=8,
                      rms_scale_ptr=rms_scale,
                      linear_scale_ptr=linear_scale)
    if bias is not None:
        c += bias

    return c


# def matmul_kernel_dynamic_quant(x,
#                                         b,
#                                         rms_scale,
#                                         linear_scale,
#                                         residual=None,
#                                         bias=None,
#                                         output_dtype=torch.float16):
#     assert x.shape[-1] == b.shape[-1]
#     assert b.is_contiguous()
#     a = x  # (M, K)
#     b = b.t()  # (K, N)
#     assert b.stride(0) == 1
#     M, K = a.shape
#     K, N = b.shape
#     if residual is not None:
#         assert residual.shape == (M, N)
#         assert residual.is_contiguous()
#     c = torch.empty((M, N), device=a.device, dtype=output_dtype)

#     def grid(META):
#         return (triton.cdiv(M, META['BLOCK_M']) *
#                 triton.cdiv(N, META['BLOCK_N']), )

#     if residual is not None:
#         _linear_add[grid](a,
#                           b,
#                           c,
#                           residual,
#                           M,
#                           N,
#                           K,
#                           a.stride(0),
#                           a.stride(1),
#                           b.stride(0),
#                           b.stride(1),
#                           c.stride(0),
#                           c.stride(1),
#                           GROUP_SIZE_M=8,
#                           rms_scale_ptr=rms_scale,
#                           linear_scale_ptr=linear_scale)
#     else:
#         _linear[grid](a,
#                       b,
#                       c,
#                       M,
#                       N,
#                       K,
#                       a.stride(0),
#                       a.stride(1),
#                       b.stride(0),
#                       b.stride(1),
#                       c.stride(0),
#                       c.stride(1),
#                       GROUP_SIZE_M=8,
#                       rms_scale_ptr=rms_scale,
#                       linear_scale_ptr=linear_scale)
#     if bias is not None:
#         c += bias

#     return c


@triton.jit
def _per_token_quant_int8(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    y_stride,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    y_ptr += row * y_stride
    y_q_ptr += row * y_stride
    y_s_ptr += row

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / 127
    y_q = tl.maximum(tl.minimum(tl.math.round(y / y_s), 127), -128).to(tl.int8)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_quant_int8(x, eps):
    x_q = torch.empty_like(x, device=x.device, dtype=torch.int8)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_s = torch.empty(x.shape[:-1] + (1, ),
                      device=x.device,
                      dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)
    # enqueue kernel
    _per_token_quant_int8[(M, )](x,
                                 x_q,
                                 x_s,
                                 M,
                                 N,
                                 eps,
                                 BLOCK=BLOCK,
                                 num_warps=num_warps)
    return x_q, x_s


# def per_token_quant_int8(x, eps):
#     M, N = x.shape
#     x_q = torch.empty((M, N), device=x.device, dtype=torch.int8)
#     x_q_arg = x_q.reshape(-1, x_q.shape[-1])
#     x_s = torch.empty(M, device=x.device, dtype=torch.float32)
#     BLOCK = triton.next_power_of_2(N)
#     # heuristics for number of warps
#     num_warps = min(max(BLOCK // 256, 1), 8)
#     # enqueue kernel
#     _per_token_quant_int8[(M, )](x,
#                                  x_q,
#                                  x_s,
#                                  x_q_arg.stride(0),
#                                  N,
#                                  eps,
#                                  BLOCK=BLOCK,
#                                  num_warps=num_warps)
#     return x_q, x_s


@triton.jit
def _rms_norm_fwd_fused_dynamic_symmetric(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    Scale,  # pointer to the scales of the output activation
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    cols = tl.arange(0, BLOCK_SIZE)
    x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
    _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    w = tl.load(W + cols, mask=mask)
    x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
    x_hat = x * rstd
    y = x_hat * w

    scale = tl.max(tl.abs(y)).to(tl.float32) / 127
    tl.store(Scale + row, scale)

    y = tl.math.round(y / scale)
    y = tl.minimum(y, 127)
    y = tl.maximum(y, -128)
    tl.store(Y + cols, y, mask=mask)


def rms_norm_dynamic_quant(x, w, eps):
    x_arg = x.reshape(-1, x.shape[-1])
    y = torch.empty_like(x, dtype=torch.int8)
    M, K = x_arg.shape
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(K))
    if K > BLOCK_SIZE:
        raise RuntimeError(
            "This rms norm doesn't support feature dim >= 64KB.")
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    scale = torch.empty(x.shape[:-1] + (1, ),
                        dtype=torch.float32,
                        device=x.device)
    _rms_norm_fwd_fused_dynamic_symmetric[(M, )](
        x_arg,
        y,
        w,
        scale,
        x_arg.stride(0),
        K,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y, scale


def test_rms_and_linear(x,
                        rms_weight,
                        linear_weight,
                        dtype=torch.float16,
                        eps=1e-5):

    def rms_norm_torch(x, w, eps):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return w * x

    def linear_torch(x, b):
        return F.linear(x, b)

    linear_weight_quant, linear_scale = per_channel_quant(
        linear_weight, 8, torch.int8)

    rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)
    assert rms_out.shape == x.shape and rms_scale.shape[:-1] == x.shape[:-1]
    linear_out = matmul_kernel_dynamic_quant(rms_out,
                                             linear_weight_quant,
                                             rms_scale,
                                             linear_scale,
                                             output_dtype=dtype)

    rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()
    linear_out_torch = linear_torch(rms_out_torch, linear_weight)
    print(f'linear_out.abs().mean() = {linear_out.abs().mean()}')
    print(f'linear_out_torch.abs().mean() = {linear_out_torch.abs().mean()}')
    print('perchannel error: ', (linear_out - linear_out_torch).abs().mean())
    cos = torch.nn.CosineSimilarity(0)
    print(
        'Output cos',
        cos(linear_out.flatten().to(torch.float32),
            linear_out_torch.flatten().to(torch.float32)))


def test_per_token_quant(x, eps):

    def per_token_quant_int8_torch(x, eps):
        _absmax = torch.clamp(x.abs().max(dim=-1, keepdim=True)[0], min=eps)
        x_s = _absmax / 127
        x_q = torch.clamp((x / x_s).round(), min=-128, max=127)
        return x_q, x_s

    x_q, x_s = per_token_quant_int8(x, eps)
    x_q_torch, x_s_torch = per_token_quant_int8_torch(x, eps)
    assert x_q.shape == x_q_torch.shape and x_s.shape == x_s_torch.shape
    cos = torch.nn.CosineSimilarity(0)
    print(
        'x_q cos',
        cos(x_q.flatten().to(torch.float32),
            x_q_torch.flatten().to(torch.float32)))
    print(
        'x_s cos',
        cos(x_s.flatten().to(torch.float32),
            x_s_torch.flatten().to(torch.float32)))


# def test_rms_and_linear(M, N, K, dtype=torch.float16, eps=1e-5, device='cuda'):

#     def rms_norm_torch(x, w, eps):
#         variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
#         x = x * torch.rsqrt(variance + eps)
#         # print(w * x)
#         return w * x

#     def linear_torch(x, b):
#         return F.linear(x, b)

#     torch.manual_seed(0)
#     x_shape = (M, K)
#     rms_w_shape = (x_shape[-1], )
#     rms_weight = torch.randn(rms_w_shape,
#                              dtype=dtype,
#                              device='cuda',
#                              requires_grad=True)
#     x = torch.randn(x_shape, dtype=dtype, device='cuda')
#     linear_weight = torch.randn((N, K),
#                                 dtype=dtype,
#                                 device='cuda',
#                                 requires_grad=True)

#     linear_weight_quant, linear_scale = per_channel_quant(
#         linear_weight, 8, torch.int8)
#     rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)
#     linear_out = matmul_kernel_dynamic_quant(rms_out,
#                                                      linear_weight_quant,
#                                                      rms_scale,
#                                                      linear_scale,
#                                                      output_dtype=dtype)

#     rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()
#     linear_out_torch = linear_torch(rms_out_torch, linear_weight)
#     print(f'linear_out.abs().mean() = {linear_out.abs().mean()}')
#     print(f'linear_out_torch.abs().mean() = {linear_out_torch.abs().mean()}')
#     print('perchannel error: ', (linear_out - linear_out_torch).abs().mean())
#     cos = torch.nn.CosineSimilarity(0)
#     print(
#         'Output cos',
#         cos(linear_out.flatten().to(torch.float32),
#             linear_out_torch.flatten().to(torch.float32)))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M'],
        x_vals=[1, 16, 32, 64, 128, 256] + [512 * i * 2 for i in range(1, 17)],
        line_arg='provider',
        line_vals=['int8_dynamic_triton_op', 'float_torch'],
        line_names=['int8_dynamic_triton_op', 'float_torch'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'),
                ('yellow', '-'), ('yellow', '-')],
        ylabel='GB/s',
        plot_name='forward',
        args={
            'dtype': torch.float16,
        }))
def bench_rms_and_linear(M, dtype, provider, eps=1e-5, device='cuda'):

    def rms_norm_torch(x, w, eps):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return w * x

    def linear_torch(x, b):
        return F.linear(x, b)

    N = 4096
    K = 4096
    x_shape = (M, K)
    rms_w_shape = (x_shape[-1], )
    rms_weight = torch.randn(rms_w_shape,
                             dtype=dtype,
                             device='cuda',
                             requires_grad=True)
    x = torch.randn(x_shape, dtype=dtype, device='cuda')
    linear_weight = torch.randn((N, K),
                                dtype=dtype,
                                device='cuda',
                                requires_grad=True)
    linear_weight_quant, linear_scale = per_channel_quant(
        linear_weight, 8, torch.int8)

    alpha = max(x.max().abs(), x.min().abs())
    rms_scale = alpha / 127

    if provider == 'int8_dynamic_triton_op':
        rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps)

        def y_fwd():

            matmul_kernel_dynamic_quant(rms_out,
                                        linear_weight_quant,
                                        rms_scale,
                                        linear_scale,
                                        output_dtype=dtype)
    elif provider == 'float_torch':
        rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()

        def y_fwd():
            linear_torch(rms_out_torch, linear_weight)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd,
                                                 quantiles=quantiles,
                                                 rep=500)
    return ms, max_ms, min_ms


if __name__ == '__main__':
    torch.manual_seed(0)
    dtype = torch.float16
    # test (bs, seq_len, dim) x (dim, out_dim)
    x = torch.randn((2, 2048, 4096), dtype=dtype, device='cuda')
    rms_weight = torch.randn((4096, ),
                             dtype=dtype,
                             device='cuda',
                             requires_grad=True)

    linear_weight = torch.randn((11008, 4096),
                                dtype=dtype,
                                device='cuda',
                                requires_grad=True)
    test_rms_and_linear(x, rms_weight, linear_weight)

    # test (M, K) x (K, N)
    x = torch.randn((4, 4096), dtype=dtype, device='cuda')
    rms_weight = torch.randn((4096, ),
                             dtype=dtype,
                             device='cuda',
                             requires_grad=True)

    linear_weight = torch.randn((2048, 4096),
                                dtype=dtype,
                                device='cuda',
                                requires_grad=True)
    test_rms_and_linear(x, rms_weight, linear_weight)

    # test per-token quant
    x = torch.randn((2, 2048, 4096), dtype=dtype, device='cuda')
    eps = 1e-7
    test_per_token_quant(x, eps)

    bench_rms_and_linear.run(print_data=True)
