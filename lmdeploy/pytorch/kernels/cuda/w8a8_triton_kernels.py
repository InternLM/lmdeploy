# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from packaging import version

from ..default.w8a8_kernels import per_channel_quant

TRITON_VERSION = version.parse(triton.__version__)
if TRITON_VERSION >= version.parse('3.0.0'):
    tl_round = tl.extra.cuda.libdevice.round
else:
    tl_round = tl.math.round


@triton.autotune(
    configs=[
        triton.Config({
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 128,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        }, num_stages=3, num_warps=8)
    ],
    key=['N', 'K'],
)
@triton.jit(do_not_specialize=['M'])
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
    ACCUMULATOR_DTYPE: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B`, and store the result in output
    tensor `C`.

    The function applies auto-tuning for optimal performance and uses Just-in- Time compilation.
    """

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
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACCUMULATOR_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=None)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=None)
        accumulator = tl.dot(a, b, accumulator, out_dtype=ACCUMULATOR_DTYPE)
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
            'BLOCK_M': 128,
            'BLOCK_N': 256,
            'BLOCK_K': 128,
        }, num_stages=3, num_warps=8),
        triton.Config({
            'BLOCK_M': 256,
            'BLOCK_N': 128,
            'BLOCK_K': 128,
        }, num_stages=3, num_warps=8)
    ],
    key=['N', 'K'],
)
@triton.jit(do_not_specialize=['M'])
def _linear_add(A, B, C, residual_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                rms_scale_ptr, linear_scale_ptr, ACCUMULATOR_DTYPE: tl.constexpr):
    """Triton-accelerated function used to perform a linear operation (dot
    product) on input tensors `A` and `B`, with addition of residual.

    The result is stored in tensor `C`. The function applies auto-tuning for optimal performance and uses Just-in-Time
    compilation.
    """

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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACCUMULATOR_DTYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=None)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=None)
        accumulator = tl.dot(a, b, accumulator, out_dtype=ACCUMULATOR_DTYPE)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = accumulator.to(tl.float32)

    rms_scale = tl.load(rms_scale_ptr + offs_am)[:, None]
    linear_scale = tl.load(linear_scale_ptr + offs_bn)[None, :]
    c = c * rms_scale * linear_scale
    c = c.to(residual_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    residual_ptrs = (residual_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    residual = tl.load(residual_ptrs, mask=c_mask, other=0.)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, c + residual, mask=c_mask)


def matmul_kernel_dynamic_quant(a, b, rms_scale, linear_scale, residual=None, bias=None, output_dtype=torch.float16):
    """This function performs matrix multiplication with dynamic quantization.

    It takes two input tensors `a` and `b`, scales them with `rms_scale` and `linear_scale`, and optionally adds a
    `residual` tensor and a `bias`. The output is returned in the specified `output_dtype`.
    """

    assert a.shape[-1] == b.shape[-1]
    assert b.ndim == 2 and b.is_contiguous()
    M = a.numel() // a.shape[-1]
    N, K = b.shape
    c_shape = a.shape[:-1] + (N, )
    if residual is not None:
        assert residual.shape == c_shape
        assert residual.is_contiguous()
    c = a.new_empty(c_shape, dtype=output_dtype)
    accumulator_dtype = tl.float32 if a.is_floating_point() else tl.int32

    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

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
                          b.stride(1),
                          b.stride(0),
                          c.stride(-2),
                          c.stride(-1),
                          GROUP_SIZE_M=8,
                          rms_scale_ptr=rms_scale,
                          linear_scale_ptr=linear_scale,
                          ACCUMULATOR_DTYPE=accumulator_dtype)
    else:
        _linear[grid](a,
                      b,
                      c,
                      M,
                      N,
                      K,
                      a.stride(-2),
                      a.stride(-1),
                      b.stride(1),
                      b.stride(0),
                      c.stride(-2),
                      c.stride(-1),
                      GROUP_SIZE_M=8,
                      rms_scale_ptr=rms_scale,
                      linear_scale_ptr=linear_scale,
                      ACCUMULATOR_DTYPE=accumulator_dtype)
    if bias is not None:
        c += bias

    return c


@triton.jit
def _per_token_quant_int8(
        y_ptr,
        y_q_ptr,
        y_s_ptr,
        y_stride: tl.constexpr,
        yq_stride: tl.constexpr,
        N,  # number of columns in X
        eps: tl.constexpr,  # epsilon to avoid division by zero
        BLOCK: tl.constexpr,
        Q_MAX: tl.constexpr,
        IS_FLOATING_POINT: tl.constexpr,  # True for floating point dtype
):
    """A Triton-accelerated function to perform per-token quantization on a
    tensor.

    This function converts the tensor values into signed 8-bit integers.
    """
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    y_ptr += row * y_stride
    y_q_ptr += row * yq_stride
    y_s_ptr += row

    cols = tl.arange(0, BLOCK)  # N <= BLOCK
    mask = cols < N

    y = tl.load(y_ptr + cols, mask=mask, other=0.).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / Q_MAX
    y_q = y / y_s
    if not IS_FLOATING_POINT:
        y_q = tl_round(y_q).to(tl.int8)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


def per_token_quant_int8(x, eps, quant_dtype=torch.int8):
    """Function to perform per-token quantization on an input tensor `x`.

    It converts the tensor values into signed 8-bit integers and returns the quantized tensor along with the scaling
    factor used for quantization.
    """
    qdtype_info = torch.finfo(quant_dtype) if quant_dtype.is_floating_point else torch.iinfo(quant_dtype)
    q_max = qdtype_info.max
    x_q = torch.empty_like(x, device=x.device, dtype=quant_dtype)
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_s = torch.empty(x.shape[:-1] + (1, ), device=x.device, dtype=torch.float32)
    BLOCK = triton.next_power_of_2(N)
    # heuristics for number of warps
    num_warps = min(max(BLOCK // 256, 1), 8)

    if x.dim() > 2:
        x = x.flatten(0, -2)
    assert x.stride(-1) == 1
    # enqueue kernel
    _per_token_quant_int8[(M, )](x,
                                 x_q,
                                 x_s,
                                 y_stride=x.stride(-2),
                                 yq_stride=x_q.stride(-2),
                                 N=N,
                                 eps=eps,
                                 BLOCK=BLOCK,
                                 Q_MAX=q_max,
                                 IS_FLOATING_POINT=quant_dtype.is_floating_point,
                                 num_warps=num_warps)

    return x_q, x_s


@triton.jit
def _compute_rms_norm(x, w, eps: tl.constexpr, N_COLS: tl.constexpr):
    """Compute rms norm."""
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf * tl.math.rsqrt(var + eps)
    out = (w * out).to(x.dtype)
    return out


@triton.jit
def rms_norm_quant_kernel(
    input,
    weight,
    output,
    out_scale,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_MIN: tl.constexpr,
    Q_MAX: tl.constexpr,
    IS_FLOATING_POINT: tl.constexpr,
):
    """Rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    out = _compute_rms_norm(x, w, eps, N_COLS)

    scale = tl.max(tl.abs(out)).to(tl.float32) / Q_MAX
    out_s_ptr = out_scale + prog_id
    tl.store(out_s_ptr, scale)
    out = out / scale
    if not IS_FLOATING_POINT:
        out = tl_round(out)
    out = tl.clamp(out, Q_MIN, Q_MAX)
    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


@triton.jit
def add_rms_norm_quant_kernel(
    input,
    weight,
    residual,
    output,
    out_scale,
    out_residual,
    input_row_stride: tl.constexpr,
    residual_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_MIN: tl.constexpr,
    Q_MAX: tl.constexpr,
    IS_FLOATING_POINT: tl.constexpr,
):
    """Rms norm kernel."""
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

    scale = tl.max(tl.abs(out)).to(tl.float32) / Q_MAX
    out_s_ptr = out_scale + prog_id
    tl.store(out_s_ptr, scale)
    out = out / scale
    if not IS_FLOATING_POINT:
        out = tl_round(out)
    out = tl.clamp(out, Q_MIN, Q_MAX)
    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def rms_norm_dynamic_quant(x, w, eps, residual=None, quant_dtype=torch.int8):
    """Performs RMS normalization with dynamic quantization.

    The function reshapes the input tensor `x`, creates an empty tensor `y` with the same shape as `x`, and calculates
    RMS normalization on the reshaped `x` using a Triton kernel `rms_norm_quant_kernel`.
    """
    qdtype_info = torch.finfo(quant_dtype) if quant_dtype.is_floating_point else torch.iinfo(quant_dtype)
    y = torch.empty_like(x, dtype=quant_dtype)
    scale = x.new_empty(x.shape[:-1] + (1, ), dtype=torch.float32)

    feat_size = w.shape[0]
    seq_len = x.numel() // x.size(-1)
    input_stride = x.stride(-2)
    BLOCK_N = triton.next_power_of_2(feat_size)
    grid = (seq_len, )

    if residual is None:
        rms_norm_quant_kernel[grid](x,
                                    w,
                                    y,
                                    scale,
                                    input_row_stride=input_stride,
                                    eps=eps,
                                    N_COLS=feat_size,
                                    BLOCK_N=BLOCK_N,
                                    Q_MIN=qdtype_info.min,
                                    Q_MAX=qdtype_info.max,
                                    IS_FLOATING_POINT=quant_dtype.is_floating_point,
                                    num_warps=4,
                                    num_stages=2)
        return y, scale
    else:
        out_residual = torch.empty_like(x)
        res_stride = residual.stride(-2)
        add_rms_norm_quant_kernel[grid](x,
                                        w,
                                        residual,
                                        y,
                                        scale,
                                        out_residual,
                                        input_row_stride=input_stride,
                                        residual_row_stride=res_stride,
                                        eps=eps,
                                        N_COLS=feat_size,
                                        BLOCK_N=BLOCK_N,
                                        Q_MIN=qdtype_info.min,
                                        Q_MAX=qdtype_info.max,
                                        IS_FLOATING_POINT=quant_dtype.is_floating_point,
                                        num_warps=4,
                                        num_stages=2)
        return y, scale, out_residual


def test_rms_and_linear(x, rms_weight, linear_weight, output_dtype=torch.float16, quant_dtype=torch.int8, eps=1e-5):
    """Test quantized rms norm and quantized linear layer."""

    def rms_norm_torch(x, w, eps):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return w * x

    def linear_torch(x, b):
        return F.linear(x, b)

    linear_weight_quant, linear_scale = per_channel_quant(linear_weight, quant_dtype)

    rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps, quant_dtype=quant_dtype)
    assert rms_out.shape == x.shape and rms_scale.shape[:-1] == x.shape[:-1]
    linear_out = matmul_kernel_dynamic_quant(rms_out,
                                             linear_weight_quant,
                                             rms_scale,
                                             linear_scale,
                                             output_dtype=output_dtype)

    rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()
    linear_out_torch = linear_torch(rms_out_torch, linear_weight)
    print(f'linear_out.abs().mean() = {linear_out.abs().mean()}')
    print(f'linear_out_torch.abs().mean() = {linear_out_torch.abs().mean()}')
    print('perchannel error: ', (linear_out - linear_out_torch).abs().mean())
    cos = torch.nn.CosineSimilarity(0)
    print('Output cos', cos(linear_out.flatten().to(torch.float32), linear_out_torch.flatten().to(torch.float32)))


def test_per_token_quant(x, eps, quant_dtype=torch.int8):
    """Test per-token quantization."""

    def per_token_quant_int8_torch(x, eps, quant_dtype):
        qdtype_info = torch.finfo(quant_dtype) if quant_dtype.is_floating_point else torch.iinfo(quant_dtype)

        _absmax = torch.clamp(x.abs().max(dim=-1, keepdim=True)[0], min=eps)
        x_s = _absmax / qdtype_info.max
        x_q = x / x_s
        if not quant_dtype.is_floating_point:
            x_q = x_q.round()
        x_q = torch.clamp(x_q, min=qdtype_info.min, max=qdtype_info.max)
        return x_q, x_s

    x_q, x_s = per_token_quant_int8(x, eps, quant_dtype=quant_dtype)
    x_q_torch, x_s_torch = per_token_quant_int8_torch(x, eps, quant_dtype=quant_dtype)
    assert x_q.shape == x_q_torch.shape and x_s.shape == x_s_torch.shape
    cos = torch.nn.CosineSimilarity(0)
    print('x_q cos', cos(x_q.flatten().to(torch.float32), x_q_torch.flatten().to(torch.float32)))
    print('x_s cos', cos(x_s.flatten().to(torch.float32), x_s_torch.flatten().to(torch.float32)))


def bench_rms_and_linear(M: int, provider: str, dtype: torch.dtype = torch.float16, eps: float = 1e-5):
    """Benchmark rms and linear."""

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
    rms_weight = torch.randn(rms_w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = torch.randn(x_shape, dtype=dtype, device='cuda')
    linear_weight = torch.randn((N, K), dtype=dtype, device='cuda', requires_grad=True)

    if provider == 'torch_fp16':
        rms_out_torch = rms_norm_torch(x, rms_weight, eps).half()

        def y_fwd():
            linear_torch(rms_out_torch, linear_weight)
    else:
        if provider == 'triton_int8':
            quant_dtype = torch.int8
        elif provider == 'triton_fp8_e4m3':
            quant_dtype = torch.float8_e4m3fn
        elif provider == 'triton_fp8_e5m2':
            quant_dtype = torch.float8_e5m2

        linear_weight_quant, linear_scale = per_channel_quant(linear_weight, quant_dtype)

        alpha = max(x.max().abs(), x.min().abs())
        if quant_dtype.is_floating_point:
            qdtype_info = torch.finfo(quant_dtype)
        else:
            qdtype_info = torch.iinfo(quant_dtype)
        rms_scale = alpha / qdtype_info.max
        rms_out, rms_scale = rms_norm_dynamic_quant(x, rms_weight, eps, quant_dtype=quant_dtype)

        def y_fwd():

            matmul_kernel_dynamic_quant(rms_out, linear_weight_quant, rms_scale, linear_scale, output_dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    def perf(ms):
        return 2 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    torch.manual_seed(0)
    device_map = torch.cuda.get_device_capability()
    is_fp8_supported = device_map[0] >= 9
    dtype = torch.float16
    # test (bs, seq_len, dim) x (dim, out_dim)
    x = torch.randn((2, 2048, 4096), dtype=dtype, device='cuda')
    rms_weight = torch.randn((4096, ), dtype=dtype, device='cuda', requires_grad=True)

    linear_weight = torch.randn((11008, 4096), dtype=dtype, device='cuda', requires_grad=True)
    test_rms_and_linear(x, rms_weight, linear_weight, quant_dtype=torch.int8)
    if is_fp8_supported:
        test_rms_and_linear(x, rms_weight, linear_weight, quant_dtype=torch.float8_e4m3fn)
        test_rms_and_linear(x, rms_weight, linear_weight, quant_dtype=torch.float8_e5m2)

    # test (M, K) x (K, N)
    x = torch.randn((4, 4096), dtype=dtype, device='cuda')
    rms_weight = torch.randn((4096, ), dtype=dtype, device='cuda', requires_grad=True)

    linear_weight = torch.randn((2048, 4096), dtype=dtype, device='cuda', requires_grad=True)
    test_rms_and_linear(x, rms_weight, linear_weight, quant_dtype=torch.int8)
    if is_fp8_supported:
        test_rms_and_linear(x, rms_weight, linear_weight, quant_dtype=torch.float8_e4m3fn)
        test_rms_and_linear(x, rms_weight, linear_weight, quant_dtype=torch.float8_e5m2)

    # test per-token quant
    x = torch.randn((4, 2048, 4096), dtype=dtype, device='cuda')
    eps = 1e-7
    test_per_token_quant(x, eps, quant_dtype=torch.int8)
    if is_fp8_supported:
        test_per_token_quant(x, eps, quant_dtype=torch.float8_e4m3fn)
        test_per_token_quant(x, eps, quant_dtype=torch.float8_e5m2)

    # benchmark triton kernels
    line_vals = ['triton_int8', 'torch_fp16']
    line_names = ['triton_int8', 'torch_fp16']

    if is_fp8_supported:
        line_vals += ['triton_fp8_e4m3', 'triton_fp8_e5m2']
        line_names += ['triton_fp8_e4m3', 'triton_fp8_e5m2']
    config = triton.testing.Benchmark(x_names=['M'],
                                      x_vals=[1, 16, 32, 64, 128, 256] + [512 * i * 2 for i in range(1, 5)],
                                      line_arg='provider',
                                      line_vals=line_vals,
                                      line_names=line_names,
                                      styles=[('blue', '-'), ('green', '-'), ('orange', '-'), ('black', '-'),
                                              ('yellow', '-')],
                                      ylabel='TFLOPS',
                                      plot_name='bench-triton',
                                      args={
                                          'dtype': torch.float16,
                                      })
    bench_funch = (triton.testing.perf_report(config))(bench_rms_and_linear)
    bench_funch.run(print_data=True)
