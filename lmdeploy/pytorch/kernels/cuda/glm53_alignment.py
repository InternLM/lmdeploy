# Copyright (c) OpenMMLab. All rights reserved.
"""Fixed reduction tiles for GLM projections, independent of request batching."""
import triton
import triton.language as tl


@triton.jit
def router_linear(X, W, Y, M, WS0: tl.constexpr, WS1: tl.constexpr):
    """Fixed GLM FP32 router reduction, with input casts only in registers."""
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    ks = tl.arange(0, 32)
    acc = tl.zeros((32, 64), tl.float32)
    for block in range(128):
        k = block * 32 + ks
        x = tl.load(X + rows[:, None] * 4096 + k[None, :], rows[:, None] < M, 0).to(tl.float32)
        w = tl.load(W + cols[None, :] * WS0 + k[:, None] * WS1, cols[None, :] < 288, 0).to(tl.float32)
        acc += tl.dot(x, w, input_precision='tf32x3')
    tl.store(Y + rows[:, None] * 288 + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < 288))


@triton.jit
def mhc_linear(X, W, Bias, Y, N: tl.constexpr, K: tl.constexpr,
               WS0: tl.constexpr, WS1: tl.constexpr, HAS_BIAS: tl.constexpr,
               COLS: tl.constexpr = 4, BLOCK_K: tl.constexpr = 2048):
    """Row-independent FP32 mHC projection with FP64 register reductions.

    Keep the same column groups and feature reductions for every request
    batch. Only input/output tensors use global memory; FP64 values live in
    registers. Grouping several columns shares each input row's loads.
    """
    pid = tl.program_id(0)
    row = (pid // tl.cdiv(N, COLS)).to(tl.int64)
    cols = pid % tl.cdiv(N, COLS) * COLS + tl.arange(0, COLS)
    ks = tl.arange(0, BLOCK_K)
    acc = tl.zeros((COLS,), tl.float64)
    for block in range(tl.cdiv(K, BLOCK_K)):
        k = block * BLOCK_K + ks
        x = tl.load(X + row * K + k, k < K, 0).to(tl.float64)
        w = tl.load(W + cols[:, None] * WS0 + k[None, :] * WS1,
                    (cols[:, None] < N) & (k[None, :] < K), 0).to(tl.float64)
        acc += tl.sum(w * x[None, :], axis=1)
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, 0).to(tl.float64)
    tl.store(Y + row * N + cols, acc, cols < N)


@triton.jit
def _head_dot(X, W, rows, cols, ks, block, acc, M, N: tl.constexpr,
              K: tl.constexpr, WS0: tl.constexpr, WS1: tl.constexpr):
    k = block * 32 + ks
    x = tl.load(X + rows[:, None] * K + k[None, :],
                (rows[:, None] < M) & (k[None, :] < K), 0)
    w = tl.load(W + cols[None, :] * WS0 + k[:, None] * WS1,
                (cols[None, :] < N) & (k[:, None] < K), 0)
    return tl.dot(x, w, acc, input_precision='tf32x3')


@triton.jit
def head_linear(X, W, Bias, Y, M, N: tl.constexpr, K: tl.constexpr,
                WS0: tl.constexpr, WS1: tl.constexpr, HAS_BIAS: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    """FP32 head with fixed K order and two register accumulation lanes.

    TF32x3 products feed separate FP32 partial sums. Combining the lanes in
    FP64 avoids another FP32 sum rounding, without a global split-K buffer.
    Changing row/column tile counts preserves each output's reduction order.
    """
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, 32)
    even = tl.zeros((BM, BN), tl.float32)
    odd = tl.zeros((BM, BN), tl.float32)
    for block in range(tl.cdiv(K, 64)):
        even = _head_dot(X, W, rows, cols, ks, block * 2, even, M, N, K, WS0, WS1)
        odd = _head_dot(X, W, rows, cols, ks, block * 2 + 1, odd, M, N, K, WS0, WS1)
    result = even.to(tl.float64) + odd.to(tl.float64)
    if HAS_BIAS:
        result += tl.load(Bias + cols, cols < N, 0).to(tl.float64)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], result,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def linear(X, W, Bias, Y, M, N: tl.constexpr, K: tl.constexpr,
           WS0: tl.constexpr, WS1: tl.constexpr, HAS_BIAS: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.program_id(1) * 64 + tl.arange(0, 64)
    ks = tl.arange(0, 32)
    acc = tl.zeros((32, 64), tl.float32)
    for block in range(tl.cdiv(K, 32)):
        k = block * 32 + ks
        x = tl.load(X + rows[:, None] * K + k[None, :], (rows[:, None] < M) & (k[None, :] < K), 0)
        w = tl.load(W + cols[None, :] * WS0 + k[:, None] * WS1,
                    (cols[None, :] < N) & (k[:, None] < K), 0)
        acc += tl.dot(x, w, input_precision='ieee')
    if HAS_BIAS:
        acc += tl.load(Bias + cols, cols < N, 0)[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))
