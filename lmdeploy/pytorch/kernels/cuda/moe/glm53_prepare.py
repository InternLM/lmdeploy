# Copyright (c) OpenMMLab. All rights reserved.
"""GLM routing and exact DeepEP FP8 message preparation in one launch."""
import torch
import triton
import triton.language as tl


@triton.jit
def _glm53_moe_prepare_kernel(Logits, Bias, X, Send, Weights, Ids, Valid,
                              ROWS: tl.constexpr, HAS_VALID: tl.constexpr,
                              IS_DUMMY: tl.constexpr, LOGIT_STRIDE: tl.constexpr,
                              MESSAGE_BYTES: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.arange(0, 512)
    logits = tl.load(Logits + row * LOGIT_STRIDE + expert, expert < 288, 0)
    bias = tl.load(Bias + expert, expert < 288, 0)
    scores = tl.sigmoid(logits)
    candidates = tl.where(expert < 288, scores + bias, -float('inf'))
    positions = tl.arange(0, 8)
    weights = tl.full((8,), 0, tl.float32)
    ids = tl.full((8,), 0, tl.int32)
    for rank in range(8):
        selected = tl.argmax(candidates, axis=0)
        weight = tl.sum(tl.where(expert == selected, scores, 0), axis=0)
        weights += tl.where(positions == rank, weight, 0)
        ids += tl.where(positions == rank, selected, 0)
        candidates = tl.where(expert == selected, -float('inf'), candidates)
    weights /= tl.sum(weights, axis=0) + 1e-20
    # Preserve the FP32 rounding boundary before combine's model-level scale.
    weights = tl.inline_asm_elementwise('mov.b32 $0, $1;', constraints='=f,f',
                                       args=[weights], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Weights + row * 8 + positions, weights * 2.5)
    valid = ROWS
    if HAS_VALID:
        valid = tl.load(Valid)
    elif IS_DUMMY:
        valid = 0
    tl.store(Ids + row * 8 + positions, tl.where(row < valid, ids, -1))

    channels = tl.arange(0, 4096)
    values = tl.load(X + row * 4096 + channels).to(tl.float32)
    grouped = tl.reshape(values, (32, 128))
    amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1e-4)
    # DeepEP's CUDA helper uses RN division and a multiply by float(1/448).
    scale = tl.div_rn(448.0, amax)
    scale_inv = amax * (1.0 / 448.0)
    quantized = tl.reshape((grouped * scale[:, None]).to(tl.float8e4nv), (4096,))
    payload = tl.cast(Send + row * MESSAGE_BYTES + 16, tl.pointer_type(tl.float8e4nv))
    tl.store(payload + channels, quantized)
    scales = tl.cast(Send + row * MESSAGE_BYTES + 16 + 4096, tl.pointer_type(tl.float32))
    tl.store(scales + tl.arange(0, 32), scale_inv)
    header = tl.cast(Send + row * MESSAGE_BYTES, tl.pointer_type(tl.int32))
    fields = tl.arange(0, 4)
    tl.store(header + fields, tl.where(fields == 0, row, 0))


def glm53_prepare(logits: torch.Tensor, bias: torch.Tensor, x: torch.Tensor,
                   send_buffer: torch.Tensor, valid_tokens: torch.Tensor | None,
                   is_dummy: bool = False, num_warps: int = 4):
    """Return scaled combine weights and masked int64 expert IDs.

    Writes directly into the next DeepEP symmetric send buffer. Call the matching prepared dispatch immediately
    afterwards on the same stream.
    """
    assert x.ndim == 2 and x.shape[1] == 4096 and x.is_contiguous()
    assert x.dtype == torch.bfloat16 and x.is_cuda
    assert logits.shape == (x.size(0), 288) and logits.dtype == torch.float32
    assert logits.stride(1) == 1 and bias.shape == (288,) and bias.dtype == torch.float32
    assert send_buffer.dtype == torch.uint8 and send_buffer.is_contiguous()
    assert send_buffer.size(0) >= x.size(0) and send_buffer.size(1) == 16 + 4096 + 32 * 4
    weights = torch.empty((x.size(0), 8), device=x.device, dtype=torch.float32)
    ids = torch.empty((x.size(0), 8), device=x.device, dtype=torch.int64)
    if x.size(0):
        _glm53_moe_prepare_kernel[(x.size(0),)](
            logits, bias, x, send_buffer, weights, ids, valid_tokens,
            x.size(0), valid_tokens is not None, is_dummy, logits.stride(0),
            send_buffer.size(1), num_warps=num_warps, num_stages=1)
    return weights, ids
