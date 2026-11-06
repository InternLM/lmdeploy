# Copyright (c) OpenMMLab. All rights reserved.
"""GLM routing, padding masks and combine scaling for native DeepEP."""
import torch
import triton
import triton.language as tl


@triton.jit
def _glm53_route_prepare_kernel(Logits, Bias, Weights, Ids, Valid,
                              ROWS: tl.constexpr, HAS_VALID: tl.constexpr,
                              IS_DUMMY: tl.constexpr, LOGIT_STRIDE: tl.constexpr):
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

def glm53_prepare_routes(logits: torch.Tensor, bias: torch.Tensor,
                         valid_tokens: torch.Tensor | None,
                         is_dummy: bool = False, num_warps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scaled combine weights and masked int64 expert IDs.

    Native DeepEP dispatch receives the original BF16 hidden states and owns FP8 quantization, packing and
    communication.
    """
    assert logits.ndim == 2 and logits.size(1) == 288 and logits.dtype == torch.float32 and logits.is_cuda
    assert logits.stride(1) == 1 and bias.shape == (288,) and bias.dtype == torch.float32
    assert bias.device == logits.device and bias.is_contiguous()
    weights = torch.empty((logits.size(0), 8), device=logits.device, dtype=torch.float32)
    ids = torch.empty((logits.size(0), 8), device=logits.device, dtype=torch.int64)
    if logits.size(0):
        _glm53_route_prepare_kernel[(logits.size(0),)](
            logits, bias, weights, ids, valid_tokens,
            logits.size(0), valid_tokens is not None, is_dummy, logits.stride(0),
            num_warps=num_warps, num_stages=1)
    return weights, ids
