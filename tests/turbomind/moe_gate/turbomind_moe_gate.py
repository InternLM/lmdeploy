from __future__ import annotations

from dataclasses import dataclass

import torch

REQUIRED_NATIVE_BRIDGE_SYMBOLS = ('moe_gate_v2',)
K_MOE_GATE_VEC_SIZE = 4
K_MOE_GATE_MAX_TILES = 16


def _load_native_bridge(required_symbols=REQUIRED_NATIVE_BRIDGE_SYMBOLS):
    try:
        import _turbomind as tm
    except ImportError:
        return None
    return tm if all(hasattr(tm, symbol) for symbol in required_symbols) else None


def is_available() -> bool:
    return _load_native_bridge() is not None


def _current_stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def tokens_padded(tokens: int) -> int:
    return (tokens + K_MOE_GATE_VEC_SIZE - 1) // K_MOE_GATE_VEC_SIZE * K_MOE_GATE_VEC_SIZE


@dataclass
class MoeGateV2Buffers:
    f2n: torch.Tensor
    f2E: torch.Tensor
    en2f: torch.Tensor
    offsets: torch.Tensor
    scales: torch.Tensor
    masks: torch.Tensor
    accum: torch.Tensor


def allocate_moe_gate_v2_buffers(
    tokens: int,
    experts: int,
    experts_per_token: int,
    *,
    device: torch.device | str = 'cuda',
) -> MoeGateV2Buffers:
    flat = experts_per_token * tokens
    padded = tokens_padded(tokens)
    return MoeGateV2Buffers(
        f2n=torch.empty(flat, device=device, dtype=torch.int32),
        f2E=torch.empty(flat, device=device, dtype=torch.int32),
        en2f=torch.empty(experts_per_token, tokens, device=device, dtype=torch.int32),
        offsets=torch.empty(experts + 1, device=device, dtype=torch.int32),
        scales=torch.empty(experts_per_token, tokens, device=device, dtype=torch.float32),
        masks=torch.empty(experts, padded, device=device, dtype=torch.int8),
        accum=torch.empty(experts * K_MOE_GATE_MAX_TILES, device=device, dtype=torch.int32),
    )


def moe_gate_v2(
    logits: torch.Tensor,
    experts_per_token: int,
    *,
    token_mask: torch.Tensor | None = None,
    softmax: bool = True,
    norm_topk: bool = False,
    routed_scale: float = 1.0,
    buffers: MoeGateV2Buffers | None = None,
):
    """Test-oriented wrapper around _turbomind.moe_gate_v2.

    token_mask is a CUDA bool tensor of shape [tokens]; tokens with mask == False are not routed (they contribute no
    f2n/f2E/offsets entries). Defaults to all-True.

    If buffers is None, allocates outputs inside the binding (correctness path). If buffers is provided, writes into
    those tensors (steady-state / bench path) and returns (f2n, f2E, en2f, offsets, scales) as the same torch tensors.
    """
    tm = _load_native_bridge()
    if tm is None:
        raise ImportError('TurboMind moe_gate_v2 bridge is unavailable')
    if not logits.is_cuda or logits.dtype != torch.float32:
        raise ValueError('logits must be CUDA float32')
    logits = logits.contiguous()
    if token_mask is None:
        token_mask = torch.ones(logits.shape[0], device=logits.device, dtype=torch.bool)
    if not token_mask.is_cuda or token_mask.dtype != torch.bool:
        raise ValueError('token_mask must be CUDA bool')
    if token_mask.shape != (logits.shape[0],):
        raise ValueError(f'token_mask must be [{logits.shape[0]}], got {tuple(token_mask.shape)}')
    token_mask = token_mask.contiguous()
    kwargs = dict(
        softmax=bool(softmax),
        norm_topk=bool(norm_topk),
        routed_scale=float(routed_scale),
        stream_ptr=_current_stream_ptr(logits.device),
    )
    if buffers is None:
        outs = tm.moe_gate_v2(
            tm.from_dlpack(logits), tm.from_dlpack(token_mask), int(experts_per_token), **kwargs)
        return tuple(torch.from_dlpack(x) for x in outs)

    outs = tm.moe_gate_v2(
        tm.from_dlpack(logits),
        tm.from_dlpack(token_mask),
        int(experts_per_token),
        f2n=tm.from_dlpack(buffers.f2n),
        f2E=tm.from_dlpack(buffers.f2E),
        en2f=tm.from_dlpack(buffers.en2f),
        offsets=tm.from_dlpack(buffers.offsets),
        scales=tm.from_dlpack(buffers.scales),
        masks=tm.from_dlpack(buffers.masks),
        accum=tm.from_dlpack(buffers.accum),
        **kwargs,
    )
    # Prefer returning caller-owned torch buffers (ignore returned TM tensors).
    _ = outs
    return buffers.f2n, buffers.f2E, buffers.en2f, buffers.offsets, buffers.scales
