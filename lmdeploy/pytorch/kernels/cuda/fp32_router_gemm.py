# Copyright (c) OpenMMLab. All rights reserved.
"""Small-token FP32 MoE router GEMM.

The kernel follows the one-block-per-expert design from vLLM's FP32 router GEMM, while using Triton so it fits
LMDeploy's PyTorch kernel stack. See vLLM PR #48335 and its source implementation:
https://github.com/vllm-project/vllm/pull/48335
https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/fp32_router_gemm.cu
"""

import functools

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_SUPPORTED_HIDDEN_SIZES = (6144, 7168)
_NUM_EXPERTS = 256
_MAX_TOKENS = 32


@functools.cache
def _is_sm90_or_newer(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device)[0] >= 9


@triton.jit
def _fp32_router_gemm_kernel(
    hidden_states,
    weight,
    output,
    num_tokens,
    stride_hm,
    stride_hk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    hidden_size: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    expert_id = tl.program_id(0)
    token_ids = tl.program_id(1) * block_m + tl.arange(0, block_m)
    k_ids = tl.arange(0, block_k)
    token_mask = token_ids < num_tokens
    accumulator = tl.zeros((block_m, ), dtype=tl.float32)

    for k_start in range(0, hidden_size, block_k):
        weight_vals = tl.load(weight + expert_id * stride_wn + (k_start + k_ids) * stride_wk).to(tl.float32)
        hidden_vals = tl.load(hidden_states + token_ids[:, None] * stride_hm + (k_start + k_ids)[None, :] * stride_hk,
                              mask=token_mask[:, None],
                              other=0.0).to(tl.float32)
        accumulator += tl.sum(hidden_vals * weight_vals[None, :], axis=1)

    tl.store(output + token_ids * stride_om + expert_id * stride_on, accumulator, mask=token_mask)


def _is_supported(hidden_states: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether the measured small-token Triton path applies."""
    input_supported = hidden_states.dtype == torch.bfloat16 or (hidden_states.dtype == torch.float32
                                                                and hidden_states.size(0) <= 8)
    return (hidden_states.is_cuda and input_supported
            and weight.device == hidden_states.device and weight.dtype == torch.float32 and weight.is_contiguous()
            and weight.size(0) == _NUM_EXPERTS and weight.size(1) in _SUPPORTED_HIDDEN_SIZES
            and hidden_states.size(1) == weight.size(1) and 0 < hidden_states.size(0) <= _MAX_TOKENS
            and _is_sm90_or_newer(hidden_states.device))


def _launch(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Launch the small-token kernel."""
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens <= 8:
        block_m = triton.next_power_of_2(num_tokens)
        block_k = 1024
        num_warps = 8
    elif num_tokens <= 16:
        block_m = 8 if hidden_size == 6144 else 16
        block_k = 1024
        num_warps = 4 if hidden_size == 6144 else 8
    elif num_tokens <= 24:
        block_m = 8
        block_k = 1024
        num_warps = 4
    else:
        block_m = 16
        block_k = 256
        num_warps = 4

    output = torch.empty((num_tokens, _NUM_EXPERTS), dtype=torch.float32, device=hidden_states.device)
    grid = (_NUM_EXPERTS, triton.cdiv(num_tokens, block_m))
    _fp32_router_gemm_kernel[grid](
        hidden_states,
        weight,
        output,
        num_tokens,
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        hidden_size=hidden_size,
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def fp32_router_gemm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute FP32 router logits, using the fast path for measured shapes."""
    hidden_states = hidden_states.flatten(0, -2)
    if _is_supported(hidden_states, weight):
        return _launch(hidden_states, weight)
    return F.linear(hidden_states.to(weight.dtype), weight)
