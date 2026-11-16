# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Inference-only chunk_gated_delta_rule wrapper.
# Kernel from https://gitcode.com/Ascend/triton-ascend-kernels/blob/master/src/triton_ascend_kernels/attention/fla/
# Original source: https://github.com/fla-org/flash-linear-attention (MIT)
# mypy: ignore-errors

import torch

from .prefill import chunk_gated_delta_rule_fwd
from .prefill.l2norm import l2norm_fwd


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    r"""
    Inference-only chunk gated delta rule (no autograd wrapper).
    Kernel: lmdeploy.pytorch.kernels.ascend.triton_ops.fla.prefill.chunk_gated_delta_rule_fwd

    Args:
        q: queries  `[B, T, H, K]`.
        k: keys     `[B, T, H, K]`.
        v: values   `[B, T, H, V]`.
        g: gating (log space) `[B, T, H]`.
        beta: betas `[B, T, H]`.
        scale: attention scale, defaults to `1 / sqrt(K)`.
        initial_state: `[N, H, K, V]`.
        output_final_state: whether to return final state `[N, H, K, V]`.
        cu_seqlens: `[N+1]` for variable-length sequences.
        head_first: kept for API compat, must be False.
    Returns:
        (o, final_state): o is `[B, T, H, V]`, final_state is `[N, H, K, V]` or None.
    """
    assert not head_first, 'head_first=True is not supported.'
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, 'chunk_gated_delta_rule does not support float32. Please use bfloat16.'
    assert len(beta.shape) == 3

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(f"Batch size must be 1 with cu_seqlens, got {q.shape[0]}.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"initial_state batch ({initial_state.shape[0]}) != num sequences ({len(cu_seqlens) - 1})."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    input_dtype = q.dtype
    o, final_state = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    return o.to(input_dtype), final_state
