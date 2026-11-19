# Copyright (c) OpenMMLab. All rights reserved.
"""Opt-in GLM KDA gate prefixes with the training FP32 scan rounding.

The original FLA 0.5.2 fused kernel runs with a fixed scan layout and FP
contraction disabled. This preserves the materialized-gate/fixed-scan result
without allocating a full FP32 token-gate tensor. All KDA state, inverse, WY,
and output kernels remain the installed FLA implementations.

The flag is read at import; callers keep the original native functions when
it is disabled. Explicit fixed_* calls select this implementation for the
validated BF16/K128/safe-gate shape; other calls delegate to native FLA.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from functools import lru_cache, update_wrapper
from types import FunctionType
from typing import Any

import torch

FIXED_PREFIX = os.getenv('LMDEPLOY_GLM53_KDA_FIXED_PREFIX', '0').lower() in {'1', 'true', 'yes', 'on'}


def fixed_kda_gate_chunk_cumsum(g: torch.Tensor,
                              A_log: torch.Tensor,
                              chunk_size: int,
                              scale: float | None = None,
                              dt_bias: torch.Tensor | None = None,
                              cu_seqlens: torch.Tensor | None = None,
                              output_dtype: torch.dtype | None = torch.float32,
                              chunk_indices: torch.Tensor | None = None,
                              lower_bound: float | None = None,
                              **kwargs) -> torch.Tensor:
    """Return a base-2 gate prefix using the original FLA fused kernel.

    The interface matches FLA's gate-prefix helper, including native fallback
    for other dtypes, gate modes, feature widths, or output/scaling choices.
    Packed callers may supply their existing chunk metadata without a readback.
    """
    from fla.ops.kda.gate import kda_gate_chunk_cumsum as native
    from fla.ops.utils.constant import RCP_LN2

    if kwargs or not _supported_gate(g, A_log, dt_bias, chunk_size, scale, output_dtype, lower_bound):
        return native(g=g, A_log=A_log, chunk_size=chunk_size, scale=scale, dt_bias=dt_bias,
                      cu_seqlens=cu_seqlens, output_dtype=output_dtype, chunk_indices=chunk_indices,
                      lower_bound=lower_bound, **kwargs)
    if cu_seqlens is not None and g.shape[0] != 1:
        # Retain the native validation/error for unsupported packed batch shapes.
        return native(g=g, A_log=A_log, chunk_size=chunk_size, scale=scale, dt_bias=dt_bias,
                      cu_seqlens=cu_seqlens, output_dtype=output_dtype, chunk_indices=chunk_indices,
                      lower_bound=lower_bound)
    import triton

    with torch.cuda.device(g.device):
        g, A_log, dt_bias = g.contiguous(), A_log.contiguous(), dt_bias.contiguous()
        output = torch.empty_like(g, dtype=torch.float32, memory_format=torch.contiguous_format)
        if not g.numel():
            return output
        batch, length, heads, width = g.shape
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.contiguous()
            if chunk_indices is None:
                from fla.ops.utils.index import prepare_chunk_indices
                chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            chunk_indices = chunk_indices.contiguous()
        chunks = triton.cdiv(length, chunk_size) if cu_seqlens is None else chunk_indices.shape[0]
        _prefix_kernel()[(triton.cdiv(width, 32), chunks, batch * heads)](
            s=g, A_log=A_log, dt_bias=dt_bias, o=output, scale=RCP_LN2,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound,
            T=length, H=heads, S=width, BT=64, BS=32, REVERSE=False,
            HAS_BIAS=True, HAS_SCALE=True, IS_VARLEN=cu_seqlens is not None, USE_LOWER_BOUND=True,
            num_warps=2, num_stages=3, num_ctas=1, enable_fp_fusion=False)
        return output


def fixed_chunk_kda(q: torch.Tensor,
                    k: torch.Tensor,
                    v: torch.Tensor,
                    g: torch.Tensor,
                    beta: torch.Tensor,
                    scale: float | None = None,
                    initial_state: torch.Tensor | None = None,
                    output_final_state: bool = False,
                    use_qk_l2norm_in_kernel: bool = False,
                    use_gate_in_kernel: bool = False,
                    use_beta_sigmoid_in_kernel: bool = False,
                    allow_neg_eigval: bool = False,
                    safe_gate: bool = False,
                    lower_bound: float | None = None,
                    disable_recompute: bool = False,
                    return_intermediate_states: bool = False,
                    state_v_first: bool = False,
                    cu_seqlens: torch.Tensor | None = None,
                    cu_seqlens_cpu: torch.Tensor | None = None,
                    cp_context: Any = None,
                    **kwargs):
    """Legacy GLM prefill adapter with a private FLA gate-prefix binding.

    Only the standard inference call used by CudaKdaImpl is adapted. Native
    FLA owns unsupported options, validation, backend dispatch, and autograd.
    This adapter preserves native q/k normalization, beta activation, chunk
    metadata and state layout, and never scans an already cumulative prefix.
    """
    from fla.ops.kda import chunk_kda as native
    from fla.ops.utils.constant import RCP_LN2

    call = dict(scale=scale, initial_state=initial_state, output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, use_gate_in_kernel=use_gate_in_kernel,
                use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel, allow_neg_eigval=allow_neg_eigval,
                safe_gate=safe_gate, lower_bound=lower_bound, disable_recompute=disable_recompute,
                return_intermediate_states=return_intermediate_states, state_v_first=state_v_first,
                cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, cp_context=cp_context, **kwargs)
    a_log, dt_bias = kwargs.get('A_log'), kwargs.get('dt_bias')
    chunk_size = kwargs.get('chunk_size', 64)
    supported = (
        set(kwargs) <= {'A_log', 'dt_bias', 'chunk_size'}
        and use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
        and safe_gate and state_v_first and not allow_neg_eigval
        and not disable_recompute and not return_intermediate_states and cp_context is None
        and _supported_gate(g, a_log, dt_bias, chunk_size, RCP_LN2, torch.float32, lower_bound)
        and q.shape == k.shape == v.shape == g.shape and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.device == k.device == v.device == g.device == beta.device
        and beta.shape == q.shape[:3] and beta.dtype == torch.bfloat16
    )
    if supported and cu_seqlens is not None:
        supported = q.shape[0] == 1
    if supported and initial_state is not None:
        count = q.shape[0] if cu_seqlens is None else cu_seqlens.numel() - 1
        supported = (initial_state.dtype == torch.float32 and initial_state.device == q.device
                     and initial_state.shape == (count, q.shape[2], 128, 128))
    tensors = (q, k, v, g, beta, a_log, dt_bias, initial_state)
    if supported and torch.is_grad_enabled() and any(x is not None and x.requires_grad for x in tensors):
        supported = False
    if not supported:
        return native(q, k, v, g, beta, **call)

    from fla.modules.l2norm import l2norm_fwd
    from fla.ops.common.gate import fused_beta_sigmoid
    from fla.ops.utils.index import prepare_chunk_indices

    with torch.cuda.device(q.device):
        q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
        a_log, dt_bias = a_log.contiguous(), dt_bias.contiguous()
        initial_state = None if initial_state is None else initial_state.contiguous()
        cu_seqlens = None if cu_seqlens is None else cu_seqlens.contiguous()
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)
        beta = fused_beta_sigmoid(beta, scale=1.0)
        indices = None if cu_seqlens is None else prepare_chunk_indices(
            cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu)
        result = _chunk_fwd_binding()(
            q=q, k=k, v=v, g=g, beta=beta, scale=q.shape[-1]**-0.5 if scale is None else scale,
            initial_state=initial_state, output_final_state=output_final_state, state_v_first=state_v_first,
            cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=indices,
            chunk_size=chunk_size, safe_gate=safe_gate, lower_bound=lower_bound,
            use_gate_in_kernel=True, A_log=a_log, dt_bias=dt_bias,
            disable_recompute=False, return_intermediate_states=False, cp_context=None)
        return result[0].type_as(q), result[1]


def _supported_gate(g, a_log, dt_bias, chunk_size, scale, output_dtype, lower_bound) -> bool:
    from fla.ops.utils.constant import RCP_LN2

    return (g.ndim == 4 and g.shape[-1] == 128 and g.dtype == torch.bfloat16 and g.device.type == 'cuda'
            and chunk_size == 64 and isinstance(scale, int | float) and scale == RCP_LN2
            and output_dtype == torch.float32 and isinstance(lower_bound, int | float) and lower_bound == -5.0
            and isinstance(a_log, torch.Tensor) and a_log.dtype == torch.float32
            and a_log.device == g.device and a_log.shape == (g.shape[2],)
            and isinstance(dt_bias, torch.Tensor) and dt_bias.dtype == torch.float32
            and dt_bias.device == g.device and dt_bias.shape == (g.shape[2] * g.shape[3],))


@lru_cache(maxsize=1)
def _prefix_kernel():
    from fla.ops.kda.gate import kda_gate_chunk_cumsum_vector_kernel
    from triton.runtime.jit import JITFunction

    kernel = kda_gate_chunk_cumsum_vector_kernel
    while not isinstance(kernel, JITFunction) and hasattr(kernel, 'fn'):
        kernel = kernel.fn
    expected = {'s', 'A_log', 'dt_bias', 'o', 'scale', 'cu_seqlens', 'chunk_indices', 'lower_bound',
                'T', 'H', 'S', 'BT', 'BS', 'REVERSE', 'HAS_BIAS', 'HAS_SCALE', 'IS_VARLEN', 'USE_LOWER_BOUND'}
    if not isinstance(kernel, JITFunction) or set(kernel.arg_names) != expected:
        raise RuntimeError('Fixed KDA prefixes require the FLA 0.5.2 native gate-prefix kernel interface.')
    return kernel


@lru_cache(maxsize=1)
def _chunk_fwd_binding() -> Callable:
    from fla.ops.kda.chunk_fwd import chunk_kda_fwd

    original = inspect.unwrap(chunk_kda_fwd)
    if not isinstance(original, FunctionType) or 'kda_gate_chunk_cumsum' not in original.__code__.co_names:
        raise RuntimeError('Fixed KDA prefixes require the FLA 0.5.2 chunk forward interface.')
    namespace = dict(original.__globals__)
    namespace['kda_gate_chunk_cumsum'] = fixed_kda_gate_chunk_cumsum
    bound = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = None if original.__kwdefaults__ is None else dict(original.__kwdefaults__)
    return update_wrapper(bound, original)


__all__ = ['FIXED_PREFIX', 'fixed_kda_gate_chunk_cumsum', 'fixed_chunk_kda']
