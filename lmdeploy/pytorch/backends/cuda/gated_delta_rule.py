# Copyright (c) OpenMMLab. All rights reserved.
from functools import lru_cache

import torch

from ..gated_delta_rule import GatedDeltaRuleBuilder, GatedDeltaRuleImpl
from .utils import has_tilelang


@lru_cache
def has_fla():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        return True
    except Exception:
        return False


class CudaGatedDeltaRuleImpl(GatedDeltaRuleImpl):

    def __init__(self):
        if not has_fla() or not has_tilelang():
            raise ImportError('fla and tilelang is required for CudaGatedDeltaRuleImpl')
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
        self.chunk_func = chunk_gated_delta_rule
        self.recurrent_func = fused_recurrent_gated_delta_rule

    def chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        output_final_state: bool = False,
        spec_state_offsets: torch.Tensor | None = None,
    ):

        assert initial_state is not None
        recurrent_state = initial_state
        batch_state = recurrent_state.index_select(0, state_indices)

        if spec_state_offsets is not None:
            batch_idx = torch.arange(batch_state.size(0), device=batch_state.device)
            init_state = batch_state[batch_idx, spec_state_offsets]
        else:
            init_state = batch_state

        if use_qk_l2norm_in_kernel:
            # l2norm in fla would recompile when seqlen changed.
            q = torch.nn.functional.normalize(q, p=2, dim=-1)
            k = torch.nn.functional.normalize(k, p=2, dim=-1)
        core_attn_out, last_state = self.chunk_func(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=init_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
        )
        if spec_state_offsets is not None:
            batch_state[batch_idx, spec_state_offsets] = last_state.to(recurrent_state.dtype)
            recurrent_state.index_copy_(0, state_indices, batch_state)
        else:
            last_state = recurrent_state.index_copy_(0, state_indices, last_state.to(recurrent_state.dtype))
        if not output_final_state:
            last_state = None
        return core_attn_out, last_state

    def fused_recurrent_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor | None = None,
        beta: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        state_indices: torch.Tensor | None = None,
        scale: float | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        output_final_state: bool = False,
        cache_seqlens: torch.Tensor | None = None,
    ):
        return self.recurrent_func(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            state_indices=state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            output_final_state=output_final_state,
            cache_seqlens=cache_seqlens,
        )


class CudaGatedDeltaRuleBuilder(GatedDeltaRuleBuilder):

    @staticmethod
    def build() -> GatedDeltaRuleImpl:
        return CudaGatedDeltaRuleImpl()
