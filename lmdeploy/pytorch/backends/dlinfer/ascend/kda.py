from typing import Any

import torch

from lmdeploy.pytorch.backends.kda import KdaImpl


def _load_ops():
    """Load the public AscendC KDA provider used by vLLM-Ascend."""
    try:
        from fla_npu.ops.ascendc import chunk_kda_fwd, recurrent_kda
    except Exception as exc:
        raise RuntimeError(
            "Ascend KDA operators are unavailable; install "
            "flash-linear-attention-npu."
        ) from exc
    return chunk_kda_fwd, recurrent_kda


class AscendKdaImpl(KdaImpl):
    """GLM-5.3 KDA using the vLLM-Ascend AscendC contracts."""

    def __init__(self):
        # KDA is registered by fla_npu.ops.ascendc, not _C_ascend. Keep the
        # handles for an early, actionable dependency error. The execution
        # wrappers below are shared with vLLM-Ascend so gate/beta/L2 semantics
        # cannot silently diverge again.
        self.chunk_kda_fwd, self.recurrent_kda = _load_ops()
        from .causal_conv1d import AscendCausalConv1dImpl
        self.conv = AscendCausalConv1dImpl()

    @staticmethod
    def _heads(x: torch.Tensor, heads: int, dim: int) -> torch.Tensor:
        return x.unflatten(-1, (heads, dim)).contiguous()

    @staticmethod
    def _select_state(recurrent_state: torch.Tensor, state_ids: torch.Tensor,
                      metadata: Any) -> tuple[torch.Tensor, int | None, torch.Tensor | None]:
        state_ids = state_ids.to(torch.int64).clamp_min(0)
        if recurrent_state.dim() == 5:
            ring = recurrent_state.size(1)
            offsets = getattr(metadata, "spec_state_offsets", None)
            history = getattr(metadata, "cache_seqlens", None)
            if offsets is not None:
                read_slots = offsets[0].to(torch.int64).remainder(ring)
            elif history is not None:
                read_slots = history.to(torch.int64).remainder(ring)
            else:
                read_slots = torch.zeros_like(state_ids)
            initial = recurrent_state[state_ids, read_slots].contiguous()
            return initial, ring, read_slots
        return recurrent_state.index_select(0, state_ids).contiguous(), None, None

    @staticmethod
    def _write_state(recurrent_state: torch.Tensor, state_ids: torch.Tensor,
                     final_state: torch.Tensor, metadata: Any, ring: int | None,
                     read_slots: torch.Tensor | None, *, prefill: bool) -> None:
        state_ids = state_ids.to(torch.int64).clamp_min(0)
        final_state = final_state.to(recurrent_state.dtype)
        if ring is None:
            recurrent_state.index_copy_(0, state_ids, final_state)
            return
        offsets = getattr(metadata, "spec_state_offsets", None)
        history = getattr(metadata, "cache_seqlens", None)
        if offsets is not None:
            write_slots = offsets[1].to(torch.int64)
        elif history is not None:
            write_slots = history.to(torch.int64) + (0 if prefill else 1)
        else:
            write_slots = read_slots if read_slots is not None else torch.zeros_like(state_ids)
        recurrent_state[state_ids, write_slots.remainder(ring)].copy_(final_state)


    def forward(self, mixed_qkv, raw_gate, raw_beta, conv_weight, conv_bias,
                a_log, dt_bias, conv_state, recurrent_state, metadata,
                num_heads, head_dim, lower_bound):
        mixed_qkv = self.conv.forward(
            mixed_qkv, conv_weight, conv_bias, conv_state, metadata, activation="silu")
        q, k, v = mixed_qkv.split(num_heads * head_dim, dim=-1)
        q = self._heads(q, num_heads, head_dim)
        k = self._heads(k, num_heads, head_dim)
        v = self._heads(v, num_heads, head_dim)
        gate = raw_gate.unflatten(-1, (num_heads, head_dim)).contiguous()
        beta = raw_beta.unflatten(-1, (num_heads,)).contiguous()
        cu = metadata.cu_seqlens
        # D6: AscendGatedDeltaMeta clamps at construction and
        # derives both int32/int64 once per step.
        state_ids = metadata.state_ids_i64
        if metadata.is_decoding and recurrent_state.dim() == 4:
            # D5a: non-MTP decode updates the slot-leading pool in place.
            # The aclnn descriptor carries the per-layer view's dim0 stride
            # and the kernel reads/writes exactly the indexed slots --
            # bitwise equal to the compact-bank round trip below (see
            # bench/parity/test_d5a_kda_confirm.py) while dropping its
            # per-layer gather + bank build + index_copy_ writeback
            # (34 x ~340us ScatterUpdate per decode step).
            output, _ = self.recurrent_kda(
                q.contiguous(), k.contiguous(), v.contiguous(), gate.contiguous(),
                beta.contiguous(), recurrent_state, cu_seqlens=cu,
                ssm_state_indices=metadata.conv_state_indices,
                A_log=a_log.reshape(-1).contiguous(),
                dt_bias=dt_bias.contiguous(),
                layout="BSND", scale=head_dim ** -0.5,
                output_final_state=False, inplace_final_state=True,
                state_v_first=True, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                safe_gate=lower_bound is not None,
                lower_bound=lower_bound if lower_bound is not None else -5.0)
            return output
        initial, ring, read_slots = self._select_state(recurrent_state, state_ids, metadata)

        if metadata.is_decoding:
            # Match vLLM-Ascend run_recurrent_kda: the operator owns Q/K
            # normalization, safe-gate construction and beta sigmoid. A compact
            # positive-index state bank keeps continuous batching semantics.
            padded_state = torch.cat([torch.zeros_like(initial[:1]), initial], dim=0)
            local_state_ids = torch.arange(
                1, state_ids.numel() + 1, device=q.device, dtype=torch.int32)
            output, _ = self.recurrent_kda(
                q.contiguous(), k.contiguous(), v.contiguous(), gate.contiguous(),
                beta.contiguous(), padded_state, cu_seqlens=cu,
                ssm_state_indices=local_state_ids,
                A_log=a_log.reshape(-1).contiguous(),
                dt_bias=dt_bias.contiguous(),
                layout="BSND", scale=head_dim ** -0.5,
                output_final_state=False, inplace_final_state=True,
                state_v_first=True, use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
                safe_gate=lower_bound is not None,
                lower_bound=lower_bound if lower_bound is not None else -5.0)
            final_state = padded_state[1:]
            self._write_state(recurrent_state, state_ids, final_state, metadata,
                              ring, read_slots, prefill=False)
            return output

        if getattr(metadata, "has_initial_state", None) is not None:
            initial = initial.clone()
            initial[~metadata.has_initial_state.to(torch.bool)] = 0

        # vLLM-Ascend passes host cu-seqlens to the AscendC chunk wrapper.
        # chunk_indices is optional in the public fla_npu API; omitting it
        # selects the same internal 64-token chunk construction.
        cu_host = cu.to(device="cpu", dtype=torch.int32).tolist()
        output, final_state, *_ = self.chunk_kda_fwd(
            q.contiguous(), k.contiguous(), v.contiguous(), gate.contiguous(),
            beta.float().sigmoid().contiguous(), head_dim ** -0.5,
            chunk_size=64, layout="BSND", initial_state=initial.float().contiguous(),
            output_final_state=True, cu_seqlens=cu_host, chunk_indices=None,
            safe_gate=lower_bound is not None,
            lower_bound=lower_bound if lower_bound is not None else -5.0,
            use_gate_in_kernel=True, A_log=a_log.reshape(-1).contiguous(),
            dt_bias=dt_bias.contiguous(), state_v_first=True,
            use_qk_l2norm_in_kernel=True)
        self._write_state(recurrent_state, state_ids, final_state, metadata,
                          ring, read_slots, prefill=True)
        return output
