# Copyright (c) OpenMMLab. All rights reserved.
"""CUDA KDA backend composed from FLA and shared LMDeploy operators.

KDA is distinct from LMDeploy's gated-delta rule, but its CUDA implementation does not need copied GLM kernels. This
adapter owns only LMDeploy cache/state semantics, reuses FLA convolution/prefill, and shares the TileLang recurrent
state-ring kernel with gated-delta rule for both AR and MTP decode.
"""

from copy import copy
from typing import Any

import torch

from lmdeploy.pytorch.backends.kda import KdaImpl

from .gated_delta_rule import GatedDeltaStepMetaUpdater, _state_scatter, _state_select
from .step_metadata import register_step_metadata_impl


def _select_state(state: torch.Tensor, metadata: Any) -> torch.Tensor:
    valid = metadata.valid_state
    if metadata.is_init is not None:
        valid = valid & ~metadata.is_init
    ids = torch.where(valid, metadata.state_ids, -1)
    # Treat the ordinary AR bank as a one-slot ring; share the same masked
    # gather/scatter kernels with GDN and speculative state checkpoints.
    return _state_select(state.unsqueeze(1), ids, torch.zeros_like(ids))


def _store_state(state: torch.Tensor, value: torch.Tensor,
                 metadata: Any) -> None:
    ids = torch.where(metadata.valid_state, metadata.state_ids, -1)
    _state_scatter(state.unsqueeze(1), ids, torch.zeros_like(ids), value)


class CudaKdaImpl(KdaImpl):
    """FLA prefill and shared channelwise gated-delta decode."""

    def __init__(self):
        try:
            from fla.modules.conv.triton.ops import (
                causal_conv1d_fwd,
                causal_conv1d_update,
            )
            from fla.ops.kda import chunk_kda
            from fla.ops.kda.gate import kda_gate_fwd
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                'GLM-5.3 KDA requires flash-linear-attention==0.5.2.'
            ) from exc
        self.causal_conv1d_fwd = causal_conv1d_fwd
        self.causal_conv1d_update = causal_conv1d_update
        self.chunk_kda = chunk_kda
        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_update as shared_conv_update
        from lmdeploy.pytorch.kernels.cuda.gated_delta_rule import fused_recurrent_gated_delta_rule
        self.shared_conv_update = shared_conv_update
        self.kda_gate = kda_gate_fwd
        self.recurrent_func = fused_recurrent_gated_delta_rule
        self.fused_recurrent_kda = self._decode_recurrent
        register_step_metadata_impl(self)

    def get_step_metadata_provider(self):
        """Reuse FLA chunk-index preparation outside model forward."""
        return GatedDeltaStepMetaUpdater()

    def _forward_spec(self, mixed_qkv, raw_gate, raw_beta, conv_state,
                      recurrent_state, metadata, **kwargs):
        """Checkpoint every verified token at its accepted-history ring slot.

        State is addressed by accepted history length, not the last proposed length. The ring therefore also handles
        zero/partial acceptance and request reordering without a scheduler-side rollback hook.
        """
        if metadata.is_decoding:
            return self._forward_spec_decode(mixed_qkv, raw_gate, raw_beta,
                                             conv_state, recurrent_state, metadata, **kwargs)
        history = metadata.cache_seqlens.long()
        ring_size = metadata.num_spec_tokens + 1
        ids = torch.where(metadata.valid_state, metadata.state_ids, -1).long()
        read_slot = history.remainder(ring_size)
        # FLA prefill consumes a chronological window; the persistent cache
        # uses the same compact token ring as Qwen3.5's causal convolution.
        conv_cache = _select_state(conv_state, metadata)
        width = kwargs['conv_weight'].shape[-1]
        offsets = torch.arange(-width, 0, device=ids.device)
        read_offsets = (history[:, None] + offsets).remainder(conv_state.size(-1))
        conv = conv_cache.gather(2, read_offsets[:, None].expand(-1, conv_cache.size(1), -1))
        recurrent = _state_select(recurrent_state, ids, read_slot)
        local = copy(metadata)
        local.num_spec_tokens = 0
        local.spec_state_offsets = None
        local.spec_conv_offsets = None
        local.state_ids = torch.arange(ids.numel(), device=ids.device)

        def store(state, values, lengths):
            slots = lengths.remainder(ring_size)
            _state_scatter(state, ids, slots, values)

        output = self.forward(mixed_qkv, raw_gate, raw_beta,
                              conv_state=conv, recurrent_state=recurrent,
                              metadata=local, **kwargs)
        lengths = history + metadata.cu_seqlens.diff()
        write_offsets = (lengths[:, None] + offsets).remainder(conv_state.size(-1))
        conv_cache.scatter_(2, write_offsets[:, None].expand(-1, conv_cache.size(1), -1), conv)
        _store_state(conv_state, conv_cache, metadata)
        store(recurrent_state, recurrent, lengths)
        return output

    def _decode_recurrent(self, q, k, v, g, beta, A_log, dt_bias, initial_state,
                          output_final_state=True, lower_bound=None, **kwargs):
        """Keep AR and MTP on the same recurrence and gate arithmetic."""
        gate = self.kda_gate(g, A_log, dt_bias, lower_bound=lower_bound)
        return self.recurrent_func(q, k, v, g=gate, beta=beta.float().sigmoid(),
                                   initial_state=initial_state, output_final_state=output_final_state,
                                   use_qk_l2norm_in_kernel=True, transpose_state_layout=True)

    def _forward_spec_decode(self, mixed_qkv, raw_gate, raw_beta, conv_state,
                             recurrent_state, metadata, **kwargs):
        """Reuse causal-convolution token rings and one verification
        recurrence.

        The recurrence is parallel across state tiles, not across causally dependent timesteps. Each timestep is saved
        for partial acceptance.
        """
        ids = metadata.state_ids.long()
        batch = ids.numel()
        steps = mixed_qkv.size(1) // batch
        ring = metadata.num_spec_tokens + 1
        if steps > ring:
            raise ValueError('KDA verification exceeds the configured state ring.')
        history = metadata.cache_seqlens
        signed_ids = torch.where(metadata.valid_state, ids, -1)
        values = mixed_qkv.reshape(batch, steps, -1).transpose(1, 2).contiguous()
        weight = kwargs['conv_weight']
        if weight.ndim == 3:
            if weight.size(1) != 1:
                raise ValueError('KDA depthwise convolution weight must have shape [D, 1, K].')
            weight = weight.squeeze(1)
        mixed = self.shared_conv_update(values, conv_state, weight,
                                        bias=kwargs['conv_bias'], activation='silu',
                                        conv_state_indices=signed_ids.to(torch.int32),
                                        cache_seqlens=history)
        mixed = mixed.transpose(1, 2)
        heads, dim = kwargs['num_heads'], kwargs['head_dim']
        q, k, v = [x.reshape(batch, steps, heads, dim).contiguous()
                   for x in mixed.split(heads * dim, dim=-1)]
        gate = self.kda_gate(raw_gate.reshape(batch, steps, heads, dim).contiguous(),
                             kwargs['a_log'], kwargs['dt_bias'], lower_bound=kwargs['lower_bound'])
        beta = raw_beta.reshape(batch, steps, heads).float().sigmoid()
        output, _ = self.recurrent_func(q, k, v, g=gate, beta=beta,
                                        initial_state=recurrent_state, state_indices=signed_ids,
                                        cache_seqlens=history, output_final_state=True,
                                        use_qk_l2norm_in_kernel=True, transpose_state_layout=True)
        return output.reshape(1, batch * steps, heads, dim)

    def _conv(
        self,
        mixed_qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        metadata: Any,
    ) -> torch.Tensor:
        selected_state = _select_state(conv_state, metadata)
        if conv_weight.dim() == 3:
            if conv_weight.size(1) != 1:
                raise ValueError(
                    'KDA depthwise convolution weight must have shape '
                    '[D, 1, K].')
            conv_weight = conv_weight.squeeze(1)
        if metadata.is_decoding:
            mixed_qkv, final_state = self.causal_conv1d_update(
                x=mixed_qkv,
                cache=selected_state,
                weight=conv_weight,
                bias=conv_bias,
                activation='silu',
            )
        else:
            mixed_qkv, final_state = self.causal_conv1d_fwd(
                x=mixed_qkv,
                weight=conv_weight,
                bias=conv_bias,
                residual=None,
                initial_state=selected_state,
                output_final_state=True,
                activation='silu',
                cu_seqlens=metadata.cu_seqlens,
            )
        _store_state(conv_state, final_state, metadata)
        return mixed_qkv

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        raw_gate: torch.Tensor,
        raw_beta: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_state: torch.Tensor,
        recurrent_state: torch.Tensor,
        metadata: Any,
        num_heads: int,
        head_dim: int,
        lower_bound: float,
    ) -> torch.Tensor:
        if (getattr(metadata, 'num_spec_tokens', 0) or 0) > 0:
            return self._forward_spec(
                mixed_qkv, raw_gate, raw_beta, conv_state, recurrent_state,
                metadata, conv_weight=conv_weight, conv_bias=conv_bias,
                a_log=a_log, dt_bias=dt_bias, num_heads=num_heads,
                head_dim=head_dim, lower_bound=lower_bound)
        batch_size = metadata.state_ids.numel()
        if metadata.is_decoding:
            query_length = mixed_qkv.size(1) // batch_size
            if query_length != 1:
                raise NotImplementedError(
                    'GLM-5.3 KDA supports single-token autoregressive decode only.')

        mixed_qkv = self._conv(mixed_qkv, conv_weight, conv_bias,
                               conv_state, metadata)
        q, k, v = mixed_qkv.split(num_heads * head_dim, dim=-1)
        q = q.unflatten(-1, (num_heads, head_dim)).contiguous()
        k = k.unflatten(-1, (num_heads, head_dim)).contiguous()
        v = v.unflatten(-1, (num_heads, head_dim)).contiguous()
        raw_gate = raw_gate.unflatten(
            -1, (num_heads, head_dim)).contiguous()
        raw_beta = raw_beta.contiguous()
        selected_state = _select_state(recurrent_state, metadata)

        if metadata.is_decoding:
            def decode_view(x: torch.Tensor) -> torch.Tensor:
                return x.squeeze(0).unflatten(
                    0, (batch_size, 1)).contiguous()

            output, final_state = self.fused_recurrent_kda(
                q=decode_view(q),
                k=decode_view(k),
                v=decode_view(v),
                g=decode_view(raw_gate),
                beta=decode_view(raw_beta),
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=selected_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                lower_bound=lower_bound,
                state_v_first=True,
            )
            output = output.flatten(0, 1).unsqueeze(0)
        else:
            output, final_state = self.chunk_kda(
                q=q,
                k=k,
                v=v,
                g=raw_gate,
                beta=raw_beta,
                A_log=a_log,
                dt_bias=dt_bias,
                initial_state=selected_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=lower_bound,
                state_v_first=True,
                cu_seqlens=metadata.cu_seqlens,
            )
        _store_state(recurrent_state, final_state, metadata)
        return output
