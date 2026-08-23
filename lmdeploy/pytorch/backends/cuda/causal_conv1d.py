# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Callable
from functools import lru_cache, partial

import torch

from ..causal_conv1d import CausalConv1dBuilder, CausalConv1dImpl
from ..gated_delta_rule import GatedDeltaMeta
from .step_metadata import register_piecewise_graph_impl
from .utils import has_tilelang


class CausalConv1dTilelangImpl(CausalConv1dImpl):
    """CausalConv1d update implementation."""

    def __init__(self):
        from lmdeploy.pytorch.kernels.cuda.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = causal_conv1d_update
        self._piecewise_prefill: Callable[..., torch.Tensor] | None = None
        register_piecewise_graph_impl(self)

    def _prefill(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        activation: str,
    ) -> torch.Tensor:
        state_ids = gated_delta_meta.state_ids

        assert x.dim() == 3
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]

        final_state = x[0, gated_delta_meta.conv_idx].transpose(-2, -1)
        if gated_delta_meta.spec_conv_offsets is not None:
            read_offsets, write_offsets = gated_delta_meta.spec_conv_offsets
            channels = torch.arange(conv_state.size(1), device=conv_state.device)[None, :, None]
            all_inits = conv_state[state_ids[:, None, None], channels, read_offsets[:, None, :]]
            conv_state[state_ids[:, None, None], channels, write_offsets[:, None, :]] = final_state
        else:
            all_inits = conv_state[state_ids, :, 1:]
            conv_state.index_copy_(0, state_ids, final_state)

        all_inits.masked_fill_(gated_delta_meta.is_init[:, None, None], 0.0)
        output = self.conv1d_fn(
            x.transpose(-2, -1),
            weight,
            bias,
            seq_idx=gated_delta_meta.seq_idx,
            initial_states=all_inits,
            return_final_states=False,
            activation=activation,
        )
        return output.transpose(-2, -1)

    def enable_piecewise_cuda_graph(self) -> None:
        """Install the causal-convolution boundary owned by this CUDA op."""
        if self._piecewise_prefill is not None:
            return

        from lmdeploy.pytorch.backends.cuda.graph_runner.piecewise import (
            PaddedTensorOutputAdapter,
            eager_boundary,
            get_piecewise_graph_execution,
        )

        @eager_boundary(
            adapter_factory=partial(PaddedTensorOutputAdapter, token_axis=1),
            reuse_bridge_after_next_step=True,
        )
        def run_eager_prefill(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
            activation: str,
        ) -> torch.Tensor:
            execution = get_piecewise_graph_execution()
            assert execution is not None
            return self._prefill(
                x[:, :execution.raw_tokens],
                weight,
                bias,
                conv_state,
                gated_delta_meta,
                activation,
            )

        def piecewise_prefill(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor | None,
            conv_state: torch.Tensor,
            gated_delta_meta: GatedDeltaMeta,
            activation: str,
        ) -> torch.Tensor:
            if get_piecewise_graph_execution() is None:
                return self._prefill(x, weight, bias, conv_state, gated_delta_meta, activation)
            return run_eager_prefill(x, weight, bias, conv_state, gated_delta_meta, activation)

        self._piecewise_prefill = piecewise_prefill

    def prefill(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        activation: str,
    ) -> torch.Tensor:
        """Run causal convolution eagerly only during piecewise prefill."""
        if self._piecewise_prefill is None:
            return self._prefill(x, weight, bias, conv_state, gated_delta_meta, activation)
        return self._piecewise_prefill(x, weight, bias, conv_state, gated_delta_meta, activation)

    def decode(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        activation: str,
    ) -> torch.Tensor:
        """Run the fixed-shape decode update."""
        meta = gated_delta_meta
        if weight.dim() == 3:
            assert weight.size(1) == 1
            weight = weight[:, 0]
        batch_size = meta.conv_state_indices.size(0)
        q_seqlen = x.size(1) // batch_size
        is_spec_decoding = q_seqlen != 1
        x = x.squeeze(0)
        if is_spec_decoding:
            x = x.unflatten(0, (batch_size, q_seqlen)).transpose(1, 2).contiguous()
        output = self.update_fn(
            x,
            conv_state,
            weight,
            bias,
            activation=activation,
            conv_state_indices=meta.conv_state_indices,
            cache_seqlens=meta.cache_seqlens,
        )
        if is_spec_decoding:
            output = output.transpose(1, 2).flatten(0, 1)
        return output[None]

    def forward(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        gated_delta_meta: GatedDeltaMeta,
        activation: str,
    ) -> torch.Tensor:
        """Dispatch causal convolution by inference phase."""
        if gated_delta_meta.is_decoding:
            return self.decode(x, weight, bias, conv_state, gated_delta_meta, activation)
        return self.prefill(x, weight, bias, conv_state, gated_delta_meta, activation)

    def supports_piecewise_cuda_graph(self) -> bool:
        """Return whether this selected CUDA implementation supports PCG."""
        return True

    def conv1d_fn(self,
                  x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  seq_idx: torch.Tensor | None = None,
                  initial_states: torch.Tensor | None = None,
                  return_final_states: bool = False,
                  activation: str | None = None):
        return self.causal_conv1d_fn(x,
                                     weight,
                                     bias=bias,
                                     seq_idx=seq_idx,
                                     initial_states=initial_states,
                                     return_final_states=return_final_states,
                                     activation=activation)

    def update_fn(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        activation: str | None = None,
        conv_state_indices: torch.Tensor | None = None,
        cache_seqlens: torch.Tensor | None = None,
    ):
        """Update conv state."""
        return self.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias=bias,
            activation=activation,
            conv_state_indices=conv_state_indices,
            cache_seqlens=cache_seqlens,
        )


class CausalConv1dDaoImpl(CausalConv1dTilelangImpl):

    def __init__(self):
        self._piecewise_prefill = None
        try:
            import causal_conv1d
            self.causal_conv1d_fn = causal_conv1d.causal_conv1d_fn
            self.causal_conv1d_update = causal_conv1d.causal_conv1d_update
        except Exception:
            raise RuntimeError(
                'causal_conv1d is not installed, please refer to https://github.com/Dao-AILab/causal-conv1d')
        register_piecewise_graph_impl(self)

    def conv1d_fn(self,
                  x: torch.Tensor,
                  weight: torch.Tensor,
                  bias: torch.Tensor | None = None,
                  seq_idx: torch.Tensor | None = None,
                  initial_states: torch.Tensor | None = None,
                  return_final_states: bool = False,
                  activation: str | None = None):
        # Dao's kernel: seq_idx and initial_states are mutually exclusive.
        return self.causal_conv1d_fn(x,
                                     weight,
                                     bias=bias,
                                     seq_idx=seq_idx,
                                     initial_states=None,
                                     return_final_states=return_final_states,
                                     activation=activation)


@lru_cache
def has_dao():
    try:
        import causal_conv1d  # noqa: F401
        causal_conv1d_fn = causal_conv1d.causal_conv1d_fn  # noqa: F841
        causal_conv1d_update = causal_conv1d.causal_conv1d_update  # noqa: F841
        return True
    except Exception:
        return False


class CausalConv1dCudaBuilder(CausalConv1dBuilder):
    """CausalConv1d update implementation builder."""

    @staticmethod
    def build() -> CausalConv1dImpl:
        """build."""
        if has_tilelang():
            return CausalConv1dTilelangImpl()
        elif has_dao():
            return CausalConv1dDaoImpl()
        else:
            raise RuntimeError('No available implementation for CausalConv1d, '
                               'please install https://tilelang.com/ or https://github.com/Dao-AILab/causal-conv1d')
