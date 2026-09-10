# Copyright (c) OpenMMLab. All rights reserved.
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .mimo_v2_flash import MiMoV2Attention, MiMoV2MLP, _get_norm_eps, _load_native_qkv_shard
from .patch import add_prefix
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin, GraphCaptureState


class MiMoV2FlashMTPLayer(nn.Module):
    """One MiMo prediction-depth layer backed by paged SWA KV."""

    def __init__(
        self,
        config: Any,
        layer_idx: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        norm_eps = _get_norm_eps(config)

        self.enorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('enorm', prefix),
        )
        self.hnorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('hnorm', prefix),
        )
        # Checkpoint eh_proj is BF16 even though the dense MLP is FP8.
        self.eh_proj = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=None,
            dp_disable_tp=True,
            prefix=add_prefix('eh_proj', prefix),
        )

        block_prefix = add_prefix('mtp_block', prefix)
        self.self_attn = MiMoV2Attention(
            config,
            is_swa=True,
            use_paged_cache=True,
            quantize_o_proj=False,
            dtype=dtype,
            device=device,
            prefix=add_prefix('self_attn', block_prefix),
        )
        self.mlp = MiMoV2MLP(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('mlp', block_prefix),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('input_layernorm', block_prefix),
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('post_attention_layernorm', block_prefix),
        )
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('final_layernorm', prefix),
        )

        self.rotary_emb = build_rotary_embedding(
            dim=config.swa_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.swa_rope_theta,
            partial_rotary_factor=config.partial_rotary_factor,
            device=device,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_value: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        """Fuse target state and run one dense SWA decoder layer."""
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                dim=-1,
            )
        )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        hidden_states = self.self_attn(
            hidden_states,
            rotary_pos_emb=(cos[0], sin[0]),
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states + residual


class MiMoV2FlashMultiTokenPredictor(nn.Module):
    """Three checkpoint prediction depths sharing target embed/head."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers != 3:
            raise ValueError(f'MiMo-V2-Flash requires 3 MTP layers, got {self.num_mtp_layers}.')
        self.embed_tokens = None
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): MiMoV2FlashMTPLayer(
                    config,
                    layer_idx,
                    dtype=dtype,
                    device=device,
                    prefix=add_prefix(f'layers.{layer_idx}', prefix),
                )
                for layer_idx in range(self.num_mtp_layers)
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: Any = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Run the prediction layer and cache selected by ``spec_step_idx``."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise RuntimeError('MiMo MTP input embedding has not been bound to the target model.')
            inputs_embeds = self.embed_tokens(input_ids)
        return self.layers[str(current_step_idx)](
            position_ids,
            previous_hidden_states,
            past_key_values[current_step_idx],
            inputs_embeds,
            attn_metadata=attn_metadata,
        )

    def prepare_hidden_states_for_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Apply the final norm belonging to the active prediction depth."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(current_step_idx)].final_layernorm(hidden_states)

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Bind the target model's shared token embedding."""
        self.embed_tokens = embed_tokens

    def get_input_embeddings(self):
        """Return the target-owned shared token embedding."""
        return self.embed_tokens


class MiMoV2FlashMTPModel(nn.Module, CudaGraphMixin):
    """MiMo-V2-Flash draft model for multi-token speculative decoding."""

    packed_modules_mapping = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    def get_cudagraph_capture_state(self,
                                    past_key_values: list[list[torch.Tensor]],
                                    attn_metadata: Any = None,
                                    num_blocks: int = 0,
                                    spec_step_idx: int = 0) -> GraphCaptureState:
        """Return the active depth cache that Graph capture must preserve."""
        del attn_metadata, num_blocks
        depth = spec_step_idx % self.model.num_mtp_layers
        return GraphCaptureState(tensors=tuple(past_key_values[depth]))

    def get_cudagraph_extra_key(
        self,
        spec_step_idx: int = 0,
        input_ids: torch.Tensor | None = None,
        attn_metadata: Any = None,
        **kwargs,
    ) -> tuple[int]:
        """Select a stable Graph for each MiMo prediction depth.

        ``spec_step_idx`` selects both a decoder module and its KV-cache
        entry.  As a Python scalar it cannot be changed by copying tensor
        inputs before graph replay, so it must participate in the graph key.

        Query length already participates in the generic Graph key. Keeping
        this key independent of rank-local KV length prevents one DP rank from
        inserting capture-time TP collectives while another rank replays.
        """
        del input_ids, attn_metadata, kwargs
        return (spec_step_idx % self.model.num_mtp_layers, )

    def get_cudagraph_warmup_specs(self, max_query_len: int) -> tuple[tuple[int, int], ...]:
        """Return every query-length and prediction-depth graph used at
        runtime."""
        return tuple(
            (query_len, depth)
            for query_len in range(1, max_query_len + 1)
            for depth in range(self.model.num_mtp_layers)
        )

    @staticmethod
    def select_weight_paths(model_path: str, default_paths: Iterable[str]) -> tuple[str, ...]:
        """Select the standalone MiMo MTP checkpoint instead of target
        shards."""
        del default_paths
        mtp_path = Path(model_path) / 'model_mtp.safetensors'
        if not mtp_path.is_file():
            raise FileNotFoundError(f'MiMo MTP checkpoint was not found: {mtp_path}')
        return (str(mtp_path),)

    _MTP_TENSOR_SUFFIXES = frozenset(
        {
            'enorm.weight',
            'hnorm.weight',
            'eh_proj.weight',
            'input_layernorm.weight',
            'pre_mlp_layernorm.weight',
            'final_layernorm.weight',
            'self_attn.q_proj.weight',
            'self_attn.q_proj.weight_scale_inv',
            'self_attn.k_proj.weight',
            'self_attn.k_proj.weight_scale_inv',
            'self_attn.v_proj.weight',
            'self_attn.v_proj.weight_scale_inv',
            'self_attn.o_proj.weight',
            'self_attn.attention_sink_bias',
            'mlp.gate_proj.weight',
            'mlp.gate_proj.weight_scale_inv',
            'mlp.up_proj.weight',
            'mlp.up_proj.weight_scale_inv',
            'mlp.down_proj.weight',
            'mlp.down_proj.weight_scale_inv',
        }
    )

    def __init__(
        self,
        config: Any,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if dtype is not torch.bfloat16:
            raise ValueError(f'MiMo-V2-Flash MTP only supports torch.bfloat16, but got {dtype}.')
        self.config = config
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = MiMoV2FlashMultiTokenPredictor(
            config,
            dtype=dtype,
            device=device,
            prefix='model',
        )

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Bind the target model's shared token embedding."""
        self.model.set_input_embeddings(embed_tokens)

    def get_input_embeddings(self):
        """Return the target-owned shared token embedding."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare the active MTP depth's inputs from the step context."""
        if context is None:
            raise ValueError('MiMo MTP requires a StepContext.')
        if context.target_inputs_embeds is not None:
            inputs_embeds = context.target_inputs_embeds
        return {
            'input_ids': context.input_ids,
            'position_ids': context.position_ids,
            'past_key_values': past_key_values,
            'attn_metadata': context.attn_metadata,
            'inputs_embeds': inputs_embeds,
            'target_hidden_states': context.target_hidden_states,
            'spec_step_idx': context.spec_step_idx,
        }

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Allocate stable storage for target hidden states during capture."""
        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(
            1,
            graph_meta.max_tokens,
            self.config.hidden_size,
            dtype=self.dtype,
        )
        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, input_ids: torch.Tensor, **kwargs):
        """Copy target hidden states into their CUDA Graph input buffer."""
        new_inputs = super().fill_buffers_cudagraph(
            graph_meta=graph_meta,
            input_ids=input_ids,
            **kwargs,
        )
        target_hidden_states = kwargs.get('target_hidden_states')
        if target_hidden_states is None:
            raise ValueError('target_hidden_states is required for MiMo MTP.')
        num_tokens = input_ids.size(-1)
        target_buffer = graph_meta.input_buffers['target_hidden_states']
        target_buffer[:, :num_tokens] = target_hidden_states
        new_inputs['target_hidden_states'] = target_buffer
        return new_inputs

    def prepare_hidden_states_for_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Apply the active prediction depth's final norm before lm_head."""
        return self.model.prepare_hidden_states_for_logits(hidden_states, spec_step_idx=spec_step_idx)

    @staticmethod
    def prepare_hidden_states_for_next_step(
        hidden_states: torch.Tensor,
        logits_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Carry MiMo's decoder output before final norm to the next depth."""
        del logits_hidden_states
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Run one same-position MiMo MTP prediction depth."""
        return self.model(
            input_ids,
            position_ids,
            target_hidden_states,
            past_key_values,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
            spec_step_idx=spec_step_idx,
        )

    @classmethod
    def _parse_mtp_name(cls, name: str) -> tuple[int, str] | None:
        match = re.fullmatch(r'model\.mtp\.layers\.(\d+)\.(.+)', name)
        if match is None:
            return None
        layer_idx = int(match.group(1))
        suffix = match.group(2)
        if layer_idx not in range(3):
            raise KeyError(f'Unexpected MiMo MTP layer index in {name!r}.')
        if suffix not in cls._MTP_TENSOR_SUFFIXES:
            raise KeyError(f'Unknown MiMo MTP tensor {name!r}.')
        return layer_idx, suffix

    @staticmethod
    def _target_name(layer_idx: int, suffix: str) -> str:
        prefix = f'model.layers.{layer_idx}'
        if suffix in {'enorm.weight', 'hnorm.weight', 'eh_proj.weight', 'final_layernorm.weight'}:
            return f'{prefix}.{suffix}'
        if suffix == 'pre_mlp_layernorm.weight':
            suffix = 'post_attention_layernorm.weight'
        return f'{prefix}.{suffix}'

    def _load_qkv_weight(
        self,
        target_name: str,
        source_name: str,
        loaded_weight: torch.Tensor,
        params_dict: dict[str, nn.Parameter],
        shard_id: str,
    ):
        tensor_kind = 'scale' if source_name.endswith('.weight_scale_inv') else 'weight'
        target_prefix = re.sub(r'\.(q|k|v)_proj$', '.qkv_proj', target_name.rsplit('.', 1)[0])
        _load_native_qkv_shard(target_prefix, tensor_kind, loaded_weight, params_dict, shard_id)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load exactly the three MTP depths and reject incomplete payloads."""
        params_dict = dict(self.named_parameters())
        seen = set()
        for name, loaded_weight in weights:
            parsed = self._parse_mtp_name(name)
            if parsed is None:
                continue
            if name in seen:
                raise KeyError(f'Duplicate MiMo MTP tensor {name!r}.')
            seen.add(name)
            layer_idx, suffix = parsed
            target_name = self._target_name(layer_idx, suffix)

            qkv_match = re.search(r'self_attn\.(q|k|v)_proj\.', suffix)
            if qkv_match is not None:
                self._load_qkv_weight(
                    target_name,
                    name,
                    loaded_weight,
                    params_dict,
                    qkv_match.group(1),
                )
                continue

            for source_projection, target_projection, shard_id in (
                ('gate_proj', 'gate_up_proj', 0),
                ('up_proj', 'gate_up_proj', 1),
            ):
                if f'mlp.{source_projection}.' in suffix:
                    target_name = target_name.replace(source_projection, target_projection)
                    load_weight(params_dict[target_name], loaded_weight, shard_id=shard_id)
                    break
            else:
                load_weight(params_dict[target_name], loaded_weight)

        expected = {
            f'model.mtp.layers.{layer_idx}.{suffix}' for layer_idx in range(3) for suffix in self._MTP_TENSOR_SUFFIXES
        }
        missing = sorted(expected - seen)
        if missing:
            raise KeyError(f'Missing {len(missing)} MiMo MTP tensors; first missing tensor: {missing[0]!r}.')
