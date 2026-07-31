# Copyright (c) OpenMMLab. All rights reserved.

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import build_colwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .hy3 import Hy3DecoderLayer
from .patch import add_prefix
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


class HYV3MultiTokenPredictorLayer(nn.Module):
    """One Hy3 multi-token prediction layer.

    Transformers currently ignores the checkpoint's MTP layer.  The forward
    contract here follows the HY-team-authored vLLM ``hy_v3_mtp`` reference:
    normalize the shifted token embedding and previous target hidden state,
    concatenate embedding first, project 2H -> H, run decoder layer 80, then
    apply the MTP-specific final RMSNorm to the residual sum.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.layer_idx = layer_idx

        self.enorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('enorm', prefix),
        )
        self.hnorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('hnorm', prefix),
        )

        # The official BF16 and FP8 checkpoints both keep eh_proj in BF16.
        self.eh_proj = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            prefix=add_prefix('eh_proj', prefix),
        )

        # Keep the checkpoint layer index. For Hy3 this is layer 80, which
        # must be an MoE layer rather than the dense layer at index zero.
        self.mtp_block = Hy3DecoderLayer(
            config,
            layer_idx=layer_idx,
            dtype=dtype,
            device=device,
            prefix=prefix,
        )
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            dtype=dtype,
            device=device,
            prefix=add_prefix('final_layernorm', prefix),
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        rotary_pos_emb: tuple[torch.Tensor, torch.Tensor],
        past_key_value: list[torch.Tensor],
        attn_metadata: Any = None,
    ) -> torch.Tensor:
        """Fuse token embeddings with target hidden states and run MTP."""
        input_embeddings = self.enorm(input_embeddings)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(torch.cat([input_embeddings, previous_hidden_states], dim=-1))
        hidden_states, residual = self.mtp_block(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class HYV3MultiTokenPredictor(nn.Module):
    """Hy3 MTP stack using the checkpoint's target-adjacent layer indices."""

    def __init__(
        self,
        config: PretrainedConfig,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        if self.num_mtp_layers < 1:
            raise ValueError('Hy3 MTP requires at least one checkpoint MTP layer.')

        # Shared with the target model. Keeping a second copy would waste
        # memory and force the draft loader to read an unrelated checkpoint
        # shard.
        self.embed_tokens = None
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): HYV3MultiTokenPredictorLayer(
                    config,
                    layer_idx=layer_idx,
                    dtype=dtype,
                    device=device,
                    prefix=add_prefix(f'layers.{layer_idx}', prefix),
                )
                for layer_idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.rotary_emb = build_rotary_embedding_from_config(config, device=device)

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
        """Run the MTP layer selected for this speculative step."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Match the Hy3 MTP reference path: position zero has no preceding
        # target state to condition on and its token embedding is ignored.
        inputs_embeds = inputs_embeds.masked_fill(position_ids[..., None] == 0, 0)

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        cos, sin = self.rotary_emb(previous_hidden_states, position_ids)
        rotary_pos_emb = (cos[0], sin[0])

        return self.layers[str(layer_idx)](
            inputs_embeds,
            previous_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_values[current_step_idx],
            attn_metadata=attn_metadata,
        )

    def get_input_embeddings(self):
        """Return the embedding table loaded from the target checkpoint."""
        return self.embed_tokens

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Share the target model's token embeddings."""
        self.embed_tokens = embed_tokens


class HYV3MTP(nn.Module, CudaGraphMixin):
    """Hy3 draft model used by multi-token speculative decoding."""

    packed_modules_mapping = {
        'qkv_proj': ['q_proj', 'k_proj', 'v_proj'],
        'gate_up_proj': ['gate_proj', 'up_proj'],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        prefix: str = '',
    ):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype
        self.model = HYV3MultiTokenPredictor(
            config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('model', prefix),
        )

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
        """Return draft hidden states; sampling is handled by the proposer."""
        return self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            previous_hidden_states=target_hidden_states,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
            spec_step_idx=spec_step_idx,
        )

    def get_input_embeddings(self):
        """Return token embeddings."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embed_tokens: nn.Module):
        """Share token embeddings with the target model."""
        self.model.set_input_embeddings(embed_tokens)

    def get_checkpoint_weight_prefixes(self) -> tuple[str, ...]:
        """Return the checkpoint prefixes needed by this draft model.

        Hy3 MTP is stored alongside the 80-layer target model. Restricting the
        loader here avoids reading every target-model safetensors shard again.
        """
        start = self.config.num_hidden_layers
        end = start + self.config.num_nextn_predict_layers
        return tuple(f'model.layers.{layer_idx}.' for layer_idx in range(start, end))

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare shifted target outputs for an MTP draft forward."""
        if context.target_inputs_embeds is not None:
            inputs_embeds = context.target_inputs_embeds
        return {
            'input_ids': context.input_ids,
            'position_ids': context.position_ids,
            'past_key_values': past_key_values,
            'attn_metadata': context.attn_metadata,
            'inputs_embeds': inputs_embeds,
            'target_hidden_states': context.target_hidden_states,
        }

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Allocate the target hidden-state CUDA Graph buffer."""
        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(
            1,
            graph_meta.max_tokens,
            self.config.hidden_size,
            dtype=self.dtype,
        )
        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, input_ids: torch.Tensor, **kwargs):
        """Copy target hidden states into stable CUDA Graph storage."""
        new_inputs = super().fill_buffers_cudagraph(
            graph_meta=graph_meta,
            input_ids=input_ids,
            **kwargs,
        )
        target_hidden_states = kwargs.get('target_hidden_states')
        if target_hidden_states is None:
            raise ValueError('target_hidden_states is required for Hy3 MTP')

        num_tokens = input_ids.size(-1)
        target_buffer = graph_meta.input_buffers['target_hidden_states']
        target_buffer[:, :num_tokens] = target_hidden_states
        new_inputs['target_hidden_states'] = target_buffer
        return new_inputs

    def _map_checkpoint_weight(self, name: str) -> tuple[str, dict[str, Any]] | None:
        """Map one Hy3 checkpoint key to the draft model parameter."""
        if name in ('model.embed_tokens.weight', 'lm_head.weight'):
            return None

        for layer_idx in range(
            self.config.num_hidden_layers,
            self.config.num_hidden_layers + self.config.num_nextn_predict_layers,
        ):
            layer_prefix = f'model.layers.{layer_idx}.'
            if not name.startswith(layer_prefix):
                continue

            suffix = name.removeprefix(layer_prefix)
            # The draft model shares the target embedding table and LM head.
            if suffix.startswith(('embed_tokens.', 'shared_head.')):
                return None

            outer_modules = ('enorm.', 'hnorm.', 'eh_proj.', 'final_layernorm.')
            if not suffix.startswith(outer_modules):
                name = f'{layer_prefix}mtp_block.{suffix}'

            if '.experts.' in name:
                expert_suffix = name.split('.experts.', 1)[1]
                expert_id_text, projection_name = expert_suffix.split('.', 1)
                if expert_id_text.isdigit():
                    expert_id = int(expert_id_text)
                    expert_mappings = {
                        'gate_proj': ('gate_up', 'gate'),
                        'up_proj': ('gate_up', 'up'),
                        'down_proj': ('down', 'down'),
                    }
                    projection, separator, _ = projection_name.partition('.')
                    if separator and projection in expert_mappings:
                        target_projection, shard_id = expert_mappings[projection]
                        source = f'.experts.{expert_id}.{projection}.'
                        target = f'.experts.{target_projection}.'
                        return name.replace(source, target), {
                            'expert_id': expert_id,
                            'shard_id': shard_id,
                        }

            stacked_mappings = (
                ('.qkv_proj.', '.q_proj.', 'q'),
                ('.qkv_proj.', '.k_proj.', 'k'),
                ('.qkv_proj.', '.v_proj.', 'v'),
                ('.gate_up_proj.', '.gate_proj.', 0),
                ('.gate_up_proj.', '.up_proj.', 1),
            )
            for target, source, shard_id in stacked_mappings:
                if source in name:
                    return name.replace(source, target), {'shard_id': shard_id}
            return name, {}

        return None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load the shared weights and checkpoint MTP layers."""
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name:
                continue

            mapped_weight = self._map_checkpoint_weight(name)
            if mapped_weight is None:
                continue

            target_name, loader_kwargs = mapped_weight
            if target_name not in params_dict:
                raise KeyError(f'Hy3 MTP checkpoint key {name!r} maps to missing parameter {target_name!r}')
            load_weight(
                params_dict[target_name],
                loaded_weight,
                **loader_kwargs,
            )
