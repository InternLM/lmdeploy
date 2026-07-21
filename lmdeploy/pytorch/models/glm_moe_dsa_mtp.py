# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import RMSNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear

from .deepseek_mtp import DeepseekMTPModel, build_deepseek_rotary_embedding
from .deepseek_v32 import (
    DeepseekV32DecoderLayer,
    DSATopKIndicesBuffer,
    _load_fused_indexer_weight,
)


class GlmMoeDsaSharedHead(nn.Module):
    """GLM MTP final normalization."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            dtype=dtype,
                            device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class GlmMoeDsaMultiTokenPredictorLayer(nn.Module):
    """GLM-MoE-DSA MTP layer."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.enorm = RMSNorm(config.hidden_size,
                             config.rms_norm_eps,
                             dtype=dtype,
                             device=device)
        self.hnorm = RMSNorm(config.hidden_size,
                             config.rms_norm_eps,
                             dtype=dtype,
                             device=device)
        self.eh_proj = build_colwise_linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=quantization_config,
            dp_disable_tp=True,
        )
        self.shared_head = GlmMoeDsaSharedHead(config,
                                               dtype=dtype,
                                               device=device)
        self.mtp_block = DeepseekV32DecoderLayer(config,
                                                 layer_idx=layer_idx,
                                                 dtype=dtype,
                                                 device=device)
        self.rotary_emb = build_deepseek_rotary_embedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_value: list[torch.Tensor],
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: Any = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        skip_topk: bool = False,
    ) -> torch.Tensor:
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1))

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        hidden_states, residual = self.mtp_block(
            hidden_states,
            (cos[0], sin[0]),
            past_key_value,
            attn_metadata=attn_metadata,
            topk_indices_buffer=topk_indices_buffer,
            skip_topk=skip_topk,
        )
        return residual + hidden_states


class GlmMoeDsaMultiTokenPredictor(nn.Module):
    """GLM-MoE-DSA multi-token predictor."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.embed_tokens = None
        self.layers = nn.ModuleDict({
            str(idx):
            GlmMoeDsaMultiTokenPredictorLayer(config,
                                              idx,
                                              dtype=dtype,
                                              device=device)
            for idx in range(self.mtp_start_layer_idx,
                             self.mtp_start_layer_idx + self.num_mtp_layers)
        })

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: Any = None,
        topk_indices_buffer: DSATopKIndicesBuffer | None = None,
        skip_topk: bool = False,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer_idx = self.mtp_start_layer_idx + current_step_idx
        return self.layers[str(layer_idx)](
            input_ids,
            position_ids,
            previous_hidden_states,
            past_key_values[current_step_idx],
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
            topk_indices_buffer=topk_indices_buffer,
            skip_topk=skip_topk,
        )

    def prepare_hidden_states_for_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Apply the GLM final norm for logits."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return layer.shared_head(hidden_states)

    def set_input_embeddings(self, embed_tokens: nn.Module):
        self.embed_tokens = embed_tokens

    def get_input_embeddings(self):
        return self.embed_tokens


class GlmMoeDsaMTPModel(DeepseekMTPModel):
    """GLM-MoE-DSA MTP model."""

    def __init__(
        self,
        config: PretrainedConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        nn.Module.__init__(self)
        self.config = config
        self.quantization_config = getattr(config, 'quantization_config', None)
        self.dtype = dtype
        self.ctx_mgr = ctx_mgr
        self.model = GlmMoeDsaMultiTokenPredictor(config,
                                                  dtype=dtype,
                                                  device=device)
        self.topk_indices_buffer = DSATopKIndicesBuffer(config.index_topk)
        self.uses_dsa_topk_buffer = getattr(config,
                                            'index_share_for_mtp_iteration',
                                            False)
        self._load_buffers = dict()

    def set_input_embeddings(self, embed_tokens: nn.Module):
        self.model.set_input_embeddings(embed_tokens)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_topk_indices_buffer(self, topk_indices_buffer: DSATopKIndicesBuffer):
        self.topk_indices_buffer = topk_indices_buffer

    def compact_topk_indices(self, row_indices: torch.Tensor):
        self.topk_indices_buffer.compact(row_indices)

    def prepare_hidden_states_for_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model.prepare_hidden_states_for_logits(hidden_states,
                                                           spec_step_idx=spec_step_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        skip_topk: bool = False,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            position_ids,
            target_hidden_states,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attn_metadata=attn_metadata,
            topk_indices_buffer=self.topk_indices_buffer,
            skip_topk=skip_topk,
            spec_step_idx=spec_step_idx,
        )

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        inputs = super().prepare_inputs_for_generation(
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )
        model_metas = context.model_metas
        inputs['skip_topk'] = (
            self.uses_dsa_topk_buffer and model_metas is not None and len(model_metas) > 0
            and all(meta is not None and meta.get('skip_topk', False) for meta in model_metas))
        return inputs

    def _load_weight_attention(self, name: str, loaded_weight: torch.Tensor,
                               params_dict: dict[str, nn.Parameter],
                               update_pe_mapping: list):
        if _load_fused_indexer_weight(name, loaded_weight, params_dict,
                                      self._load_buffers):
            return
        return super()._load_weight_attention(name, loaded_weight, params_dict,
                                              update_pe_mapping)
