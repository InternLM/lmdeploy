# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.nn import build_rotary_embedding_from_config
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .llama import LlamaDecoderLayer
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


class EagleLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__(config, layer_idx, dtype=dtype, device=device, is_tp=False)

        # Skip the input_layernorm
        # https://github.com/SafeAILab/EAGLE/blob/35c78f6cdc19a73e05cf5c330b4c358dad970c6a/eagle/model/cnets.py#L427
        if layer_idx == 0:
            del self.input_layernorm
            setattr(self, 'input_layernorm', lambda x: x)


class EagleLlamaModel(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            EagleLlamaDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        # build fc
        self.fc = nn.Linear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

        # build rotary embedding in LlamaModel
        self.rotary_emb = build_rotary_embedding_from_config(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        previous_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of LlamaModel.forward."""
        # token embedding
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_tokens(input_ids)
        previous_hidden_states = previous_hidden_states.to(inputs_embeds)
        hidden_states = torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        hidden_states = self.fc(hidden_states)
        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )
        hidden_states = hidden_states + residual
        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class EagleLlamaForCausalLM(nn.Module, CudaGraphMixin):

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self, config, ctx_mgr, dtype=None, device=None):
        nn.Module.__init__(self)
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype
        # build LLamaModel
        self.model = EagleLlamaModel(config, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        target_hidden_states: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            previous_hidden_states=target_hidden_states,
        )
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        target_hidden_states = context.target_hidden_states
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            target_hidden_states=target_hidden_states,
        )

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(1,
                                                                                     max_tokens,
                                                                                     self.config.hidden_size,
                                                                                     dtype=self.dtype)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)

        num_tokens = kwargs['input_ids'].size(-1)

        is_decoding = graph_meta.is_decoding
        input_buffers = graph_meta.input_buffers
        padded_num_tokens = new_inputs['input_ids'].size(-1)

        target_hidden_states = kwargs.get('target_hidden_states')
        assert target_hidden_states is not None
        input_buffers['target_hidden_states'][:, :num_tokens] = target_hidden_states
        if is_decoding:
            new_inputs['target_hidden_states'] = input_buffers['target_hidden_states'][:, :padded_num_tokens, :]
        else:
            new_inputs['target_hidden_states'] = input_buffers['target_hidden_states']

        return new_inputs

    def update_weights(self):
        """Update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            name = 'model.' + name
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
