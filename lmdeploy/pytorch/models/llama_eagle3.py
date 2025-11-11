# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext
from lmdeploy.pytorch.nn import RMSNorm, build_rotary_embedding_from_config
from lmdeploy.pytorch.nn.linear import build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .llama import LlamaDecoderLayer
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin


class Eagle3LlamaDecoderLayer(LlamaDecoderLayer):
    """Llama decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config, layer_idx, dtype=dtype, device=device, is_tp=False)
        self.layer_idx = layer_idx

        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

        # override attention qkv
        self.self_attn.qkv_proj = build_qkv_proj(
            2 * hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=False,
        )

        self.hidden_norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        attn_metadata: Any = None,
    ):

        residual = hidden_states
        embeds = self.input_layernorm(embeds)
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([embeds, hidden_states], dim=-1)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Eagle3LlamaModel(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build layer
        self.midlayer = Eagle3LlamaDecoderLayer(config, layer_idx=0, dtype=dtype, device=device)
        target_hidden_size = getattr(config, 'target_hidden_size', config.hidden_size)
        self.fc = build_rowwise_linear(
            target_hidden_size * 3,
            config.hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)
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
            inputs_embeds = self.embed_tokens(input_ids).to(self.dtype)
        previous_hidden_states = previous_hidden_states.to(inputs_embeds)
        if previous_hidden_states.shape[-1] != inputs_embeds.shape[-1]:
            # previous_hidden_states if from target model
            previous_hidden_states = self.fc(previous_hidden_states)
        # rotary embedding
        cos, sin = self.rotary_emb(previous_hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        past_key_value = past_key_values[0]
        hidden_states, residual = self.midlayer(
            inputs_embeds,
            previous_hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )
        hidden_states, hidden_states_prenorm = self.norm(hidden_states, residual)
        outputs = dict(hidden_states=hidden_states, hidden_states_prenorm=hidden_states_prenorm)
        return outputs

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class Eagle3LlamaForCausalLM(nn.Module, CudaGraphMixin):

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

        if config.num_hidden_layers != 1:
            raise ValueError('eagle3 only supports 1 decode layer')

        # build LLamaModel
        self.model = Eagle3LlamaModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.draft_vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long, device=device),
            requires_grad=False,
        )
        self.include_embed_tokens = False

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

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        logits = self.lm_head(hidden_states)
        return logits

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens
        target_hidden_states = kwargs.get('target_hidden_states')
        assert target_hidden_states is not None
        target_hidden_size = target_hidden_states.size(-1)
        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        input_buffers['target_hidden_states'] = input_buffers['input_ids'].new_zeros(1,
                                                                                     max_tokens,
                                                                                     target_hidden_size,
                                                                                     dtype=self.dtype)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)

        num_tokens = kwargs['input_ids'].size(-1)

        input_buffers = graph_meta.input_buffers

        target_hidden_states = kwargs.get('target_hidden_states')
        assert target_hidden_states is not None
        input_buffers['target_hidden_states'][:, :num_tokens] = target_hidden_states

        new_inputs['target_hidden_states'] = input_buffers['target_hidden_states']

        return new_inputs

    def get_outputs_cudagraph(self, output_buffers: Dict[str, torch.Tensor], input_ids: torch.Tensor, **kwargs):
        """Get outputs from buffers."""
        num_tokens = input_ids.size(-1)
        outputs = dict()
        outputs['hidden_states'] = output_buffers['hidden_states'][:, :num_tokens]
        outputs['hidden_states_prenorm'] = output_buffers['hidden_states_prenorm'][:, :num_tokens]
        return outputs

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
            if 'd2t' in name:
                name = 'draft_id_to_target_id'
                base = torch.arange(self.config.draft_vocab_size,
                                    device=loaded_weight.device,
                                    dtype=loaded_weight.dtype)
                loaded_weight += base
            elif 'lm_head.weight' not in name:
                name = 'model.' + name
            if 'embed_tokens' in name:
                self.include_embed_tokens = True
            if 't2d' in name:
                continue
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
