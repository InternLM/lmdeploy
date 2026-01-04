# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, GeluAndMul, RMSNorm, RopeType, build_rotary_embedding,
                                 build_rotary_embedding_from_config)
from lmdeploy.pytorch.nn.linear import (build_down_linear, build_gateup_linear, build_o_proj, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class GemmaAttention(nn.Module):
    """Rewrite module of GemmaAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        # packed qkv
        self.qkv_proj = build_qkv_proj(hidden_size,
                                       num_q_heads=num_heads,
                                       num_kv_heads=num_key_value_heads,
                                       head_size=head_dim,
                                       bias=config.attention_bias,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device,
                                       num_replicate_kv_heads=num_replicate_kv_heads)

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.model_type = config.model_type

        # attention
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_key_value_heads
        self.scaling = 1 / math.sqrt(config.head_dim)
        if hasattr(config, 'query_pre_attn_scalar'):
            self.scaling = config.query_pre_attn_scalar**-0.5
        if self.model_type == 'gemma3_text':
            sliding_window_pattern = getattr(config, 'sliding_window_pattern', 6)
            is_sliding = bool((layer_idx + 1) % sliding_window_pattern)
            self.sliding_window = (getattr(config, 'sliding_window', -1) if is_sliding else -1)
        else:
            self.sliding_window = (getattr(config, 'sliding_window', -1) if not bool(layer_idx % 2) else -1)
        logit_softcapping = getattr(config, 'attn_logit_softcapping', 0.0)
        if logit_softcapping is None:
            logit_softcapping = 0.0
        self.attn_fwd = Attention(num_heads,
                                  head_dim,
                                  scale=self.scaling,
                                  num_kv_heads=num_key_value_heads,
                                  sliding_window=self.sliding_window,
                                  logit_softcapping=logit_softcapping)

        # o_proj
        self.o_proj = build_o_proj(num_heads * head_dim,
                                   hidden_size,
                                   bias=config.attention_bias,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True)

        if self.model_type == 'gemma3_text':
            self.q_norm = RMSNorm(config.head_dim,
                                  config.rms_norm_eps,
                                  quant_config=quantization_config,
                                  dtype=dtype,
                                  device=device)
            self.k_norm = RMSNorm(config.head_dim,
                                  config.rms_norm_eps,
                                  quant_config=quantization_config,
                                  dtype=dtype,
                                  device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        rotary_pos_emb_local: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
        global_attn_masks: torch.Tensor = None,
        local_attn_masks: torch.Tensor = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)

        if self.model_type == 'gemma3_text':
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # apply rotary embedding
        if rotary_pos_emb_local is not None and self.sliding_window != -1:
            cos, sin = rotary_pos_emb_local
        else:
            cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        gemma3_naive_attn_with_masks = global_attn_masks is not None and local_attn_masks is not None
        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=not gemma3_naive_attn_with_masks,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # gemma3 VL applied different attn masks
        # intentionally compute attn twice to fill kv cache
        if gemma3_naive_attn_with_masks is True:
            attn_masks = local_attn_masks if self.sliding_window > 0 else global_attn_masks

            attn_output = self.naive_attn_with_masks(query_states,
                                                     key_states,
                                                     value_states,
                                                     out=attn_output,
                                                     attn_masks=attn_masks,
                                                     seq_lens=attn_metadata.q_seqlens)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output

    # adapted from https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/model_executor/models/gemma3.py#L218  # noqa
    def naive_attn_with_masks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        attn_masks: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        q_len = q.shape[0]
        q = q.view(q_len, -1, self.head_dim)
        # Expand the key and value to handle GQA.
        num_queries_per_kv = self.num_heads // self.num_kv_heads
        k = k.view(q_len, -1, self.head_dim)
        k = k.repeat_interleave(num_queries_per_kv, dim=-2)
        v = v.view(q_len, -1, self.head_dim)
        v = v.repeat_interleave(num_queries_per_kv, dim=-2)

        start_idx = 0
        for seq_len, attn_mask in zip(seq_lens, attn_masks):
            end_idx = start_idx + seq_len
            query = q[start_idx:end_idx].unsqueeze(0)
            key = k[start_idx:end_idx].unsqueeze(0)
            value = v[start_idx:end_idx].unsqueeze(0)

            # Transpose.
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                self.scaling,
            )
            output = output.transpose(1, 2).flatten(-2, -1)
            out[start_idx:end_idx] = output
            start_idx = end_idx
        return out


class GemmaMLP(nn.Module):
    """mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_gateup_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        hidden_activation = config.hidden_activation
        if hidden_activation is None:
            hidden_activation = 'gelu_pytorch_tanh'
            assert hidden_activation == 'gelu_pytorch_tanh'
        self.act_fn = GeluAndMul(approximate='tanh')

        # down
        self.down_proj = build_down_linear(config.intermediate_size,
                                           config.hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        out = self.down_proj(act)
        return out


class GemmaDecoderLayer(nn.Module):
    """Llama decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = GemmaAttention(config, layer_idx, dtype=dtype, device=device)

        # build MLP
        self.mlp = GemmaMLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps,
                                                quant_config=quantization_config,
                                                dtype=dtype,
                                                device=device)

        self.model_type = config.model_type
        if self.model_type in ('gemma2', 'gemma3_text'):
            self.pre_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                     config.rms_norm_eps,
                                                     quant_config=quantization_config,
                                                     dtype=dtype,
                                                     device=device)
            self.post_feedforward_layernorm = RMSNorm(config.hidden_size,
                                                      config.rms_norm_eps,
                                                      quant_config=quantization_config,
                                                      dtype=dtype,
                                                      device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        rotary_pos_emb_local: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None,
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        global_attn_masks: torch.Tensor = None,
        local_attn_masks: torch.Tensor = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_emb_local=rotary_pos_emb_local,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
            global_attn_masks=global_attn_masks,
            local_attn_masks=local_attn_masks,
        )

        # Fully Connected

        if self.model_type in ('gemma2', 'gemma3_text'):
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, residual = self.pre_feedforward_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
            hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    """This module overrides nn.Embeddings' forward by multiplying with
    embeddings scale."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 padding_idx: int,
                 dtype=torch.dtype,
                 embed_scale: Optional[float] = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx, dtype=dtype)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale


class GemmaModel(nn.Module):
    """model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if self.config.model_type == 'gemma3_text':
            self.embed_tokens = Gemma3TextScaledWordEmbedding(config.vocab_size,
                                                              config.hidden_size,
                                                              self.padding_idx,
                                                              dtype=dtype,
                                                              embed_scale=config.hidden_size**0.5)
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size,
                                             config.hidden_size,
                                             self.padding_idx,
                                             dtype=dtype,
                                             device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            GemmaDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = build_rotary_embedding_from_config(config)

        if self.model_type == 'gemma3_text':
            rope_dim = config.head_dim
            rope_max_pos_emb = config.max_position_embeddings
            rope_base = config.rope_local_base_freq
            self.rotary_emb_local = build_rotary_embedding(
                rope_dim,
                rope_max_pos_emb,
                rope_base,
                emb_type=RopeType.Default,
            )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        global_attn_masks: torch.Tensor = None,
        local_attn_masks: torch.Tensor = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        if self.model_type != 'gemma3_text':
            hidden_states = hidden_states * (self.config.hidden_size**0.5)

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)
        rotary_pos_emb_local = None
        if self.model_type == 'gemma3_text':
            cos_local, sin_local = self.rotary_emb_local(hidden_states, position_ids)
            cos_local, sin_local = cos_local[0], sin_local[0]
            rotary_pos_emb_local = (cos_local, sin_local)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_local=rotary_pos_emb_local,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
                global_attn_masks=global_attn_masks,
                local_attn_masks=local_attn_masks,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class GemmaForCausalLM(nn.Module, CudaGraphMixin):
    """ModelForCausalLM."""

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

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build model
        self.model = GemmaModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)
        self.final_logit_softcapping = getattr(config, 'final_logit_softcapping', None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        global_attn_masks: torch.Tensor = None,
        local_attn_masks: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            global_attn_masks=global_attn_masks,
            local_attn_masks=local_attn_masks,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )

    def update_weights(self):
        """Update weights."""
        self.lm_head.weight = self.model.embed_tokens.weight

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]
        norm_layers = [
            '.norm', '.input_layernorm', '.post_attention_layernorm', 'pre_feedforward_layernorm',
            'post_feedforward_layernorm', 'q_norm', 'k_norm'
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if 'lm_head' in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                for weight_name in norm_layers:
                    if weight_name not in name:
                        continue
                    loaded_weight += 1
                    break
                param = params_dict[name]
                load_weight(param, loaded_weight)
