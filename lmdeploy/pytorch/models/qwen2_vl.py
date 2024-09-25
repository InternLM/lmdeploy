# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin, next_power_of_2


def _apply_mrope_selection(hidden_states: torch.Tensor,
                           mrope_position_ids: torch.Tensor,
                           mrope_section: List[int],
                           position_ids: torch.Tensor,
                           rotary_emb_func: Callable):
    _mrope_position_ids = torch.zeros(3,
                                      position_ids.shape[-1],
                                      dtype=position_ids.dtype,
                                      device=position_ids.device)
    _mrope_position_ids[:, :mrope_position_ids.shape[-1]] = mrope_position_ids
    cos, sin = rotary_emb_func(hidden_states, _mrope_position_ids)
    _cos = torch.zeros(cos.shape[1],
                       cos.shape[-1],
                       dtype=cos.dtype,
                       device=cos.device)
    _sin = torch.zeros_like(_cos)
    mrope_section = mrope_section * 2

    def _apply_split(src, dst):
        start = 0
        for i, m in enumerate(src.split(mrope_section, dim=-1)):
            dst[:, start:start + mrope_section[i]] = m[i % 3]
            start += mrope_section[i]

    _apply_split(cos, _cos)
    _apply_split(sin, _sin)

    return _cos, _sin


class Qwen2Attention(nn.Module):
    """Rewrite module of Qwen2Attention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=config.sliding_window,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(
            qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(config.intermediate_size,
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
        return self.down_proj(act)


class Qwen2DecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = Qwen2Attention(config, dtype=dtype, device=device)

        # builf MLP
        self.mlp = Qwen2MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Qwen2Model(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.mrope_section = config.rope_scaling['mrope_section']
        quantization_config = getattr(config, 'quantization_config', None)

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_position_ids: torch.LongTensor = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        if mrope_position_ids is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            cos, sin = cos[0], sin[0]
        else:
            cos, sin = _apply_mrope_selection(hidden_states,
                                              mrope_position_ids,
                                              self.mrope_section, position_ids,
                                              self.rotary_emb)
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

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


class Qwen2VLForConditionalGeneration(nn.Module, CudaGraphMixin):
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
        self.model = Qwen2Model(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        mrope_position_ids: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
        )

        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits

    def update_weights(self):
        """update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""

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
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=context.mrope_position_ids,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
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
            if 'visual' in name:
                continue
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
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

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta,
                                                       **kwargs)
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'] = mrope_position_ids.new_zeros(
                3, max_tokens)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta,
                                                    **kwargs)

        input_ids = kwargs.get('input_ids')
        attn_metadata = kwargs.get('attn_metadata')
        block_offsets = attn_metadata.block_offsets
        num_tokens = input_ids.size(-1)
        batch_size, _ = block_offsets.size()
        new_batch_size = next_power_of_2(batch_size)

        is_decoding = graph_meta.is_decoding
        input_buffers = graph_meta.input_buffers
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers[
                'mrope_position_ids'][:, :num_tokens] = mrope_position_ids
            if is_decoding:
                new_inputs['mrope_position_ids'] = input_buffers[
                    'mrope_position_ids'][:, :new_batch_size]
            else:
                new_inputs['mrope_position_ids'] = input_buffers[
                    'mrope_position_ids']

        return new_inputs
