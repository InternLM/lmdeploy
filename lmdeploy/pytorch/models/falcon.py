# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RopeType,
                                 build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class FalconAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of
    the same size.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = self.num_attention_heads
        self.head_size = (self.hidden_size // config.num_attention_heads)
        self.multi_query_attention = config.multi_query
        if self.multi_query_attention:
            self.num_kv_heads = 1
        self.query_key_value = build_qkv_proj(
            config.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            bias=config.bias,
            replicate_kv=self.multi_query_attention,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # apply rotary
        self.apply_rotary_pos_emb = ApplyRotaryEmb()
        self.rotary = config.rotary

        # attention
        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_size,
            num_kv_heads=self.num_kv_heads,
            alibi=config.alibi,
        )

        # o_proj
        self.dense = build_rowwise_linear(self.hidden_size,
                                          config.hidden_size,
                                          bias=config.bias,
                                          quant_config=quantization_config,
                                          dtype=dtype,
                                          device=device,
                                          is_tp=True,
                                          tp_align_size=self.head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.query_key_value(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        (query_states, key_states,
         value_states) = self.query_key_value.split_qkv(qkv_states)

        # apply rotary embedding
        if self.rotary:
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
        attn_output = self.dense(attn_output)
        return attn_output


class FalconMLP(nn.Module):
    """Falcon mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.add_bias = config.bias
        ffn_hidden_size = getattr(config, 'ffn_hidden_size',
                                  config.hidden_size * 4)
        # gate up
        self.dense_h_to_4h = build_colwise_linear(
            config.hidden_size,
            ffn_hidden_size,
            bias=self.add_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = nn.GELU()

        # down
        self.dense_4h_to_h = build_rowwise_linear(
            ffn_hidden_size,
            config.hidden_size,
            bias=self.add_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.dense_h_to_4h(x)
        act = self.act_fn(gate_up)
        return self.dense_4h_to_h(act)


class FalconDecoderLayer(nn.Module):
    """falcon decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        hidden_size = config.hidden_size

        # build attention layer
        self.self_attention = FalconAttention(config,
                                              dtype=dtype,
                                              device=device)

        # builf MLP
        self.mlp = FalconMLP(config, dtype=dtype, device=device)

        if not hasattr(config, 'num_ln_in_parallel_attn'):
            config.num_ln_in_parallel_attn = None
        if (config.num_ln_in_parallel_attn is None
                and config.new_decoder_architecture):
            config.num_ln_in_parallel_attn = 2

        if not config.parallel_attn:
            self.post_attention_layernorm = nn.LayerNorm(
                hidden_size,
                eps=config.layer_norm_epsilon,
                dtype=dtype,
                device=device)
            self.input_layernorm = nn.LayerNorm(hidden_size,
                                                eps=config.layer_norm_epsilon,
                                                dtype=dtype,
                                                device=device)
        else:
            if config.num_ln_in_parallel_attn == 2:
                # The layer norm before self-attention
                self.ln_attn = nn.LayerNorm(hidden_size,
                                            eps=config.layer_norm_epsilon,
                                            dtype=dtype,
                                            device=device)
                # The layer norm before the MLP
                self.ln_mlp = nn.LayerNorm(hidden_size,
                                           eps=config.layer_norm_epsilon,
                                           dtype=dtype,
                                           device=device)
            else:
                self.input_layernorm = nn.LayerNorm(
                    hidden_size,
                    eps=config.layer_norm_epsilon,
                    dtype=dtype,
                    device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        attn_metadata: Any = None,
    ):

        residual = hidden_states
        if (self.config.new_decoder_architecture
                and self.config.num_ln_in_parallel_attn == 2):
            attention_layernorm_out = self.ln_attn(hidden_states)
            mlp_layernorm_out = self.ln_mlp(hidden_states)
        else:
            attention_layernorm_out = self.input_layernorm(hidden_states)

        # Self Attention
        attention_output = self.self_attention(
            hidden_states=attention_layernorm_out,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        if not self.config.new_decoder_architecture:
            if self.config.parallel_attn:
                mlp_layernorm_out = attention_layernorm_out
            else:
                residual = attention_output + residual
                mlp_layernorm_out = self.post_attention_layernorm(residual)

        if (self.config.new_decoder_architecture and self.config.parallel_attn
                and self.config.num_ln_in_parallel_attn == 1):
            mlp_layernorm_out = attention_layernorm_out

        # MLP.
        mlp_output = self.mlp(mlp_layernorm_out)

        if self.config.new_decoder_architecture or self.config.parallel_attn:
            mlp_output += attention_output

        outputs = mlp_output + residual
        return outputs


class FalconModel(nn.Module):
    """falcon model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            self.embed_dim,
                                            dtype=dtype,
                                            device=device)

        # build all decode layers
        self.h = nn.ModuleList([
            FalconDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self.ln_f = nn.LayerNorm(self.embed_dim,
                                 eps=config.layer_norm_epsilon,
                                 dtype=dtype,
                                 device=device)

        scaling_factor = 1.0
        if not hasattr(config, 'rope_scaling'):
            emb_type = RopeType.LinearScaling
        else:
            rope_scaling = config.rope_scaling
            rope_type = rope_scaling['rope_type']
            if rope_type == 'linear':
                emb_type = RopeType.LinearScaling
            elif rope_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            else:
                raise RuntimeError(f'Unsupported rope type: {rope_type}')
            scaling_factor = rope_scaling.get('scaling_factor', scaling_factor)

        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = getattr(config, 'max_position_embeddings', 2048)
        rope_base = getattr(config, 'rope_base', 10000)
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        for idx, decoder_layer in enumerate(self.h):
            past_key_value = past_key_values[idx]
            hidden_states = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states = self.ln_f(hidden_states)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.word_embeddings


class FalconForCausalLM(nn.Module, CudaGraphMixin):
    """rewrote model of FalconForCausalLM."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build LLamaModel
        self.transformer = FalconModel(config, dtype=dtype, device=device)
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
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.transformer.get_input_embeddings()

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
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_pos_emb.inv_freq' in name:
                continue
            if ('rotary_pos_emb.cos_cached' in name
                    or 'rotary_pos_emb.sin_cached' in name):
                continue
            if (self.config.tie_word_embeddings
                    and 'output_layer.weight' in name):
                continue
            if '.query_key_value' in name:
                param = params_dict[name]
                q, k, v = param.weight_spliter(loaded_weight)
                load_weight(param, q, shard_id='q')
                load_weight(param, k, shard_id='k')
                load_weight(param, v, shard_id='v')
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
