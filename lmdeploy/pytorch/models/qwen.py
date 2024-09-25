# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


class QWenAttention(torch.nn.Module):
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
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.projection_size = config.kv_channels * config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = self.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.c_attn = build_qkv_proj(
            config.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # apply rotary
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_dim,
            num_kv_heads=self.num_kv_heads,
        )

        # o_proj
        self.c_proj = build_rowwise_linear(self.projection_size,
                                           config.hidden_size,
                                           bias=not config.no_bias,
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
        qkv_states = self.c_attn(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        (query_states, key_states,
         value_states) = self.c_attn.split_qkv(qkv_states)

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
        attn_output = self.c_proj(attn_output)
        return attn_output


class QWenMLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        ff_dim_in = config.intermediate_size // 2
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [ff_dim_in, ff_dim_in],
            bias=not config.no_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.c_proj = build_rowwise_linear(ff_dim_in,
                                           config.hidden_size,
                                           bias=not config.no_bias,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.c_proj(act)


class QWenBlock(torch.nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an output of
    the same size.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 layer_number: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_number = layer_number
        hidden_size = config.hidden_size
        self.bf16 = config.bf16

        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.attn = QWenAttention(config, dtype=dtype, device=device)

        # builf MLP
        self.mlp = QWenMLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.ln_1 = RMSNorm(hidden_size,
                            config.layer_norm_epsilon,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        # build attention layer norm
        self.ln_2 = RMSNorm(hidden_size,
                            config.layer_norm_epsilon,
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
            layernorm_output = self.ln_1(hidden_states)
        else:
            layernorm_output, residual = self.ln_1(hidden_states, residual)

        # Self Attention
        layernorm_input = self.attn(
            hidden_states=layernorm_output,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        layernorm_output, residual = self.ln_2(layernorm_input, residual)
        mlp_output = self.mlp(layernorm_output)

        outputs = (mlp_output, residual)
        return outputs


class QWenModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.vocab_size = config.vocab_size
        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(self.vocab_size,
                                self.embed_dim,
                                dtype=dtype,
                                device=device)

        # build all decode layers
        self.h = nn.ModuleList([
            QWenBlock(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            assert config.rotary_pct < 1
            self.rotary_ndims = int(config.kv_channels * config.rotary_pct)
        rope_dim = (self.rotary_ndims
                    if self.rotary_ndims is not None else config.kv_channels)
        rope_max_pos_emb = getattr(config, 'max_position_embeddings', 4096)
        rope_base = config.rotary_emb_base
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

        self.ln_f = RMSNorm(self.embed_dim,
                            eps=config.layer_norm_epsilon,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.h):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

        # norm
        hidden_states, residual = self.ln_f(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.wte


class QWenLMHeadModel(nn.Module, CudaGraphMixin):
    """rewrote model."""

    packed_modules_mapping = {
        'gate_up_proj': [
            'w2',
            'w1',
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
        # build Model
        self.transformer = QWenModel(config, dtype=dtype, device=device)

        # output_layers
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
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.w2', 0),
            ('.gate_up_proj', '.w1', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'visual' in name:
                continue
            if 'rotary_pos_emb.inv_freq' in name:
                continue
            if ('rotary_pos_emb.cos_cached' in name
                    or 'rotary_pos_emb.sin_cached' in name):
                continue
            if (self.config.tie_word_embeddings and 'lm_head.weight' in name):
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if '.c_attn' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight)
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
