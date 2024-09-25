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


class InternLM2Attention(nn.Module):
    """Rewrite module of InternLM2Attention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads

        # packed qkv
        self.wqkv = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=config.bias,
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
        )

        # o_proj
        self.wo = build_rowwise_linear(num_heads * head_dim,
                                       hidden_size,
                                       bias=config.bias,
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
        """Rewrite of InternLM2Attention.forward."""
        # qkv proj
        qkv_states = self.wqkv(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.wqkv.split_qkv(
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
        attn_output = self.wo(attn_output)
        return attn_output


class InternLM2MLP(nn.Module):
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
        self.w2 = build_rowwise_linear(config.intermediate_size,
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
        return self.w2(act)


class InternLM2DecoderLayer(nn.Module):
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
        self.attention = InternLM2Attention(config, dtype=dtype, device=device)

        # builf MLP
        self.feed_forward = InternLM2MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.attention_norm = RMSNorm(config.hidden_size,
                                      config.rms_norm_eps,
                                      quant_config=quantization_config,
                                      dtype=dtype,
                                      device=device)

        # build attention layer norm
        self.ffn_norm = RMSNorm(config.hidden_size,
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
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.attention(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class InternLM2Model(nn.Module):
    """internlm2 model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        quantization_config = getattr(config, 'quantization_config', None)

        self.tok_embeddings = nn.Embedding(config.vocab_size,
                                           config.hidden_size,
                                           self.padding_idx,
                                           dtype=dtype,
                                           device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            InternLM2DecoderLayer(config,
                                  layer_idx,
                                  dtype=dtype,
                                  device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        # build rotary embedding in Model
        rope_scaling = config.rope_scaling
        scaling_factor = 1.0
        emb_type = RopeType.LinearScaling
        if rope_scaling is not None:
            scaling_factor = rope_scaling.get('factor', scaling_factor)
            rope_type = rope_scaling['type']
            if rope_type == 'linear':
                emb_type = RopeType.LinearScaling
            if rope_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            else:
                raise RuntimeError(f'Unsupported rope type: {rope_type}')
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
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
        """Rewrite of forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.tok_embeddings(input_ids)

        hidden_states = inputs_embeds

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

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.tok_embeddings


class InternLM2ForCausalLM(nn.Module, CudaGraphMixin):
    """rewrote model of InternLM2ForCausalLM."""

    packed_modules_mapping = {
        'gate_up_proj': [
            'w1',
            'w3',
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
        self.model = InternLM2Model(config, dtype=dtype, device=device)
        # build lm_head
        self.output = build_rowwise_linear(config.hidden_size,
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
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.output(hidden_states)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """support cudagraph."""
        seq_lens = input_ids.size(1)
        if seq_lens <= 512:
            return True
        return False

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
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.w1', 0),
            ('.gate_up_proj', '.w3', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                load_weight(param, loaded_weight, shard_id=shard_id)
                break
            else:
                if '.wqkv' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight, layout='hgd')
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
