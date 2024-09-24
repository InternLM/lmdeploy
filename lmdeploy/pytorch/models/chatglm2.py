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

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


def get_vision_expert_mask(token_type_ids: torch.LongTensor):
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1]
                                 == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:]
                                                          == VISION_TOKEN_TYPE)
    language_token_mask = ~vision_token_mask
    return vision_token_mask, language_token_mask


def build_position_ids(x: torch.BoolTensor) -> torch.LongTensor:
    tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (
        tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (
        tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | (
        (tmp[:, 1:] == VISION_TOKEN_TYPE) &
        (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y


def _get_cogvlm_position_ids(context):
    """get cogvlm position_ids."""
    q_seqlens = context.q_seqlens
    history_lengths = context.kv_seqlens - q_seqlens
    vision_input_info = context.vision_inputs
    position_id_offsets = (vision_input_info.history_image_token_lengths -
                           vision_input_info.history_image_nums * 3)
    lang_ids = None
    vis_ids = None
    if context.is_decoding:
        position_ids = history_lengths - position_id_offsets
    else:
        if vision_input_info.input_embeddings is not None and len(
                vision_input_info.input_embeddings) > 0:
            starts = history_lengths - vision_input_info.history_lengths
            ends = starts + q_seqlens
            token_type_ids = vision_input_info.input_embedding_indexing.to(
                torch.int)
            history_position_lengths = (vision_input_info.history_lengths -
                                        position_id_offsets)
            position_ids_all = (history_position_lengths[:, None] +
                                build_position_ids(token_type_ids))
            position_ids = torch.cat([
                pids[s:e]
                for (pids, s, e) in zip(position_ids_all, starts, ends)
            ])
            vision_token_mask_all, _ = get_vision_expert_mask(token_type_ids)
            vision_token_mask = torch.cat([
                masks[s:e]
                for (masks, s, e) in zip(vision_token_mask_all, starts, ends)
            ])
            mask_indexing = torch.arange(vision_token_mask.shape[-1],
                                         device=vision_token_mask.device)
            vis_ids = mask_indexing[vision_token_mask]
            lang_ids = mask_indexing[~vision_token_mask]

        else:
            position_ids = context.attention_mask.long().cumsum(-1) - 1
            position_ids += (history_lengths -
                             position_id_offsets).unsqueeze(-1)
            device = position_ids.device
            position_ids_1d = [
                ids[:l] for ids, l in zip(position_ids.cpu(), q_seqlens.cpu())
            ]
            position_ids = torch.cat(position_ids_1d).to(device)

    return position_ids, lang_ids, vis_ids


class SelfAttention(torch.nn.Module):
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

        self.projection_size = config.kv_channels * config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = self.num_attention_heads
        self.head_size = (self.projection_size // config.num_attention_heads)
        self.multi_query_attention = config.multi_query_attention
        if self.multi_query_attention:
            self.num_kv_heads = config.multi_query_group_num
        self.query_key_value = build_qkv_proj(
            config.hidden_size,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            bias=config.add_bias_linear or config.add_qkv_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # apply rotary
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_size,
            num_kv_heads=self.num_kv_heads,
        )

        # o_proj
        self.dense = build_rowwise_linear(self.projection_size,
                                          config.hidden_size,
                                          bias=config.add_bias_linear,
                                          quant_config=quantization_config,
                                          dtype=dtype,
                                          device=device,
                                          is_tp=True)

    @staticmethod
    def _extract_rope(states: torch.Tensor):
        """extract rope."""
        rope = states.chunk(2, -1)[0]
        rope = rope.unflatten(-1, (-1, 2))
        rope = rope.transpose(-2, -1).flatten(-2, -1).contiguous()
        return rope

    @staticmethod
    def _fill_rope(states: torch.Tensor, rope: torch.Tensor):
        """fill rope."""
        rope_part = states.chunk(2, -1)[0]
        rope = rope.unflatten(-1, (2, -1))
        rope = rope.transpose(-2, -1).flatten(-2, -1)
        rope_part.copy_(rope)
        return states

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
        cos, sin = rotary_pos_emb
        q_rope = self._extract_rope(query_states)
        k_rope = self._extract_rope(key_states)
        q_rope, k_rope = self.apply_rotary_pos_emb(
            q_rope,
            k_rope,
            cos,
            sin,
            inplace=True,
        )
        query_states = self._fill_rope(query_states, q_rope)
        key_states = self._fill_rope(key_states, k_rope)

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


class MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.add_bias = config.add_bias_linear
        # gate up
        self.dense_h_to_4h = build_merged_colwise_linear(
            config.hidden_size,
            [config.ffn_hidden_size, config.ffn_hidden_size],
            bias=self.add_bias,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.dense_4h_to_h = build_rowwise_linear(
            config.ffn_hidden_size,
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


class GLMBlock(torch.nn.Module):
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
        self.apply_residual_connection_post_layernorm = \
            config.apply_residual_connection_post_layernorm
        assert not self.apply_residual_connection_post_layernorm

        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attention = SelfAttention(config, dtype=dtype, device=device)

        # builf MLP
        self.mlp = MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.layernorm_epsilon,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.layernorm_epsilon,
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
            layernorm_output = self.input_layernorm(hidden_states)
        else:
            layernorm_output, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        layernorm_input = self.self_attention(
            hidden_states=layernorm_output,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        layernorm_output, residual = self.post_attention_layernorm(
            layernorm_input, residual)
        mlp_output = self.mlp(layernorm_output)

        outputs = (mlp_output, residual)
        return outputs


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm

        def build_layer(layer_number):
            """build layer."""
            return GLMBlock(config, layer_number, dtype=dtype, device=device)

        self.layers = torch.nn.ModuleList(
            [build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            assert config.rmsnorm
            self.final_layernorm = RMSNorm(config.hidden_size,
                                           config.layernorm_epsilon,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device)

    def _get_layer(self, layer_number: int):
        """get layer."""
        return self.layers[layer_number]

    def forward(
        self,
        hidden_states: torch.LongTensor,
        rotary_pos_emb: List[torch.Tensor],
        past_key_values: Optional[List[torch.FloatTensor]],
        attn_metadata: Any,
    ):
        """forward."""
        residual = None
        for index in range(self.num_layers):
            layer = self._get_layer(index)
            hidden_states, residual = layer(
                hidden_states,
                rotary_pos_emb,
                past_key_value=past_key_values[index],
                residual=residual,
                attn_metadata=attn_metadata,
            )

        if self.post_layer_norm:
            hidden_states, _ = self.final_layernorm(hidden_states, residual)
        return hidden_states


class Embedding(nn.Module):
    """Language model embeddings."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(config.padded_vocab_size,
                                            self.hidden_size,
                                            dtype=dtype,
                                            device=device)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        """Rewrite to not transpose hidden_statens for all models."""
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class ChatGLMModel(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.embedding = Embedding(config, dtype=dtype, device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rotary_dim = (config.hidden_size // config.num_attention_heads
                      if config.kv_channels is None else config.kv_channels)
        rope_max_pos_emb = 1 << 20
        rope_base = 10000 * getattr(config, 'rope_ratio', 1.0)
        self.rotary_pos_emb = build_rotary_embedding(
            rotary_dim // 2,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

        # build encoder
        self.encoder = GLMTransformer(config, dtype=dtype, device=device)

        # output_layers
        self.output_layer = build_rowwise_linear(config.hidden_size,
                                                 config.padded_vocab_size,
                                                 bias=False,
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
            inputs_embeds = self.embedding(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        cos, sin = self.rotary_pos_emb(hidden_states, position_ids)
        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        hidden_states = self.encoder(
            hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embedding


class ChatGLMForConditionalGeneration(nn.Module, CudaGraphMixin):
    """rewrote model of LlamaForCausalLM."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build Model
        self.transformer = ChatGLMModel(config, dtype=dtype, device=device)

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
        return self.transformer.output_layer(hidden_states)

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
        if context.vision_inputs is not None:
            position_ids = _get_cogvlm_position_ids(context)[0][None]

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
            if 'transformer.vision' in name:
                continue
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
            elif '.dense_h_to_4h' in name:
                param = params_dict[name]
                gate, up = param.weight_spliter(loaded_weight)
                load_weight(param, gate, shard_id=0)
                load_weight(param, up, shard_id=1)
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
