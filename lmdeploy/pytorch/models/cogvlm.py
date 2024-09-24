# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin


def get_world_rank():
    """get current world size and rank."""
    import torch.distributed as dist
    world_size = 1
    rank = 0

    if dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    return world_size, rank


class VisionExpertAttention(nn.Module):
    """Rewrite module of VisionExpertAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        is_cogvlm2 = hasattr(config, 'num_multi_query_heads')
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_multi_query_heads',
                                      num_heads)
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        self.hidden_size = hidden_size
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        # packed qkv
        self.vision_expert_query_key_value = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=is_cogvlm2,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )
        self.language_expert_query_key_value = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=False,
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
        self.vision_expert_dense = build_rowwise_linear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            all_reduce=False)
        self.language_expert_dense = build_rowwise_linear(
            hidden_size,
            hidden_size,
            bias=False,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            all_reduce=False)
        world_size, _ = get_world_rank()
        self.world_size = world_size
        self.all_reduce = world_size > 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        bsz, seqlen, _ = hidden_states.size()
        hidden_size = self.hidden_size // self.world_size
        kv_size = self.num_kv_heads * self.head_dim // self.world_size

        # qkv proj
        if lang_ids is None and vision_ids is None:
            qkv_states = self.language_expert_query_key_value(hidden_states)
        else:
            qkv_states = hidden_states.new_empty(bsz, seqlen,
                                                 hidden_size + kv_size * 2)
            if lang_ids is not None:
                qkv_states[:, lang_ids] = self.language_expert_query_key_value(
                    hidden_states[:, lang_ids])
            if vision_ids is not None:
                qkv_states[:, vision_ids] = self.vision_expert_query_key_value(
                    hidden_states[:, vision_ids])
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = \
            self.language_expert_query_key_value.split_qkv(qkv_states)

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
        if lang_ids is None and vision_ids is None:
            attn_output = self.language_expert_dense(attn_output)
        else:
            new_attn_output = torch.empty_like(hidden_states)
            if lang_ids is not None:
                new_attn_output[:, lang_ids] = self.language_expert_dense(
                    attn_output[:, lang_ids])
            if vision_ids is not None:
                new_attn_output[:, vision_ids] = self.vision_expert_dense(
                    attn_output[:, vision_ids])
            attn_output = new_attn_output

        if self.all_reduce:
            dist.all_reduce(attn_output)
        return attn_output


class MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        assert config.hidden_act == 'silu'

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
                                              is_tp=True,
                                              all_reduce=False)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class VisionExpertMLP(nn.Module):
    """vision expert mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.language_mlp = MLP(config, dtype=dtype, device=device)
        self.vision_mlp = MLP(config, dtype=dtype, device=device)
        world_size, _ = get_world_rank()
        self.all_reduce = world_size > 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
    ):
        """forward."""
        if lang_ids is None and vision_ids is None:
            output = self.language_mlp(hidden_states)
        else:
            output = torch.empty_like(hidden_states)
            if lang_ids is not None:
                output[:,
                       lang_ids] = self.language_mlp(hidden_states[:,
                                                                   lang_ids])
            if vision_ids is not None:
                output[:,
                       vision_ids] = self.vision_mlp(hidden_states[:,
                                                                   vision_ids])
        if self.all_reduce:
            dist.all_reduce(output)
        return output


class CogVLMDecoderLayer(nn.Module):
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
        self.self_attn = VisionExpertAttention(config,
                                               dtype=dtype,
                                               device=device)

        # builf MLP
        self.mlp = VisionExpertMLP(config, dtype=dtype, device=device)

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
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
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
            lang_ids=lang_ids,
            vision_ids=vision_ids,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(
            hidden_states,
            lang_ids=lang_ids,
            vision_ids=vision_ids,
        )

        outputs = (hidden_states, residual)
        return outputs


class CogVLMModel(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        quantization_config = getattr(config, 'quantization_config', None)

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            CogVLMDecoderLayer(config, layer_idx, dtype=dtype, device=device)
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
        rope_max_pos_emb = 2048
        rope_base = 10000
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
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

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
                lang_ids=lang_ids,
                vision_ids=vision_ids,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


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


class CogVLMForCausalLM(nn.Module, CudaGraphMixin):
    """ModelForCausalLM."""

    packed_modules_mapping = {
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
        self.model = CogVLMModel(config, dtype=dtype, device=device)
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
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            lang_ids=lang_ids,
            vision_ids=vision_ids,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
        **kwargs,
    ):
        """support cudagraph."""
        return inputs_embeds is None

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
        position_ids, lang_ids, vis_ids = _get_cogvlm_position_ids(context)
        position_ids = position_ids[None]
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
            lang_ids=lang_ids,
            vision_ids=vis_ids,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'model.vision' in name:
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
                if '_expert_query_key_value' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight)
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
