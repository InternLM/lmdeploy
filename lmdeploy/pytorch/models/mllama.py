# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.models.llama import LlamaConfig
from transformers.models.mllama.modeling_mllama import MllamaTextConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.nn.rotary_embedding import Llama3Parameters
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin

MLLAMA_IMAGE_TOKEN_ID = 128256
MLLAMA_IMAGE_TOKEN = '<|image|>'


class LlamaAttention(nn.Module):
    """Rewrite module of LlamaAttention."""

    def __init__(self,
                 config: LlamaConfig,
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
            bias=getattr(config, 'attention_bias', False),
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
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
                                           hidden_size,
                                           bias=getattr(
                                               config, 'attention_bias',
                                               False),
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
            k_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class MllamaTextCrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self,
                 config: Optional[MllamaTextConfig] = None,
                 layer_idx: Optional[int] = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // self.num_heads
        self.layer_idx = layer_idx
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            self.hidden_size,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_key_value_heads,
            head_size=self.head_dim,
            bias=getattr(config, 'attention_bias', False),
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )
        self.o_proj = build_rowwise_linear(self.num_heads * self.head_dim,
                                           self.hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

        # attention
        self.attn_fwd = Attention(
            self.num_heads,
            self.head_dim,
            num_kv_heads=self.num_key_value_heads,
            v_head_size=self.head_dim,
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        cross_attention_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cross_attn_metadata: Any = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel."""
        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, _, _ = self.qkv_proj.split_qkv(qkv_states)
        query_states = query_states.view(-1, query_states.shape[-2],
                                         self.head_dim)
        query_states = self.q_norm(query_states)

        if cross_attention_states is not None:
            qkv_states = self.qkv_proj(cross_attention_states)
            qkv_states = qkv_states.flatten(0, -2)
            _, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)
            key_states = key_states.view(-1, key_states.shape[-2],
                                         self.head_dim)
            value_states = value_states.view(-1, value_states.shape[-2],
                                             self.head_dim)
            key_states = self.k_norm(key_states)
        else:
            key_states = None
            value_states = None

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            cross_attn_metadata,
            k_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    """llama mlp."""

    def __init__(self,
                 config: LlamaConfig,
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


class MllamaSelfAttentionDecoderLayer(nn.Module):
    """llama decoder layer."""

    def __init__(self,
                 config: LlamaConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = LlamaAttention(config, dtype=dtype, device=device)

        # builf MLP
        self.mlp = LlamaMLP(config, dtype=dtype, device=device)

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

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
                past_key_value: Optional[List[torch.FloatTensor]],
                cross_attention_states: Optional[torch.FloatTensor] = None,
                full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
                residual: Optional[torch.Tensor] = None,
                attn_metadata: Any = None,
                cross_attn_metadata: Any = None):

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


class MllamaCrossAttentionDecoderLayer(nn.Module):
    """llama decoder layer."""

    def __init__(self,
                 config: LlamaConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.cross_attn = MllamaTextCrossAttention(config,
                                                   dtype=dtype,
                                                   device=device)

        # builf MLP
        self.mlp = LlamaMLP(config, dtype=dtype, device=device)

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
        self.cross_attn_attn_gate = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype))
        self.cross_attn_mlp_gate = torch.nn.Parameter(
            torch.zeros(1, dtype=dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: torch.Tensor,
        full_text_row_masked_out_mask: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        cross_attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Cross Attention
        hidden_states = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            cross_attn_metadata=cross_attn_metadata,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh(
        ) * hidden_states
        residual = hidden_states

        # Fully Connected
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = full_text_row_masked_out_mask * hidden_states
        hidden_states = residual + self.cross_attn_mlp_gate.tanh(
        ) * hidden_states

        outputs = (hidden_states, None)
        return outputs


class MllamaTextModel(nn.Module):
    """llama model."""

    def __init__(self,
                 config: LlamaConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size + 8,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)
        self.cross_attention_layers = config.cross_attention_layers

        # build all decode layers
        layers = []
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.cross_attention_layers:
                layers.append(
                    MllamaCrossAttentionDecoderLayer(config,
                                                     layer_idx,
                                                     dtype=dtype,
                                                     device=device))
            else:
                layers.append(
                    MllamaSelfAttentionDecoderLayer(config,
                                                    layer_idx,
                                                    dtype=dtype,
                                                    device=device))
        self.layers = nn.ModuleList(layers)

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            dtype=dtype,
                            device=device)

        # build rotary embedding in LlamaModel
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        scaling_factor = 1.0
        llama3_params = None
        rope_scaling = config.rope_scaling
        if rope_scaling is None:
            emb_type = RopeType.LinearScaling
        else:
            if 'scaling_factor' in rope_scaling:
                scaling_factor = rope_scaling['scaling_factor']
            elif 'factor' in rope_scaling:
                scaling_factor = rope_scaling['factor']

            rope_type = rope_scaling['rope_type']
            if rope_type == 'dynamic':
                emb_type = RopeType.DynamicNTKScaling
            elif rope_type == 'linear':
                emb_type = RopeType.LinearScaling
            elif rope_type == 'llama3':
                emb_type = RopeType.Llama3
                low_freq_factor = rope_scaling.get('low_freq_factor', 1.0)
                high_freq_factor = rope_scaling.get('high_freq_factor', 1.0)
                llama3_params = Llama3Parameters(low_freq_factor,
                                                 high_freq_factor)
            else:
                raise RuntimeError(f'Unsupported rope type: {rope_type}')

        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            scaling_factor,
            llama3_params=llama3_params,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        cross_attention_states: Optional[torch.FloatTensor] = None,
        full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cross_attn_metadata: Any = None,
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
            if full_text_row_masked_out_mask is None and idx in self.cross_attention_layers:  # noqa
                continue
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                cross_attention_states=cross_attention_states,
                full_text_row_masked_out_mask=full_text_row_masked_out_mask,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
                cross_attn_metadata=cross_attn_metadata,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


class MllamaForCausalLM(nn.Module):
    """llama model."""

    def __init__(self,
                 config: MllamaTextConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.text_config = config.get_text_config()
        self.vocab_size = self.text_config.vocab_size

        self.model = MllamaTextModel(config, dtype=dtype, device=device)
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
        cross_attention_states: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        cross_attn_metadata: Any = None,
        **kwargs,
    ):
        """model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            cross_attention_states=cross_attention_states,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            cross_attn_metadata=cross_attn_metadata,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)


class MllamaForConditionalGeneration(nn.Module, CudaGraphMixin):
    """rewrote model of MllamaForConditionalGeneration."""

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
                 config: LlamaConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build MllamaForCausalLM
        self.language_model = MllamaForCausalLM(config.text_config,
                                                dtype=dtype,
                                                device=device)
        self.dtype = dtype

    def flat_encoder_result(self, cross_attention_states: torch.Tensor,
                            attn_metadata: Any, input_ids: torch.LongTensor):
        # since every state share the same shape
        cross_attention_states = torch.cat(cross_attention_states, 0)
        full_text_row_masked_out_mask = torch.ones(
            (attn_metadata.q_seqlens.sum(), 1), dtype=torch.bool)
        start_pos = 0
        img_idx = torch.where(input_ids == MLLAMA_IMAGE_TOKEN_ID)[1]
        for img_id, q_seq_len in zip(img_idx.cpu(),
                                     attn_metadata.q_seqlens.cpu()):
            full_text_row_masked_out_mask[start_pos:img_id] = False
            start_pos += q_seq_len
        full_text_row_masked_out_mask = full_text_row_masked_out_mask.to(
            cross_attention_states.device)

        return cross_attention_states, full_text_row_masked_out_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        cross_attention_states: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        cross_attn_metadata: Any = None,
        **kwargs,
    ):
        """model forward, return logits."""
        if cross_attn_metadata is None:
            full_text_row_masked_out_mask = None
        # FIXME basically, we want to inference
        # text requests and image requests separately
        elif cross_attention_states is None and (
                cross_attn_metadata.kv_seqlens is None
                or int(cross_attn_metadata.kv_seqlens.sum()) == 0):
            full_text_row_masked_out_mask = None
        elif cross_attn_metadata.is_decoding:
            cross_attention_states = None
            full_text_row_masked_out_mask = torch.ones(
                (attn_metadata.q_seqlens.sum(), 1),
                dtype=torch.bool,
                device=input_ids.device)
        else:
            cross_attention_states, full_text_row_masked_out_mask = \
                self.flat_encoder_result(cross_attention_states, cross_attn_metadata, input_ids)  # noqa
        hidden_states = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            cross_attention_states=cross_attention_states,
            full_text_row_masked_out_mask=full_text_row_masked_out_mask,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            cross_attn_metadata=cross_attn_metadata,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def support_cuda_graph(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        """support cudagraph."""

        return False

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.language_model.model.get_input_embeddings()

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
        cross_attention_states = context.cross_attention_states
        if cross_attention_states is not None:
            cross_attention_states = [
                t.to(input_ids.device) for t in cross_attention_states
                if t is not None
            ]
        cross_attn_metadata = context.cross_attn_metadata

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
            cross_attention_states=cross_attention_states,
            cross_attn_metadata=cross_attn_metadata,
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
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
                continue
            if 'vision_model' in name or 'multi_modal_projector' in name:
                continue
            if self.config.text_config.tie_word_embeddings and 'lm_head.weight' in name:  # noqa
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
