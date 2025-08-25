# Copyright (c) OpenMMLab. All rights reserved.

from argparse import Namespace
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig

import lmdeploy.pytorch.distributed as dist
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, RMSNorm, RopeType, SiluAndMul, build_rotary_embedding
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_merged_colwise_linear, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


class VisionExpertAttention(nn.Module):
    """Rewrite module of VisionExpertAttention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        is_cogvlm2 = hasattr(config, 'num_multi_query_heads')
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_heads)
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        self.hidden_size = hidden_size
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        # packed qkv
        self.vision_expert_query_key_value = build_qkv_proj(hidden_size,
                                                            num_q_heads=num_heads,
                                                            num_kv_heads=num_key_value_heads,
                                                            head_size=head_dim,
                                                            bias=is_cogvlm2,
                                                            quant_config=quantization_config,
                                                            dtype=dtype,
                                                            device=device,
                                                            num_replicate_kv_heads=num_replicate_kv_heads)
        self.language_expert_query_key_value = build_qkv_proj(hidden_size,
                                                              num_q_heads=num_heads,
                                                              num_kv_heads=num_key_value_heads,
                                                              head_size=head_dim,
                                                              bias=False,
                                                              quant_config=quantization_config,
                                                              dtype=dtype,
                                                              device=device,
                                                              num_replicate_kv_heads=num_replicate_kv_heads)

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
        )

        # o_proj
        self.vision_expert_dense = build_rowwise_linear(hidden_size,
                                                        hidden_size,
                                                        bias=False,
                                                        quant_config=quantization_config,
                                                        dtype=dtype,
                                                        device=device,
                                                        is_tp=True,
                                                        all_reduce=False)
        self.language_expert_dense = build_rowwise_linear(hidden_size,
                                                          hidden_size,
                                                          bias=False,
                                                          quant_config=quantization_config,
                                                          dtype=dtype,
                                                          device=device,
                                                          is_tp=True,
                                                          all_reduce=False)
        world_size, _ = get_tp_world_rank()
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
            qkv_states = hidden_states.new_empty(bsz, seqlen, hidden_size + kv_size * 2)
            if lang_ids is not None:
                qkv_states[:, lang_ids] = self.language_expert_query_key_value(hidden_states[:, lang_ids])
            if vision_ids is not None:
                qkv_states[:, vision_ids] = self.vision_expert_query_key_value(hidden_states[:, vision_ids])
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
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        if lang_ids is None and vision_ids is None:
            attn_output = self.language_expert_dense(attn_output)
        else:
            new_attn_output = torch.empty_like(hidden_states)
            if lang_ids is not None:
                new_attn_output[:, lang_ids] = self.language_expert_dense(attn_output[:, lang_ids])
            if vision_ids is not None:
                new_attn_output[:, vision_ids] = self.vision_expert_dense(attn_output[:, vision_ids])
            attn_output = new_attn_output

        if self.all_reduce:
            dist.all_reduce(attn_output)
        return attn_output


class MLP(nn.Module):
    """mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
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
    """Vision expert mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.language_mlp = MLP(config, dtype=dtype, device=device)
        self.vision_mlp = MLP(config, dtype=dtype, device=device)
        world_size, _ = get_tp_world_rank()
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
                output[:, lang_ids] = self.language_mlp(hidden_states[:, lang_ids])
            if vision_ids is not None:
                output[:, vision_ids] = self.vision_mlp(hidden_states[:, vision_ids])
        if self.all_reduce:
            dist.all_reduce(output)
        return output


class CogVLMDecoderLayer(nn.Module):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = VisionExpertAttention(config, dtype=dtype, device=device)

        # build MLP
        self.mlp = VisionExpertMLP(config, dtype=dtype, device=device)

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
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

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
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(
            hidden_states,
            lang_ids=lang_ids,
            vision_ids=vision_ids,
        )

        outputs = (hidden_states, residual)
        return outputs


class PatchEmbedding(nn.Module):
    """Vision embedding."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels,
                              config.hidden_size,
                              kernel_size=config.patch_size,
                              stride=config.patch_size,
                              dtype=dtype,
                              device=device)
        self.cls_embedding = nn.Parameter(torch.empty(1, config.hidden_size, dtype=dtype, device=device))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size, dtype=dtype, device=device)

    def forward(self, images):
        """forward."""
        x = self.proj(images)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.position_embedding.weight.unsqueeze(0)
        return x


class EVA2CLIPAttention(nn.Module):
    """Vision attention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        hidden_size = config.hidden_size
        num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim**-0.5

        # packed qkv
        self.query_key_value = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_size=head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # o_proj
        self.dense = build_rowwise_linear(hidden_size,
                                          hidden_size,
                                          bias=True,
                                          quant_config=quantization_config,
                                          dtype=dtype,
                                          device=device,
                                          is_tp=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        # qkv proj
        qkv_states = self.query_key_value(hidden_states)
        q, k, v = self.query_key_value.split_qkv(qkv_states)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.dense(attn_output)
        return attn_output


class EVA2CLIPMLP(nn.Module):
    """Vision MLP."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        from transformers.activations import ACT2FN

        # gate up
        quantization_config = getattr(config, 'quantization_config', None)
        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        if config.hidden_act in ['gelu', 'gelu_fast', 'quick_gelu', 'gelu_python']:
            self.activation_fn = nn.GELU()
        else:
            self.activation_fn = ACT2FN[config.hidden_act]

        # down
        self.fc2 = build_rowwise_linear(config.intermediate_size,
                                        config.hidden_size,
                                        bias=True,
                                        quant_config=quantization_config,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward."""
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class EVA2CLIPTransformerLayer(nn.Module):
    """Vision trans layer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.attention = EVA2CLIPAttention(config, dtype=dtype, device=device)
        self.mlp = EVA2CLIPMLP(config, dtype=dtype, device=device)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size,
                                                     eps=config.layer_norm_eps,
                                                     dtype=dtype,
                                                     device=device)

    def forward(self, hidden_states):
        """forward."""
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class EVA2CLIPTransformer(nn.Module):
    """Vision transformer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.layers = nn.ModuleList(
            [EVA2CLIPTransformerLayer(config, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        """forward."""
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):
    """GLU."""

    def __init__(self,
                 config: PretrainedConfig,
                 in_features: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False, dtype=dtype, device=device)
        self.norm1 = nn.LayerNorm(config.hidden_size, dtype=dtype, device=device)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size,
                                       config.intermediate_size,
                                       bias=False,
                                       dtype=dtype,
                                       device=device)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False, dtype=dtype, device=device)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size,
                                       config.hidden_size,
                                       bias=False,
                                       dtype=dtype,
                                       device=device)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


@vlm_model
class EVA2CLIPModel(nn.Module):
    """Vision model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        vision_config = Namespace(**config.vision_config)

        self.patch_embedding = PatchEmbedding(vision_config, dtype=dtype, device=device)
        self.transformer = EVA2CLIPTransformer(vision_config, dtype=dtype, device=device)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size, dtype=dtype, device=device)
        if vision_config.num_positions == 1226:
            # cogvlm-chat-hf
            self.conv = None
        else:
            # cogvlm2
            self.conv = nn.Conv2d(in_channels=vision_config.hidden_size,
                                  out_channels=vision_config.hidden_size,
                                  kernel_size=2,
                                  stride=2,
                                  dtype=dtype,
                                  device=device)
        self.boi = nn.Parameter(torch.empty(1, 1, config.hidden_size, dtype=dtype, device=device))
        self.eoi = nn.Parameter(torch.empty(1, 1, config.hidden_size, dtype=dtype, device=device))

    def forward(self, images):
        """forward."""
        x = self.patch_embedding(images)
        x = self.transformer(x)

        x = x[:, 1:]
        # cogvlm2
        if self.conv is not None:
            b, s, h = x.shape
            grid_size = int(s**0.5)
            x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
            x = self.conv(x)

            x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        return x


class CogVLMModel(nn.Module):
    """model."""

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
            CogVLMDecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # vision model
        self.vision = EVA2CLIPModel(config, dtype=dtype, device=device)

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
        images: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            if images is not None:
                images_features = self.vision(images)

            inputs_embeds = self.embed_tokens(input_ids)
            if vision_ids is not None:
                inputs_embeds[0, vision_ids] = images_features.flatten(0, 1)

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
        """Get input embeddings."""
        return self.embed_tokens


LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class CogVLMForCausalLM(nn.Module, CudaGraphMixin, DeployModelMixin):
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
        # preprocessor
        self.input_processor = CogVLMInputProcessor(self.config, dtype)
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
        images: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        lang_ids: torch.LongTensor = None,
        vision_ids: torch.LongTensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            images=images,
            inputs_embeds=inputs_embeds,
            lang_ids=lang_ids,
            vision_ids=vision_ids,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

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

        # position_ids, lang_ids, vis_ids = _get_cogvlm_position_ids(context)
        position_ids = context.position_ids
        lang_ids = None
        vis_ids = None

        # vision inputs
        images = None
        if context.input_multimodals is not None:
            images = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            images = [data for im_data in images for data in im_data]
            if len(images) == 0:
                images = None

        if images is not None:
            image_token_id = images[0].meta['image_token_id']
            vis_mask = input_ids[0] == image_token_id
            images = torch.stack([data.data for data in images])

            # get lang_ids
            vis_range = torch.arange(0, input_ids.size(-1), device=input_ids.device)
            vis_ids = vis_range[vis_mask]
            lang_ids = vis_range[~vis_mask]

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
            images=images,
            inputs_embeds=inputs_embeds,
            lang_ids=lang_ids,
            vision_ids=vis_ids,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if '.vision.' in name:
                    continue
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
                elif '.query_key_value' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight)
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)

    def _get_model_metas(self, context: StepContext):
        """Get model metas."""
        model_metas = context.model_metas
        if model_metas is None:
            batch_size = context.q_seqlens.numel()
            return [dict(num_img_tokens=0)] * batch_size
        return [dict(num_img_tokens=0) if meta is None else meta for meta in model_metas]

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """Update model meta."""
        model_metas = self._get_model_metas(context)
        input_multimodals = context.input_multimodals
        if input_multimodals is None:
            input_imgs = [[] for _ in model_metas]
        else:
            input_imgs = []
            for mm in input_multimodals:
                if mm is None:
                    input_imgs.append([])
                else:
                    input_imgs.append(mm.get('image', []))

        num_pad = self.input_processor.vision_token_num - 3

        batched_num_img_tokens = []
        new_model_metas = []
        for meta, imgs in zip(model_metas, input_imgs):
            if meta is None:
                num_img_tokens = 0
            else:
                num_img_tokens = meta.get('num_img_tokens', 0)

            batched_num_img_tokens.append(num_img_tokens)

            num_img_tokens += num_pad * len(imgs)
            new_model_metas.append(dict(num_img_tokens=num_img_tokens))

        # prepare cogvlm position_ids
        q_seqlens = context.q_seqlens
        position_ids = context.position_ids

        if context.is_decoding or all(len(imgs) == 0 for imgs in input_imgs):
            num_img_tokens = torch.tensor(batched_num_img_tokens, device=position_ids.device)
            position_ids -= num_img_tokens[None]
        else:
            batched_position_ids = position_ids[0].split(q_seqlens)
            for pos_ids, num_img_tok, imgs in zip(batched_position_ids, batched_num_img_tokens, input_imgs):
                pos_ids -= num_img_tok
                if len(imgs) == 0:
                    continue

                seq_len = pos_ids.size(0)
                start = pos_ids[0].cpu().item()
                new_pos_ids = []

                imgs = sorted(imgs, key=lambda img: img.start)
                for img in imgs:
                    img_pad_pos = img.start + 1 - num_img_tok
                    num_pad = img.end - img.start - 2
                    new_pos_ids += list(range(start, img_pad_pos))
                    new_pos_ids += [img_pad_pos] * num_pad
                    start = img_pad_pos + 1
                    num_img_tok += num_pad

                remain = seq_len - len(new_pos_ids)
                new_pos_ids += list(range(start, start + remain))

                new_pos_ids = pos_ids.new_tensor(new_pos_ids)
                pos_ids[:] = new_pos_ids

            position_ids = torch.cat(batched_position_ids)[None]
        context.position_ids = position_ids

        return new_model_metas

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class CogVLMInputProcessor(BaseModelInputProcessor):
    """Input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype
        image_size: int = config.vision_config['image_size']
        patch_size: int = config.vision_config['patch_size']
        if config.vision_config['num_positions'] == 1226:
            # # cogvlm-chat-hf
            self.vision_token_num = 2 + (image_size // patch_size)**2
        else:
            # cogvlm2
            self.vision_token_num = 2 + (image_size // patch_size // 2)**2

    def preprocess_input(self, input_ids: List[int], input_multimodals=None, **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            image_token_id = input_mm['image_token_id']
            num_pad = input_mm['image_tokens']
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
