# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType, SiluAndMul, build_rotary_embedding,
                                 build_rotary_params)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_down_linear, build_gateup_linear, build_o_proj,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h] and returns output of the same size.
    """

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.projection_size = config.kv_channels * config.num_attention_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_size = (self.projection_size // config.num_attention_heads)
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)
        self.query_key_value = build_qkv_proj(config.hidden_size,
                                              num_q_heads=self.num_attention_heads,
                                              num_kv_heads=self.num_kv_heads,
                                              head_size=self.head_size,
                                              bias=config.add_bias_linear or config.add_qkv_bias,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              num_replicate_kv_heads=num_replicate_kv_heads)

        # apply rotary
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            self.num_attention_heads,
            self.head_size,
            num_kv_heads=self.num_kv_heads,
        )

        # o_proj
        self.dense = build_o_proj(self.projection_size,
                                  config.hidden_size,
                                  bias=config.add_bias_linear,
                                  quant_config=quantization_config,
                                  dtype=dtype,
                                  device=device,
                                  is_tp=True)

    @staticmethod
    def _extract_rope(states: torch.Tensor):
        """Extract rope."""
        rope = states.chunk(2, -1)[0]
        rope = rope.unflatten(-1, (-1, 2))
        rope = rope.transpose(-2, -1).flatten(-2, -1).contiguous()
        return rope

    @staticmethod
    def _fill_rope(states: torch.Tensor, rope: torch.Tensor):
        """Fill rope."""
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
        (query_states, key_states, value_states) = self.query_key_value.split_qkv(qkv_states)

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
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.dense(attn_output)
        return attn_output


class MLP(nn.Module):
    """mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)

        self.add_bias = config.add_bias_linear
        # gate up
        self.dense_h_to_4h = build_gateup_linear(
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
        self.dense_4h_to_h = build_down_linear(config.ffn_hidden_size,
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

    Transformer layer takes input with size [s, b, h] and returns an output of the same size.
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

        # build MLP
        self.mlp = MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.layernorm_epsilon,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
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
            layernorm_output, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        layernorm_input = self.self_attention(
            hidden_states=layernorm_output,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        layernorm_output, residual = self.post_attention_layernorm(layernorm_input, residual)
        mlp_output = self.mlp(layernorm_output)

        outputs = (mlp_output, residual)
        return outputs


class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.num_layers = config.num_layers
        self.post_layer_norm = config.post_layer_norm

        def build_layer(layer_number):
            """Build layer."""
            return GLMBlock(config, layer_number, dtype=dtype, device=device)

        self.layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layers)])

        if self.post_layer_norm:
            assert config.rmsnorm
            self.final_layernorm = RMSNorm(config.hidden_size, config.layernorm_epsilon, dtype=dtype, device=device)

    def _get_layer(self, layer_number: int):
        """Get layer."""
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

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(config.padded_vocab_size, self.hidden_size, dtype=dtype, device=device)
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids):
        """Rewrite to not transpose hidden_statens for all models."""
        # Embeddings.
        embeddings = self.word_embeddings(input_ids)
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


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
                                       config.ffn_hidden_size,
                                       bias=False,
                                       dtype=dtype,
                                       device=device)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False, dtype=dtype, device=device)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size,
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
        from argparse import Namespace
        vision_config = Namespace(**config.vision_config)

        self.patch_embedding = PatchEmbedding(vision_config, dtype=dtype, device=device)
        self.transformer = EVA2CLIPTransformer(vision_config, dtype=dtype, device=device)
        self.linear_proj = GLU(config, in_features=config.hidden_size, dtype=dtype, device=device)
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size,
                              out_channels=config.hidden_size,
                              kernel_size=2,
                              stride=2,
                              dtype=dtype,
                              device=device)
        self.boi = nn.Parameter(torch.empty(1, 1, config.hidden_size, dtype=dtype, device=device))
        self.eoi = nn.Parameter(torch.empty(1, 1, config.hidden_size, dtype=dtype, device=device))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images):
        """forward."""
        x = self.patch_embedding(images)
        x = self.transformer(x)

        x = x[:, 1:]

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2)
        x = self.conv(x)

        x = x.flatten(2).transpose(1, 2)
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        x = x / self.scaling_factor
        return x


class ChatGLMModel(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.embedding = Embedding(config, dtype=dtype, device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rotary_dim = (config.hidden_size //
                      config.num_attention_heads if config.kv_channels is None else config.kv_channels)
        rope_max_pos_emb = 1 << 20
        rope_base = 10000 * getattr(config, 'rope_ratio', 1.0)
        rope_params = dict(emb_type=emb_type,
                           dim=rotary_dim // 2,
                           max_position_embeddings=rope_max_pos_emb,
                           base=rope_base)
        update_params = build_rotary_params(config)
        rope_params.update(update_params)
        self.rotary_pos_emb = build_rotary_embedding(**rope_params)

        # build encoder
        self.encoder = GLMTransformer(config, dtype=dtype, device=device)

        # output_layers
        self.output_layer = build_rowwise_linear(config.hidden_size,
                                                 config.padded_vocab_size,
                                                 bias=False,
                                                 dtype=dtype,
                                                 device=device)

        self.vision = None
        if hasattr(config, 'vision_config'):
            self.vision = EVA2CLIPModel(config, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        images: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """forward."""

        # token embedding
        if inputs_embeds is None:
            images_features = None
            if images is not None:
                images_features = self.vision(images)
                images_features = images_features.flatten(0, 1)[None]
            inputs_embeds = self.embedding(input_ids)
            if images is not None:
                inputs_embeds.masked_scatter_(image_mask[..., None], images_features)

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
        """Get input embeddings."""
        return self.embedding


class ChatGLMForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):
    """Rewrote model of LlamaForCausalLM."""

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

        self.input_processor = ChatGLMInputProcessor(self.config, dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        images: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            images=images,
            image_mask=image_mask,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.transformer.output_layer(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.transformer.get_input_embeddings()

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

        images = None
        image_mask = None
        if context.input_multimodals is not None:
            images = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            images = [data for im_data in images for data in im_data]
            if len(images) != 0:
                image_token_id = images[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                images = torch.stack([data.data for data in images])
            else:
                images = None
                image_mask = None

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
            image_mask=image_mask,
            inputs_embeds=inputs_embeds,
        )

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
        if not hasattr(self.config, 'vision_config'):
            return model_metas

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

        config = self.config
        image_size: int = config.vision_config['image_size']
        patch_size: int = config.vision_config['patch_size']
        vision_token_num = ((image_size // patch_size // 2) * (image_size // patch_size // 2) + 2)
        num_pad = vision_token_num - 3

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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'transformer.vision' in name:
                if '.query_key_value' in name:
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight)
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    param = params_dict[name]
                    load_weight(param, loaded_weight)
                continue

            if 'rotary_pos_emb.inv_freq' in name:
                continue
            if ('rotary_pos_emb.cos_cached' in name or 'rotary_pos_emb.sin_cached' in name):
                continue
            if (self.config.tie_word_embeddings and 'output_layer.weight' in name):
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

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class ChatGLMInputProcessor(BaseModelInputProcessor):
    """Input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

        if hasattr(config, 'vision_config'):
            vision_config = config.vision_config
            self.image_size = vision_config['image_size']
            self.patch_size = vision_config['patch_size']
            self.num_patches = (self.image_size // self.patch_size)**2
            self.num_positions = self.num_patches + 1
            self.vision_token_num = self.num_patches // 4

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            num_pad = input_mm['image_tokens']
            image_token_id = input_mm['image_token_id']
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
