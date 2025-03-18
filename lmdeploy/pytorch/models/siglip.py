# Copyright (c) OpenMMLab. All rights reserved.

import math
from typing import Iterable, Set, Tuple, Union

import torch
from torch import nn
from transformers import SiglipVisionConfig

from lmdeploy.pytorch.model_inputs import StepContextManager
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight


class SiglipVisionEmbeddings(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 **kwargs):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(in_channels=config.num_channels,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         padding='valid',
                                         dtype=dtype,
                                         device=device)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim, dtype=dtype, device=device)
        self.register_buffer('position_ids', torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position
        encodings, to be able to use the model on higher resolution images.
        This method is also adapted to support torch.jit tracing and no class
        embeddings.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """  # noqa

        num_patches = embeddings.shape[1]
        num_positions = self.position_embedding.weight.shape[0]

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(math.sqrt(num_positions))
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode='bicubic',
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class SiglipAttention(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 **kwargs) -> None:
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).')

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv_proj = build_qkv_proj(self.embed_dim,
                                       num_q_heads=self.num_heads,
                                       num_kv_heads=self.num_heads,
                                       head_size=self.head_dim,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       bias=True,
                                       device=device)

        self.out_proj = build_rowwise_linear(self.embed_dim,
                                             self.embed_dim,
                                             bias=True,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel."""
        batch_size, q_len, _ = hidden_states.size()
        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = qkv_states.chunk(3, dim=-1)
        query_states = query_states.view(batch_size, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, -1, self.head_dim).transpose(1, 2)

        out = nn.functional.scaled_dot_product_attention(query_states, key_states, value_states, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        attn_output = self.out_proj(out)

        return attn_output, None


class SiglipMLP(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 **kwargs) -> None:
        super().__init__()
        from transformers.activations import ACT2FN
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.activation_fn = ACT2FN[config.hidden_act]
        quantization_config = getattr(config, 'quantization_config', None)
        self.fc1 = build_colwise_linear(config.hidden_size,
                                        config.intermediate_size,
                                        bias=True,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True,
                                        quant_config=quantization_config)
        self.fc2 = build_rowwise_linear(config.intermediate_size,
                                        config.hidden_size,
                                        bias=True,
                                        quant_config=quantization_config,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class SiglipEncoderLayer(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 **kwargs) -> None:
        super().__init__()

        self.embed_dim = config.hidden_size

        self.self_attn = SiglipAttention(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.mlp = SiglipMLP(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None


class SiglipEncoder(nn.Module):

    def __init__(self,
                 config: SiglipVisionConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 **kwargs) -> None:
        super().__init__()

        self.config = config
        num_hidden_layers = config.num_hidden_layers

        self.layers = nn.ModuleList([
            SiglipEncoderLayer(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
            for layer_idx in range(num_hidden_layers)
        ])

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        **kwargs,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            hidden_states, _ = encoder_layer(hidden_states)
        return hidden_states


class SiglipMultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(
        self,
        config: SiglipVisionConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = torch.nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.mlp = SiglipMLP(config=config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]


class SiglipVisionTransformer(nn.Module):

    def __init__(
        self,
        config: SiglipVisionConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config, ctx_mgr=ctx_mgr, device=device, dtype=dtype)

        self.encoder = SiglipEncoder(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)

        num_hidden_layers = config.num_hidden_layers
        if len(self.encoder.layers) > config.num_hidden_layers:
            raise ValueError(f'The original encoder only has {num_hidden_layers} '
                             f'layers, but you requested {len(self.encoder.layers)} layers.')

        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)

        self.use_head = (True if not hasattr(config, 'vision_use_head') else config.vision_use_head)
        if self.use_head:
            self.head = SiglipMultiheadAttentionPoolingHead(config=config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = True,
    ) -> torch.Tensor:

        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):
    config_class = SiglipVisionConfig
    main_input_name = 'pixel_values'

    def __init__(
        self,
        config: SiglipVisionConfig,
        ctx_mgr: StepContextManager,
        dtype: torch.dtype = None,
        device: torch.device = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.vision_model = SiglipVisionTransformer(config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        return self.vision_model(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('qkv_proj', 'q_proj', 'q'),
            ('qkv_proj', 'k_proj', 'k'),
            ('qkv_proj', 'v_proj', 'v'),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        layer_count = len(self.vision_model.encoder.layers)

        for name, loaded_weight in weights:
            # post_layernorm is optional in SiglipVisionModel
            if (name.startswith('vision_model.post_layernorm') and self.vision_model.post_layernorm is None):
                continue

            # omit layers when num_hidden_layers_override is set
            if name.startswith('vision_model.encoder.layers'):
                layer_idx = int(name.split('.')[3])
                if layer_idx >= layer_count:
                    continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
