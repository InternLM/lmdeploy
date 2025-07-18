# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.llava.configuration_llava import LlavaConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import build_model_from_hf_config
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


class LlavaMultiModalProjector(nn.Module):

    def __init__(self, config: LlavaConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        from transformers.activations import ACT2FN

        self.linear_1 = nn.Linear(config.vision_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True,
                                  dtype=dtype,
                                  device=device)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size,
                                  config.text_config.hidden_size,
                                  bias=True,
                                  dtype=dtype,
                                  device=device)

    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class CLIPVisionEmbeddings(nn.Module):
    """Clip vision embedding."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
            dtype=dtype,
            device=device,
        )

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            self.num_positions,
            self.embed_dim,
            dtype=dtype,
            device=device,
        )
        self.register_buffer('position_ids',
                             torch.arange(self.num_positions, device=device).expand((1, -1)),
                             persistent=False)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """This method allows to interpolate the pre-trained position
        encodings, to be able to use the model on higher resolution images.

        This method is also adapted to support torch.jit tracing.
        """

        num_patches = embeddings.shape[1] - 1
        position_embedding = self.position_embedding.weight.unsqueeze(0)
        num_positions = position_embedding.shape[1] - 1

        # always interpolate when tracing
        # to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embedding(self.position_ids)

        from transformers.utils import torch_int

        class_pos_embed = position_embedding[:, :1]
        patch_pos_embed = position_embedding[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode='bicubic',
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.FloatTensor, interpolate_pos_encoding=False) -> torch.Tensor:
        batch_size, _, height, width = pixel_values.shape
        if not interpolate_pos_encoding and (height != self.image_size or width != self.image_size):
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model"
                             f' ({self.image_size}*{self.image_size}).')
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    """Clip attention."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.scale = self.head_dim**-0.5

        # o_proj
        self.out_proj = build_rowwise_linear(self.embed_dim,
                                             self.embed_dim,
                                             bias=True,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(
        self,
        hidden_states,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ):
        """forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        q, k, v = self.qkv_proj.split_qkv(qkv_states)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if attention_mask is not None and causal_attention_mask is not None:
            attn_mask = attention_mask + causal_attention_mask
        elif causal_attention_mask is not None:
            attn_mask = causal_attention_mask
        else:
            attn_mask = attention_mask

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=self.scale)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.out_proj(attn_output)
        return attn_output


class CLIPMLP(nn.Module):
    """Clip mlp."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        from transformers.activations import ACT2FN
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )
        self.fc2 = build_rowwise_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    """Clip encoder layer."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config, dtype=dtype, device=device)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.mlp = CLIPMLP(config, dtype=dtype, device=device)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
    ):
        """forward."""
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """Clip encoder."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [CLIPEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        vision_feature_layer: int = -1,
    ):
        """forward."""
        hidden_states = inputs_embeds
        num_vision_layers = len(self.layers) + vision_feature_layer + 1
        for _, encoder_layer in enumerate(self.layers[:num_vision_layers]):
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                causal_attention_mask=causal_attention_mask,
            )

            hidden_states = layer_outputs

        return hidden_states


class CLIPVisionTransformer(nn.Module):
    """Clip vision transformer."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = CLIPVisionEmbeddings(config, dtype=dtype, device=device)
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.encoder = CLIPEncoder(config, dtype=dtype, device=device)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        interpolate_pos_encoding: bool = False,
        vision_feature_layer: int = -1,
    ) -> BaseModelOutputWithPooling:
        """forward."""
        hidden_states = self.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states, vision_feature_layer=vision_feature_layer)

        last_hidden_state = encoder_outputs
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=None,
            attentions=None,
        )


@vlm_model
class CLIPVisionModel(nn.Module):
    """Clip vision model."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(config, dtype=dtype, device=device)

    def forward(self,
                pixel_values: torch.FloatTensor,
                interpolate_pos_encoding: bool = False,
                vision_feature_layer: int = -1,
                **kwargs):
        """forward."""
        return self.vision_model(pixel_values,
                                 interpolate_pos_encoding=interpolate_pos_encoding,
                                 vision_feature_layer=vision_feature_layer)


def build_vision_model(vision_config, dtype: torch.dtype = None, device: torch.device = None):
    """Build vision model."""
    model_type = vision_config.model_type

    if model_type == 'clip_vision_model':
        return CLIPVisionModel(vision_config, dtype, device)
    else:
        raise NotImplementedError(f'<{model_type}> is not implemented.')


class LlavaForConditionalGeneration(nn.Module, CudaGraphMixin, DeployModelMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        text_config = config.text_config

        self.vision_tower = build_vision_model(config.vision_config, dtype=dtype, device=device)

        self.language_model = build_model_from_hf_config(text_config, dtype=dtype, device=device)

        self.multi_modal_projector = LlavaMultiModalProjector(config, dtype=dtype, device=device)

        self.input_processor = LLavaInputProcessor(config, dtype)

    def get_image_features(self,
                           pixel_values,
                           vision_feature_layer: int = -1,
                           vision_feature_select_strategy: str = 'default'):
        """Get image features."""
        selected_image_feature = self.vision_tower(pixel_values, vision_feature_layer=vision_feature_layer)[0]
        if vision_feature_select_strategy == 'default':
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == 'full':
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(f'Unexpected select feature strategy: {vision_feature_select_strategy}'  # noqa: E501
                             )
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = image_features.flatten(0, 1)[None]

        return image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            image_features = None
            if pixel_values is not None:
                vision_feature_layer = self.config.vision_feature_layer
                select_strategy = self.config.vision_feature_select_strategy
                image_features = self.get_image_features(pixel_values,
                                                         vision_feature_layer=vision_feature_layer,
                                                         vision_feature_select_strategy=select_strategy)
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                inputs_embeds.masked_scatter_(image_mask[..., None], image_features)

        return self.language_model.forward(input_ids=input_ids,
                                           inputs_embeds=inputs_embeds,
                                           past_key_values=past_key_values,
                                           position_ids=position_ids,
                                           attn_metadata=attn_metadata)

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # vision inputs
        pixel_values = None
        image_mask = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                image_token_id = pixel_values[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                pixel_values = torch.cat([data.data for data in pixel_values])
            else:
                pixel_values = None
                image_mask = None

        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_mask=image_mask,
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
        ]

        # vis model
        lang_prefix = 'language_model.'
        prefix_length = len(lang_prefix)
        new_weights = dict()
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith(lang_prefix):
                new_key = name[prefix_length:]
                new_weights[new_key] = loaded_weight
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

        self.language_model.load_weights(new_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class LLavaInputProcessor(BaseModelInputProcessor):
    """Llava input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

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


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):

    from transformers.image_processing_utils import select_best_resolution

    if not isinstance(grid_pinpoints, list):
        raise TypeError('grid_pinpoints should be a list of tuples or lists')

    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()

    height, width = select_best_resolution(image_size, grid_pinpoints)
    return height // patch_size, width // patch_size


def unpad_image(tensor, original_size):
    """Unpads a PyTorch tensor of a padded and resized image."""
    if not isinstance(original_size, (list, tuple)):
        original_size = original_size.tolist()
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(round(original_height * scale_factor, 7))
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(round(original_width * scale_factor, 7))
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


def image_size_to_num_patches(image_size, grid_pinpoints, patch_size: int):
    """Calculate the number of patches after the preprocessing for images of
    any resolution."""
    from transformers.image_processing_utils import select_best_resolution
    if not isinstance(grid_pinpoints, list):
        raise TypeError('grid_pinpoints should be a list of tuples or lists')

    if not isinstance(image_size, (list, tuple)):
        image_size = image_size.tolist()

    best_resolution = select_best_resolution(image_size, grid_pinpoints)
    height, width = best_resolution

    num_patches = (height // patch_size) * (width // patch_size)
    # add the base patch
    num_patches += 1
    return num_patches


class LlavaNextForConditionalGeneration(LlavaForConditionalGeneration):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config=config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=dtype, device=device))
        self.input_processor = LLavaNextInputProcessor(config, dtype)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int,
        vision_feature_select_strategy: str,
    ):
        # ! infer image_num_patches from image_sizes
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            ) for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            # stacked if input is
            # (batch_size, num_patches, num_channels, height, width)
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            # otherwise has to be stacked from list of
            # (num_patches, num_channels, height, width)
            raise ValueError(f'pixel_values of shape {pixel_values.shape}, '
                             'expect to be of 4 or 5 dimensions')

        selected_image_feature = self.vision_tower(pixel_values, vision_feature_layer=vision_feature_layer)[0]
        if vision_feature_select_strategy == 'default':
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == 'full':
            selected_image_feature = selected_image_feature
        image_features = self.multi_modal_projector(selected_image_feature)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        return image_features

    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):

        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = (self.config.vision_config.image_size // self.config.vision_config.patch_size)

                if vision_feature_select_strategy == 'default':
                    expected_num_patches = height * width
                elif vision_feature_select_strategy == 'full':
                    expected_num_patches = height * width + 1
                if expected_num_patches != base_image_feature.shape[0]:
                    raise ValueError('The number of patches is '
                                     'not consistent with the image size.')

                (num_patch_height, num_patch_width) = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        return image_features

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.Tensor = None,
        image_sizes: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            image_features = None
            if pixel_values is not None:
                vision_feature_layer = self.config.vision_feature_layer
                select_strategy = self.config.vision_feature_select_strategy
                image_sizes = image_sizes.tolist()
                image_features = self.get_image_features(pixel_values,
                                                         image_sizes,
                                                         vision_feature_layer=vision_feature_layer,
                                                         vision_feature_select_strategy=select_strategy)
                image_features = self.pack_image_features(
                    image_features,
                    image_sizes,
                    vision_feature_select_strategy=select_strategy,
                    image_newline=self.image_newline,
                )
                image_features = image_features[None]
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                inputs_embeds.masked_scatter_(image_mask[..., None], image_features)

        return self.language_model.forward(input_ids=input_ids,
                                           inputs_embeds=inputs_embeds,
                                           past_key_values=past_key_values,
                                           position_ids=position_ids,
                                           attn_metadata=attn_metadata)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # vision inputs
        pixel_values = None
        image_sizes = None
        image_mask = None
        if context.input_multimodals is not None:
            img_mms = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            img_mms = [data for im_data in img_mms for data in im_data]
            if len(img_mms) > 0:
                image_token_id = img_mms[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                pixel_values = torch.cat([data.data.flatten(0, 1) for data in img_mms])
                image_sizes = torch.cat([data.meta['image_sizes'] for data in img_mms])
            else:
                pixel_values = None
                image_sizes = None

        # get inputs from context
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing

        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            image_mask=image_mask,
            inputs_embeds=inputs_embeds,
        )


class LLavaNextInputProcessor(BaseModelInputProcessor):
    """Llava input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

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
            image_sizes = input_mm['image_sizes']
            offset = input_mm['offset']
            image_token_id = input_mm['image_token_id']
            num_pad = input_mm['image_tokens']
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_sizes=image_sizes, image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
