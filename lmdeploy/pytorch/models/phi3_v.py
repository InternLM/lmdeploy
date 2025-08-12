# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPVisionConfig, CLIPVisionModel, PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn.linear import build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .phi3 import Phi3ForCausalLM, Phi3Model
from .utils.model import DeployModelMixin, vlm_model

CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(attention_dropout=0.0,
                                                     dropout=0.0,
                                                     hidden_act='quick_gelu',
                                                     hidden_size=1024,
                                                     image_size=336,
                                                     initializer_factor=1.0,
                                                     initializer_range=0.02,
                                                     intermediate_size=4096,
                                                     layer_norm_eps=1e-05,
                                                     num_attention_heads=16,
                                                     num_channels=3,
                                                     num_hidden_layers=24,
                                                     patch_size=14,
                                                     projection_dim=768)


@vlm_model
class Phi3ImageEmbedding(nn.Module):
    """Image embedding."""

    # from https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/c45209e90a4c4f7d16b2e9d48503c7f3e83623ed/image_embedding_phi3_v.py#L83 # noqa: E501
    def __init__(self,
                 config: PretrainedConfig,
                 wte=None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 **kwargs):
        super().__init__()
        self.config = config
        hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size

        self.wte = wte

        if (isinstance(config.img_processor, dict) and config.img_processor.get('name', None) == 'clip_vision_model'):
            assert 'model_name' in config.img_processor, ('model_name must be provided for CLIPVisionModel')
            assert 'image_dim_out' in config.img_processor, ('image_dim_out must be provided for CLIPVisionModel')
            assert 'num_img_tokens' in config.img_processor, ('num_img_tokens must be provided for CLIPVisionModel')
            assert config.img_processor['model_name'] == 'openai/clip-vit-large-patch14-336'
            clip_config = CLIP_VIT_LARGE_PATCH14_336_CONFIG
            self.img_processor = CLIPVisionModel(clip_config).to(device).to(dtype)
            image_dim_out = config.img_processor['image_dim_out']
            self.num_img_tokens = config.img_processor['num_img_tokens']
        else:
            raise NotImplementedError(f'img_processor = {config.img_processor}, not implemented')

        self.image_dim_out = image_dim_out
        self.img_sizes = None

        self.use_hd_transform = kwargs.get('use_hd_transform', False)
        self.with_learnable_separator = kwargs.get('with_learnable_separator', False)
        self.hd_transform_order = kwargs.get('hd_transform_order', 'glb_sub')
        # with_hd_transform and with_learnable_separator should have same value
        assert (self.use_hd_transform == self.with_learnable_separator), (
            'use_hd_transform and with_learnable_separator '
            'should have same value')
        if self.with_learnable_separator:
            assert self.use_hd_transform, ('learnable separator is only for hd transform')
            # 1024 * 4, merge spatial to channel dimension
            self.glb_GN = nn.Parameter(torch.empty([1, 1, self.image_dim_out * 4], dtype=dtype, device=device))
            self.sub_GN = nn.Parameter(torch.empty([1, 1, 1, self.image_dim_out * 4], dtype=dtype, device=device))

        projection_cls = kwargs.get('projection_cls', 'linear')
        if projection_cls == 'linear':
            self.img_projection = nn.Linear(image_dim_out, hidden_size, dtype=dtype, device=device)
        elif projection_cls == 'mlp' and self.use_hd_transform:
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out * 4, dim_projection, dtype=dtype, device=device)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection, dtype=dtype, device=device)])
            self.img_projection = nn.Sequential(*layers)
        elif projection_cls == 'mlp':
            dim_projection = hidden_size
            depth = 2
            layers = [nn.Linear(image_dim_out, dim_projection, dtype=dtype, device=device)]
            for _ in range(1, depth):
                layers.extend([nn.GELU(), nn.Linear(dim_projection, dim_projection, dtype=dtype, device=device)])
            self.img_projection = nn.Sequential(*layers)
        else:
            raise NotImplementedError(f'projection_cls = {projection_cls}, not implemented')

        self.vocab_size = config.vocab_size
        self.img_features = None

        if isinstance(config.img_processor, dict):
            self.layer_idx = config.img_processor.get('layer_idx', -2)
            self.type_feature = config.img_processor.get('type_feature', 'patch')
        else:
            self.layer_idx = -2
            self.type_feature = 'patch'

    def get_img_features(self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
        LAYER_IDX = self.layer_idx
        TYPE_FEATURE = self.type_feature

        img_processor_output = self.img_processor(img_embeds, output_hidden_states=True)
        img_feature = img_processor_output.hidden_states[LAYER_IDX]

        if TYPE_FEATURE == 'patch':
            patch_feature = img_feature[:, 1:]
            return patch_feature

        if TYPE_FEATURE == 'cls_patch':
            return img_feature

        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_sizes=None,
        image_mask: torch.Tensor = None,
    ) -> torch.FloatTensor:
        """forward."""
        inputs_embeds = self.wte(input_ids)
        assert self.use_hd_transform
        num_images, num_crops, c, h, w = pixel_values.shape
        assert c == 3 and h == w == 336
        img_features = self.get_img_features(pixel_values.flatten(0, 1)).reshape(num_images, num_crops, -1,
                                                                                 self.image_dim_out)
        image_features_proj = self.hd_feature_transform(img_features, image_sizes)
        # update image feature to inputs_embeds
        inputs_embeds.masked_scatter_(image_mask[..., None], image_features_proj)
        return inputs_embeds

    def hd_feature_transform(self, image_features, image_sizes):
        """
        image_features: (num_images, num_crops+1, 24*24, 1024)
        """
        assert (self.hd_transform_order == 'sub_glb'), f'hd_transform_order `{self.hd_transform_order}` not implemented'
        if isinstance(self.img_projection, nn.Sequential):
            target_device = self.img_projection[0].bias.device
            target_dtype = self.img_projection[0].bias.dtype
        else:  # It's a single nn.Linear layer
            target_device = self.img_projection.bias.device
            target_dtype = self.img_projection.bias.dtype

        global_image_features = image_features[:, 0]  # (num_images, 24*24, 1024)
        # global feature can be viewed as a special HD case with num_crops 1x1
        global_image_features_hd = self.reshape_hd_patches_2x2merge(global_image_features, 1, 1)
        global_image_features_hd_newline = self.add_image_newline(global_image_features_hd)

        all_image_embeddings = []
        # need a for loop to process each image because of different image sizes
        # (patch arrangement is different for each image)
        for i, img_size in enumerate(image_sizes):
            h, w = img_size
            h_crop = h // 336
            w_crop = w // 336
            num_crops = h_crop * w_crop

            # NOTE: real num_crops is padded
            # (num_crops, 24*24, 1024)
            sub_image_features = image_features[i, 1:1 + num_crops]
            sub_image_features_hd = self.reshape_hd_patches_2x2merge(sub_image_features, h_crop, w_crop)
            sub_image_features_hd_newline = self.add_image_newline(sub_image_features_hd)

            # [sub features, separator, global features]
            all_image_embeddings.extend([
                sub_image_features_hd_newline.squeeze(0),  # (h_crop*12*(w_crop*12+1), 4096)
                self.glb_GN.squeeze(0),
                global_image_features_hd_newline[i],
            ])

        image_features_proj = self.img_projection(
            torch.cat(all_image_embeddings, dim=0).to(target_device).to(target_dtype))

        return image_features_proj

    def reshape_hd_patches_2x2merge(self, image_features, h_crop, w_crop):
        """
        image_features: (num_images*num_crops, 24*24, 1024)
        output: (num_images, h_crop*12, w_crop*12, 4096), h_crop*w_crop == num_crops
        """
        N, L, C = image_features.shape
        assert L == 24 * 24 and C == 1024 and N % (h_crop * w_crop) == 0
        num_images = N // (h_crop * w_crop)
        H = int(L**0.5)
        image_features_hd = (
            image_features.reshape(N, H, H, C)  # N, 24, 24, 1024
            .reshape(N, H // 2, 2, H // 2, 2, C)  # N, 12, 2, 12, 2, 1024
            .permute(0, 1, 3, 2, 4, 5)  # N, 12, 12, 2, 2, 1024
            .reshape(N, -1, 4 * C)  # N, 144, 4096
            .reshape(num_images, h_crop, w_crop, H // 2, H // 2, -1)  # n_img, h_crop, w_crop, 12, 12, 4096
            .permute(0, 1, 3, 2, 4, 5)  # n_img, h_crop, 12, w_crop, 12, 4096
            .reshape(num_images, h_crop * H // 2, w_crop * H // 2, 4 * C)  # n_img, h_crop*12, w_crop*12, 4096
        )
        return image_features_hd

    def add_image_newline(self, image_features_hd):
        """
        image_features_hd: (num_images, h_crop*12, w_crop*12, 4096)
        output: (num_images, (h_crop*12) * (w_crop*12+1), 4096)
        """
        num_images, h, w, hid_dim = image_features_hd.shape
        # add the newline token to the HD image feature patches
        newline_embeddings = self.sub_GN.expand(num_images, h, -1, -1)  # (n_img, h, 1, hid_dim)
        image_features_hd_newline = torch.cat([image_features_hd, newline_embeddings],
                                              dim=2).reshape(num_images, -1, hid_dim)
        return image_features_hd_newline


class Phi3VModel(Phi3Model):
    """Phi3v model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__(config=config, dtype=dtype, device=device)

        self.vision_embed_tokens = None
        if isinstance(config.embd_layer, dict):
            # vision embedding layer
            embedding_config = {'embedding_cls': config.embd_layer['embedding_cls'], **config.embd_layer}
            self.vision_embed_tokens = Phi3ImageEmbedding(config,
                                                          wte=self.embed_tokens,
                                                          dtype=dtype,
                                                          device=device,
                                                          **embedding_config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[torch.LongTensor] = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        """Rewrite of LlamaModel.forward."""

        if inputs_embeds is None and pixel_values is not None:
            inputs_embeds = self.vision_embed_tokens(
                input_ids,
                pixel_values,
                image_sizes,
                image_mask,
            )

        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
        )


class Phi3VForCausalLM(Phi3ForCausalLM, DeployModelMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config, ctx_mgr, dtype=dtype, device=device)
        self.config = config
        self.ctx_mgr = ctx_mgr
        # build model
        self.model = Phi3VModel(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

        self.input_processor = Phi3VInputProcessor(config, dtype)

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
        """forward."""
        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            image_mask=image_mask,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        output = super().prepare_inputs_for_generation(past_key_values=past_key_values,
                                                       inputs_embeds=inputs_embeds,
                                                       context=context)

        # vision inputs
        pixel_values = None
        if context.input_multimodals is not None:
            input_mms = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            input_mms = [data for im_data in input_mms for data in im_data]
            if len(input_mms) > 0:
                pixel_values = torch.cat([data.data for data in input_mms])
                image_sizes = torch.cat([data.meta['image_sizes'] for data in input_mms])
                image_token_id = input_mms[0].meta['image_token_id']
                image_mask = output['input_ids'] == image_token_id
                output['pixel_values'] = pixel_values
                output['image_sizes'] = image_sizes
                output['image_mask'] = image_mask

        return output

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""
        import itertools

        vis_prefix = 'vision_embed_tokens.'
        # create two ierators from weights for llm and vlm
        llm_weights, vlm_weights = itertools.tee(weights, 2)
        llm_weights = ((name, tensor) for name, tensor in llm_weights if vis_prefix not in name)
        vlm_weights = ((name, tensor) for name, tensor in vlm_weights if vis_prefix in name)
        super().load_weights(llm_weights)

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in vlm_weights:
            param = params_dict[name]
            load_weight(param, loaded_weight)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class Phi3VInputProcessor(BaseModelInputProcessor):
    """Phi3V input processor."""

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
            pixel_values = input_mm['pixel_values']
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
