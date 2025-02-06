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
from .utils.model import DeployModelMixin

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


class Phi3ImageEmbedding(nn.Module):
    """image embedding."""

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

        target_device = pixel_values.device
        target_dtype = pixel_values.dtype

        img_embeds = pixel_values
        img_sizes = image_sizes
        img_sizes = img_sizes.cpu()

        if self.use_hd_transform and img_sizes is not None and len(img_sizes):
            assert img_embeds.ndim == 5, f'img_embeds size: {img_embeds.size()}, expect 5D tensor for hd transform'  # noqa E501
            # img_embeds: (num_images, max_num_crops, 3, H, W)
            # img_sizes: (num_images, 2).view(1, -1)

            bs = img_embeds.shape[0]
            # Nx(HW)xC
            img_features = self.get_img_features(img_embeds.flatten(0, 1))
            base_feat_height = base_feat_width = int(img_features.shape[1]**0.5)

            assert base_feat_height == 24 and base_feat_width == 24, f'base_feat_height: {base_feat_height}, base_feat_width: {base_feat_width}, expect 24x24 features for hd transform'  # noqa E501

            # bs x max_num_crops x (24x24) x C
            img_features = img_features.view(bs, -1, base_feat_height * base_feat_width, self.image_dim_out)
            C = self.image_dim_out
            H = base_feat_height

            output_imgs = []
            output_len = []
            # training is tensor, inference is list
            if isinstance(img_sizes, torch.Tensor):
                img_sizes = img_sizes.view(-1, 2)
            for _bs in range(bs):
                h, w = img_sizes[_bs]
                h = h // 336
                w = w // 336
                B_ = h * w

                # 1 x (24x24) x 1024
                global_img_feature = img_features[_bs, :1]

                # 1 x 12 x 12 x 4096
                glb_img = global_img_feature.reshape(1, H // 2, 2, H // 2, 2,
                                                     C).permute(0, 1, 3, 2, 4, 5).reshape(1, H // 2, H // 2, 4 * C)
                temp_glb_GN = self.sub_GN.repeat(1, H // 2, 1, 1)

                # 1 x 156 x 4096
                glb_img = torch.cat([glb_img, temp_glb_GN], dim=2).reshape(1, -1, 4 * C)

                # (max_num_crops-1) x (12x12) x C
                sub_img = img_features[_bs, 1:]
                # 16x574x1024
                # get rid of padding sub_img
                sub_img = sub_img[:B_]

                # (num_crops, 12, 2, 12, 2, 1024)
                # ->(num_crops, 12, 12, 2, 2, 1024)
                # -> (num_crops, 12*12, 4*1024)
                sub_img = (sub_img.reshape(B_, H // 2, 2, H // 2, 2, C).permute(0, 1, 3, 2, 4, 5))
                sub_img = sub_img.reshape(1, h, w, 12, 12, -1).permute(0, 1, 3, 2, 4,
                                                                       5).reshape(1, h * 12, w * 12, 4 * C)
                temp_sub_GN = self.sub_GN.repeat(1, h * 12, 1, 1)
                sub_img = torch.cat([sub_img, temp_sub_GN], dim=2).reshape(1, -1, 4 * C)
                # (1, num_img_tokens, 1024*4)

                # glb + sub
                if self.hd_transform_order == 'glb_sub':
                    output_imgs.append(torch.cat([glb_img, self.glb_GN, sub_img], dim=1))
                elif self.hd_transform_order == 'sub_glb':
                    output_imgs.append(torch.cat([sub_img, self.glb_GN, glb_img], dim=1))
                else:
                    raise NotImplementedError(f'hd_transform_order = {self.hd_transform_order}')  # noqa E501

                temp_len = int((h * w + 1) * 144 + 1 + (h + 1) * 12)
                assert temp_len == output_imgs[-1].shape[
                    1], f'temp_len: {temp_len}, output_imgs[-1].shape[1]: {output_imgs[-1].shape[1]}'  # noqa E501
                output_len.append(temp_len)

            img_set_tensor = []
            for _output_img in output_imgs:
                img_feature_proj = self.img_projection(_output_img.to(target_device).to(target_dtype))
                img_feature_proj = img_feature_proj.flatten(0, 1)
                img_set_tensor.append(img_feature_proj)
            img_set_tensor = torch.cat(img_set_tensor)[None]
        elif img_embeds.ndim == 4:
            tt = (self.get_img_features(img_embeds).to(target_device).to(target_dtype).reshape(-1, self.image_dim_out))
            img_set_tensor = self.img_projection(tt)  # adapted visual features.
        elif img_embeds.ndim == 3:
            tt = (img_embeds.to(target_device).to(target_dtype).view(-1, self.image_dim_out))
            img_set_tensor = self.img_projection(tt)  # adapted visual features.
        else:
            raise NotImplementedError

        hidden_states = self.wte(input_ids)

        hidden_states.masked_scatter_(image_mask[..., None], img_set_tensor)

        return hidden_states


class Phi3VModel(Phi3Model):
    """phi3v model."""

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
        """prepare input."""
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
        """load weights."""
        super().load_weights(weights)

        vis_prefix = 'vision_embed_tokens.'
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if not (vis_prefix in name):
                continue
            param = params_dict[name]
            load_weight(param, loaded_weight)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """get input processor."""
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
        """prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            image_sizes = input_mm['image_sizes']
            offset = input_mm['offset']
            image_token_id = input_mm.get('image_token_id', 0)
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
