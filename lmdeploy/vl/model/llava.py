# Copyright (c) OpenMMLab. All rights reserved.
# Modified from
# https://github.com/haotian-liu/LLaVA.git
import warnings
from contextlib import contextmanager
from typing import List, Union

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import (buffers_aware_empty,
                                     load_model_from_weight_files, rewrite_ctx)

logger = get_logger('lmdeploy')


def check_llava_install():
    """check llava install."""
    try:
        import llava  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use LlavaVLModel, please install llava by '
            'pip install git+https://github.com/haotian-liu/LLaVA.git --no-deps'  # noqa: E501
        )


def _clip_vision_tower_load_model(self, **kwargs):
    logger.info(f'CLIPVisionTower.load_model: {self.vision_tower_name}')
    from transformers import (CLIPImageProcessor, CLIPVisionConfig,
                              CLIPVisionModel)
    self.image_processor = CLIPImageProcessor.from_pretrained(
        self.vision_tower_name)
    config = CLIPVisionConfig.from_pretrained(self.vision_tower_name,
                                              trust_remote_code=True)
    self.vision_tower = CLIPVisionModel._from_config(config=config)
    self.vision_tower.requires_grad_(False)
    self.is_loaded = True


@contextmanager
def init_llava_vision_tower(config):
    """skip download vision model if possible."""
    if getattr(config, 'unfreeze_mm_vision_tower', False):
        origin_func_path = [
            'llava.model.multimodal_encoder.clip_encoder.CLIPVisionTower.load_model'  # noqa: E501
        ]
        rewrite_func = [_clip_vision_tower_load_model]
        with rewrite_ctx(origin_func_path, rewrite_func):
            yield
    else:
        yield


class LlavaVisionModel(VisonModel):
    """Llava visual model."""

    def __init__(self, model_path, device='cuda:0', with_llm: bool = False):
        self.with_llm = with_llm
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        """build model & load weights."""
        # check llava install
        check_llava_install()

        # currently, only support llava llama
        from llava.model.language_model.llava_llama import (  # noqa
            LlavaConfig, LlavaLlamaForCausalLM)
        self.config = LlavaConfig.from_pretrained(self.model_path)
        assert self.config.model_type in ['llava', 'llava_llama'], \
            'currently, only support llava llama'
        from accelerate import init_empty_weights

        # init empty model, skip layer initialization
        with init_empty_weights(), warnings.catch_warnings(), \
                init_llava_vision_tower(self.config):
            warnings.simplefilter('ignore')
            self.config.quantization_config = {
            }  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(self.config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                del model.lm_head
                del model.model.embed_tokens
                del model.model.layers
                del model.model.norm
            else:
                self.vl_model = model

        # move model to cpu
        with torch.device('cpu'):
            buffers_aware_empty(model, 'cpu')
        # init empty vision_tower, the embedding layer in CLIPVisionModel
        # can't init right under init_empty_weights
        with init_llava_vision_tower(self.config):
            vision_tower = model.get_vision_tower()
            vision_tower.is_loaded = False
            vision_tower.load_model()
        # load weight
        load_model_from_weight_files(model, self.model_path)
        model.to(self.device).eval()

        self.model = model.model
        self.vision_tower = model.model.vision_tower.half()
        self.mm_projector = model.model.mm_projector.half()

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """encode images."""
        image_features = self.vision_tower(images)
        image_features = self.mm_projector(image_features)
        return image_features

    def preprocess(
            self,
            images: List[Image]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """preprocess."""
        # TODO: gpu processor
        from llava.mm_utils import process_images
        images = [x.convert('RGB') for x in images]
        image_processor = self.vision_tower.image_processor
        outputs = process_images(images, image_processor, self.config)
        return outputs

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        from llava.model.llava_arch import (get_anyres_image_grid_shape,
                                            unpad_image)
        image_sizes = [x.size for x in images]
        images = self.preprocess(images)
        if isinstance(images, list):
            images = [x.to(self.device, dtype=torch.float16) for x in images]
        else:
            images = images.to(self.device, dtype=torch.float16)
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type',
                                          'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio',
                                         'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.vision_tower.num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = \
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.vision_tower.config.image_size)
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height,
                                width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1,
                                                                  2).flatten(
                                                                      2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(
                                    *image_feature.shape[:-1], 1).to(
                                        image_feature.device)),
                                                      dim=-1)
                            image_feature = image_feature.flatten(1,
                                                                  2).transpose(
                                                                      0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (image_feature,
                                 self.model.image_newline[None].to(
                                     image_feature.device)),
                                dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError('Unexpected mm_patch_merge_type: '
                                 f'{self.config.mm_patch_merge_type}')
        else:
            image_features = self.encode_images(images)
            image_features = [x for x in image_features]
        return image_features
