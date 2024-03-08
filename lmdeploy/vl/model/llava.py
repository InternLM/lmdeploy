# Copyright (c) OpenMMLab. All rights reserved.
# Modified from
# https://github.com/haotian-liu/LLaVA.git

import os
from typing import List

import torch
import torch.nn as nn
from PIL.Image import Image

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.utils import load_model_from_weight_files

logger = get_logger('lmdeploy')


class LlavaVLModelWrapper(nn.Module):
    """Llava visual model."""

    def __init__(self, model_path, device='cuda'):
        super().__init__()
        self.model_path = model_path
        self.device = device

        # currently, only support llava llama
        from llava.model.language_model.llava_llama import LlavaConfig
        self.config = LlavaConfig.from_pretrained(model_path)
        assert self.config.model_type in ['llava', 'llava_llama'], \
            'currently, only support llava llama'
        self.build_model()

    def _build_vision_tower(self):
        from llava.model.multimodal_encoder.builder import build_vision_tower
        return build_vision_tower(self.config)

    def _build_vision_projector(self):
        from llava.model.multimodal_projector.builder import \
            build_vision_projector
        return build_vision_projector(self.config)

    def build_model(self):
        cfg = self.config
        vision_tower = getattr(cfg, 'mm_vision_tower',
                               getattr(cfg, 'vision_tower', None))
        if os.path.exists(os.path.join(self.model_path, vision_tower)):
            vision_tower = os.path.join(self.model_path, vision_tower)
            cfg.mm_vision_tower = vision_tower

        with torch.device(self.device):
            self.vision_tower = self._build_vision_tower()
            self.mm_projector = self._build_vision_projector()

            if 'unpad' in getattr(cfg, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.randn(cfg.hidden_size,
                                dtype=self.vision_tower.dtype))

    def preprocess(self, images: List[Image]):
        """preprocess."""
        # TODO: gpu processor
        from llava.mm_utils import process_images
        images = [x.convert('RGB') for x in images]
        image_processor = self.vision_tower.image_processor
        outputs = process_images(images, image_processor, self.config)
        return outputs

    def _forward(self, inputs: torch.Tensor):
        outputs = inputs.to(device=self.vision_tower.device,
                            dtype=self.vision_tower.dtype)
        if outputs.ndim == 5:
            outputs = outputs.flatten(0, 1)
        outputs = self.vision_tower.vision_tower(outputs,
                                                 output_hidden_states=True)
        outputs = self.vision_tower.feature_select(outputs)
        outputs = self.mm_projector(outputs)
        return outputs

    @torch.no_grad()
    def forward(self, images: List[Image]):
        """forward."""
        outputs = self.preprocess(images)
        if isinstance(outputs, list):
            new_outputs = []
            for output in outputs:
                new_outputs.append(self._forward(output))
        else:
            new_outputs = self._forward(outputs)
            new_outputs = torch.split(new_outputs,
                                      new_outputs.shape[0] // len(images),
                                      dim=0)

        return self.postprocess(new_outputs, images)

    def postprocess(self, image_features: List[torch.Tensor],
                    images: List[Image]):
        """postprocess."""
        from llava.model.llava_arch import (get_anyres_image_grid_shape,
                                            unpad_image)
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type',
                                      'flat')
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio',
                                     'square')
        if mm_patch_merge_type == 'flat':
            outputs = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith('spatial'):
            outputs = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.vision_tower.num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    assert image_aspect_ratio == 'anyres'
                    image_size = images[image_idx].size
                    num_patch_width, num_patch_height = \
                        get_anyres_image_grid_shape(
                            image_size,
                            self.config.image_grid_pinpoints,
                            self.vision_tower.config.image_size)
                    image_feature = image_feature.view(num_patch_height,
                                                       num_patch_width, height,
                                                       width, -1)
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1,
                                                              3).contiguous()
                        image_feature = image_feature.flatten(1,
                                                              2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_size)
                        image_feature = torch.cat(
                            (image_feature,
                             self.image_newline[:, None, None].expand(
                                 *image_feature.shape[:-1], 1).to(
                                     image_feature.device)),
                            dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(
                            0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3,
                                                              4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat(
                        (base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat(
                            (image_feature, self.image_newline[None].to(
                                image_feature.device)),
                            dim=0)
                outputs.append(image_feature)

        else:
            raise ValueError('Unexpected mm_patch_merge_type: '
                             f'{self.config.mm_patch_merge_type}')

        return outputs


class LlavaVLModel(nn.Module):

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = LlavaVLModelWrapper(model_path)
        self.model.eval().half()
        load_model_from_weight_files(self, self.model_path)

    def forward(self, images: List[Image]):
        """forward."""
        return self.model.forward(images)
