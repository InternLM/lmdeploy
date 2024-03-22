# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
from accelerate import init_empty_weights
from PIL.Image import Image
from transformers import AutoConfig

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import load_model_from_weight_files


def check_deepseek_vl_install():
    """check deepseek_vl install."""
    try:
        import deepseek_vl  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use DeepSeekVLModel, please install deepseek_vl by '
            'pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git')


class DeepSeekVisionModel(VisonModel):
    """Qwen vision model."""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        check_deepseek_vl_install()
        from deepseek_vl.models.modeling_vlm import (MultiModalityConfig,
                                                     model_name_to_cls)
        with init_empty_weights():
            config: MultiModalityConfig = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True)
            vision_config = config.vision_config
            vision_cls = model_name_to_cls(vision_config.cls)
            self.vision_model = vision_cls(**vision_config.params)

            aligner_config = config.aligner_config
            aligner_cls = model_name_to_cls(aligner_config.cls)
            self.aligner = aligner_cls(aligner_config.params)

        with torch.device(self.device):
            self.vision_model.to_empty(device=self.device)
            self.aligner.to_empty(device=self.device)
            load_model_from_weight_files(self.vision_model, self.model_path)
            load_model_from_weight_files(self.aligner, self.model_path)

        self.aligner.eval().half()
        self.aligner.eval().half()

        from deepseek_vl.models import VLChatProcessor
        self.image_processor = VLChatProcessor.from_pretrained(
            self.model_path).image_processor

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        pixel_values = self.image_processor(outputs,
                                            return_tensors='pt').pixel_values
        from einops import rearrange
        images = rearrange(pixel_values, 'b n c h w -> (b n) c h w')
        # [b x n_images, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        outputs = torch.split(images_embeds, 0, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
