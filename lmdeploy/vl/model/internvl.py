# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModel, CLIPImageProcessor

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import load_model_from_weight_files


class InternVLVisionModel(VisonModel):
    """InternVL vision model."""

    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        """Load model."""
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = AutoConfig.from_pretrained(self.model_path,
                                                trust_remote_code=True)
            model = AutoModel.from_config(config, trust_remote_code=True)
            del model.language_model
            model.half()

        model.to_empty(device='cpu')
        load_model_from_weight_files(model, self.model_path)
        self.model = model
        self.model.to(self.device).eval()
        self.image_processor = CLIPImageProcessor.from_pretrained(
            self.model_path)

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        pixel_values = self.image_processor(images=images,
                                            return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        outputs = self.model.extract_feature(pixel_values)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
