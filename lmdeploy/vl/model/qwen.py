# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import load_model_from_weight_files


class QwenVisionModel(VisonModel):
    """Qwen vision model."""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = AutoConfig.from_pretrained(self.model_path,
                                                trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            del model.lm_head
            for key in ['wte', 'h', 'ln_f']:
                setattr(model.transformer, key, None)

        model.to_empty(device='cpu')
        load_model_from_weight_files(model, self.model_path)

        self.model = model.transformer.visual
        self.model.to(self.device).eval().half()

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [self.model.image_transform(x) for x in outputs]
        outputs = torch.stack(outputs, dim=0)
        outputs = self.model(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
