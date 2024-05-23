# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
from PIL.Image import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import load_model_from_weight_files


class CogVLMVisionModel(VisonModel):
    """CogVLM vision model."""

    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.dtype = torch.float16
        self.hf_config = AutoConfig.from_pretrained(model_path,
                                                    trust_remote_code=True)
        self.build_model()
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (self.hf_config.vision_config['image_size'], ) * 2,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(self.hf_config,
                                                     trust_remote_code=True)
            del model.lm_head
            for key in ['layers', 'norm', 'embed_tokens']:
                setattr(model.model, key, None)

        model.to_empty(device='cpu')
        load_model_from_weight_files(model, self.model_path)

        self.model = model.model.vision
        self.model.to(self.device).eval().to(self.dtype)

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [self.image_transform(x) for x in outputs]
        outputs = torch.stack(outputs, dim=0).to(self.device).to(self.dtype)
        outputs = self.model(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
