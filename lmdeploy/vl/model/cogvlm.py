# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List

import torch
from PIL.Image import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import add_device_hook, disable_logging


class CogVLMVisionModel(VisonModel):
    """CogVLM vision model."""

    def __init__(self, model_path: str, with_llm: bool = False):
        self.with_llm = with_llm
        self.model_path = model_path
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
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        with init_empty_weights(), warnings.catch_warnings():
            model = AutoModelForCausalLM.from_config(self.hf_config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                del model.lm_head
                for key in ['layers', 'norm', 'embed_tokens']:
                    setattr(model.model, key, None)

        no_split_module_classes = ['TransformerLayer']
        max_memory = get_balanced_memory(
            model,
            dtype=torch.half,
            no_split_module_classes=no_split_module_classes)
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes,
            max_memory=max_memory,
            dtype=torch.half)
        same_device_keys = [('model.vision.linear_proj', 'model.vision.boi',
                             'model.vision.eoi')]
        for keys in same_device_keys:
            keys = [k for k in keys if k in device_map]
            if len(keys) <= 1:
                continue
            for k in keys[1:]:
                device_map[k] = device_map[keys[0]]

        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=no_split_module_classes,
                dtype=torch.half)
        self.model = model.model.vision
        self.model.eval()
        add_device_hook(self.model, next(iter(device_map.values())))

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [self.image_transform(x) for x in outputs]
        outputs = torch.stack(outputs, dim=0).to(device='cuda:0',
                                                 dtype=torch.half)
        outputs = self.model(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
