# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoConfig

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


@VISION_MODELS.register_module()
class GLM4VisionModel(VisonModel):
    """glm-4v-9b vision model."""

    _arch = 'ChatGLMModel'

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        arch = config.architectures[0]
        if arch == cls._arch and hasattr(config, 'vision_config'):
            return True
        return False

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import infer_auto_device_map
        from torchvision import transforms

        with init_empty_weights(), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_config(self.hf_config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                del model.transformer.embedding
                del model.transformer.rotary_pos_emb
                del model.transformer.encoder
                del model.transformer.output_layer
            else:
                self.vl_model = model

        no_split_module_classes = ['TransformerLayer']

        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=no_split_module_classes,
            max_memory=self.max_memory,
            dtype=torch.half)

        same_device_keys = [
            ('transformer.vision.linear_proj', 'transformer.vision.boi',
             'transformer.vision.eoi')
        ]
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

        model.eval()
        self.model = model
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (self.hf_config.vision_config['image_size'], ) * 2,
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [self.image_transform(x) for x in outputs]
        outputs = torch.stack(outputs, dim=0).to(device='cuda:0',
                                                 dtype=torch.half)
        outputs = self.model.transformer.vision(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
