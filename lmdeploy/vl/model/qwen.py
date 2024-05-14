# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.messages import VisonConfig
from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import disable_logging


class QwenVisionModel(VisonModel):
    """Qwen vision model."""

    def __init__(self, model_path: str, vision_config: VisonConfig = None):
        self.model_path = model_path
        self.vision_config = (vision_config
                              if vision_config is not None else VisonConfig())
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

        device_map = self.vision_config.device_map
        if isinstance(device_map, str):
            max_memory = None
            from accelerate.utils import (get_balanced_memory,
                                          infer_auto_device_map)
            if device_map != 'sequential':
                max_memory = get_balanced_memory(
                    model,
                    dtype=torch.half,
                    no_split_module_classes=['VisualAttentionBlock'])
            device_map = infer_auto_device_map(
                model,
                no_split_module_classes=['VisualAttentionBlock'],
                max_memory=max_memory,
                dtype=torch.half)
            same_device_keys = [('transformer.visual.conv1',
                                 'transformer.visual.positional_embedding'),
                                ('transformer.visual.ln_post',
                                 'transformer.visual.proj')]
            for (a, b) in same_device_keys:
                if a in device_map and b in device_map:
                    device_map[b] = device_map[a]

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=device_map,
                no_split_module_classes=['VisualAttentionBlock'],
                dtype=torch.half)

        self.model = model.transformer.visual

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
