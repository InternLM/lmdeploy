# Copyright (c) OpenMMLab. All rights reserved.

from typing import List

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import disable_logging


@VISION_MODELS.register_module()
class QwenVisionModel(VisonModel):
    """Qwen vision model."""

    _arch = 'QWenLMHeadModel'

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights():
            config = self.hf_config
            config.quantization_config = {}  # disable vision part quantization
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            if not self.with_llm:
                del model.lm_head
                for key in ['wte', 'h', 'ln_f']:
                    setattr(model.transformer, key, None)
            else:
                self.vl_model = model

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(
            model,
            max_memory=self.max_memory,
            dtype=torch.half,
            no_split_module_classes=['VisualAttentionBlock', 'Resampler'])
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=['VisualAttentionBlock', 'Resampler'],
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
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['VisualAttentionBlock'],
                dtype=torch.half)

        self.model = model.transformer.visual.eval()

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
