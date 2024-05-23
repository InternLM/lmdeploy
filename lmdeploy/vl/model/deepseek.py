# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import disable_logging


def check_deepseek_vl_install():
    """check deepseek_vl install."""
    try:
        import deepseek_vl  # noqa: F401
    except ImportError:
        raise ImportError(
            'To use DeepSeekVLModel, please install deepseek_vl by '
            'pip install git+https://github.com/deepseek-ai/DeepSeek-VL.git'
            ' --no-deps')


class DeepSeekVisionModel(VisonModel):
    """Qwen vision model."""

    def __init__(self, model_path, with_llm: bool = False):
        self.with_llm = with_llm
        self.model_path = model_path
        self.build_model()

    def build_model(self):
        check_deepseek_vl_install()
        # empty init
        from accelerate import init_empty_weights
        from deepseek_vl.models import VLChatProcessor
        with init_empty_weights():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_pretrained(self.model_path)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(model,
                                         dtype=torch.half,
                                         no_split_module_classes=['Block'])
        device_map = infer_auto_device_map(model,
                                           no_split_module_classes=['Block'],
                                           max_memory=max_memory,
                                           dtype=torch.half)
        same_device_keys = [
            ('vision_model.vision_tower_high.vision_tower.pos_embed',
             'vision_model.vision_tower_high.vision_tower.patch_embed'),
            ('vision_model.vision_tower_low.vision_tower.pos_embed',
             'vision_model.vision_tower_low.vision_tower.patch_embed')
        ]
        for (a, b) in same_device_keys:
            if a in device_map and b in device_map:
                device_map[b] = device_map[a]
        downsamples = []
        ka = 'vision_model.vision_tower_high.vision_tower.downsamples'
        kb = 'vision_model.vision_tower_high.vision_tower.hd_alpha_downsamples'  # noqa: E501
        for k in device_map:
            if k.startswith(ka):
                downsamples.append(k)
        if len(downsamples) == 1:
            device_map[ka] = device_map[kb]
        elif len(downsamples) > 1:
            numbers = [int(x[len(ka) + 1:]) for x in downsamples]
            device_map[f'{ka}.{numbers[-1]}'] = device_map[kb]

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(model=model,
                                         checkpoint=self.model_path,
                                         device_map=device_map,
                                         dtype=torch.half)

        self.vision_model = model.vision_model.eval()
        self.aligner = model.aligner.eval()
        self.image_processor = VLChatProcessor.from_pretrained(
            self.model_path).image_processor

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        pixel_values = self.image_processor(outputs,
                                            return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(dtype=torch.float16)
        # [b x n_images, T2, D]
        images_embeds = self.aligner(self.vision_model(pixel_values))

        outputs = torch.split(images_embeds, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
