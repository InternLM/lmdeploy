# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoModelForCausalLM

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import (buffers_aware_empty,
                                     load_model_from_weight_files)


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

    def __init__(self, model_path, device='cuda:0', with_llm: bool = False):
        self.with_llm = with_llm
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        check_deepseek_vl_install()
        # empty init
        from accelerate import init_empty_weights
        from deepseek_vl.models import VLChatProcessor
        with init_empty_weights():
            warnings.simplefilter('ignore')
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True)
            if not self.with_llm:
                del model.language_model
            else:
                self.vl_model = model
        # load weight
        buffers_aware_empty(model, 'cpu')
        load_model_from_weight_files(model, self.model_path)
        self.vision_model = model.vision_model
        self.aligner = model.aligner
        self.vision_model.eval().half().to(self.device)
        self.aligner.eval().half().to(self.device)
        self.image_processor = VLChatProcessor.from_pretrained(
            self.model_path).image_processor

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        outputs = [x.convert('RGB') for x in images]
        pixel_values = self.image_processor(outputs,
                                            return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        # [b x n_images, T2, D]
        images_embeds = self.aligner(self.vision_model(pixel_values))

        outputs = torch.split(images_embeds, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs
