# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from contextlib import contextmanager
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.vl.model.base import VisonModel
from lmdeploy.vl.model.utils import add_sys_path, rewrite_ctx


def _CLIPVisionModel_from_pretrained(vision_tower_name):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    config = CLIPVisionConfig.from_pretrained(vision_tower_name)
    model = CLIPVisionModel._from_config(config)
    return model


@contextmanager
def init_empty_vit():
    """skip download vision model."""
    origin_func_path = [
        'transformers.CLIPVisionModel.from_pretrained',
    ]
    rewrite_func = [
        _CLIPVisionModel_from_pretrained,
    ]
    with rewrite_ctx(origin_func_path, rewrite_func):
        yield


class Xcomposer2VisionModel(VisonModel):
    """InternLM-Xcomposer2 vision model."""

    def __init__(self, model_path, device='cuda:0'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        from accelerate import init_empty_weights, load_checkpoint_in_model
        with init_empty_weights(), warnings.catch_warnings(), init_empty_vit():
            warnings.simplefilter('ignore')
            config = AutoConfig.from_pretrained(self.model_path,
                                                trust_remote_code=True)
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            model = model.half()
            del model.model
            del model.output

        model.to_empty(device='cpu')

        # additional components.
        with add_sys_path(self.model_path), init_empty_vit():
            # CLIPVisionModel embedding layer won't init right
            # with init_empty_weights
            model.vit.load_model()
            model.vit.resize_pos()
            if config.architectures[0] == 'InternLM2ForCausalLM':
                # internlm-xcomposer2-4khd-7b
                from ixc_utils import HD_transform
                self.HD_transform = HD_transform
                self._forward_func = self._forward_4khd_7b
            else:
                # internlm-xcomposer2-7b
                self._forward_func = self._forward_7b

        load_checkpoint_in_model(model, self.model_path)
        self.model = model.to(self.device).eval()

    def _forward_7b(self, images: List[Image]) -> List[torch.Tensor]:
        """internlm-xcomposer2-7b vit forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [
            self.model.vis_processor(x).unsqueeze(0).half() for x in outputs
        ]
        outputs = torch.cat(outputs, dim=0)
        outputs = self.model.vit(outputs)
        outputs = self.model.vision_proj(outputs)
        outputs = torch.split(outputs, 1, dim=0)
        outputs = [x.squeeze() for x in outputs]
        return outputs

    def _forward_4khd_7b(self, images: List[Image]) -> List[torch.Tensor]:
        """internlm-xcomposer2-4khd-7b vit forward."""
        outputs = [x.convert('RGB') for x in images]
        outputs = [self.HD_transform(x, hd_num=25) for x in outputs]
        outputs = [
            self.model.vis_processor(x).unsqueeze(0).to(dtype=torch.half)
            for x in outputs
        ]
        # Too much memory used here. Currently, since we cannot set batch size,
        # we inference images one by one.
        embeds = []
        for x in outputs:
            embed, _ = self.model.vit([x], self.model.plora_glb_GN,
                                      self.model.plora_sub_GN)
            embed = self.model.vision_proj(embed).squeeze()
            embeds.append(embed)
        # embeds, split = self.model.vit(outputs, self.model.plora_glb_GN,
        #                                self.model.plora_sub_GN)
        # embeds = self.model.vision_proj(embeds)
        # embeds = torch.split(embeds, split, dim=1)
        # embeds = [x.squeeze() for x in embeds]
        return embeds

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        return self._forward_func(images)
