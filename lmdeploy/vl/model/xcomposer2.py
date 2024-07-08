# Copyright (c) OpenMMLab. All rights reserved.

import warnings
from contextlib import contextmanager
from typing import List

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import (add_device_hook, add_sys_path,
                                     disable_logging, rewrite_ctx)


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


@VISION_MODELS.register_module()
class Xcomposer2VisionModel(VisonModel):
    """InternLM-Xcomposer2 vision model."""

    @classmethod
    def match(cls, config: AutoConfig):
        """check whether the config match the model."""
        target = 'InternLMXComposer2ForCausalLM'
        if config.architectures[0] == target:
            return True
        for _, v in getattr(config, 'auto_map', {}).items():
            if target in v:
                return True
        return False

    def build_model(self):
        from accelerate import init_empty_weights
        with init_empty_weights(), warnings.catch_warnings(), init_empty_vit():
            warnings.simplefilter('ignore')
            config = self.hf_config
            model = AutoModelForCausalLM.from_config(config,
                                                     trust_remote_code=True)
            model.vit.load_model()
            model.vit.resize_pos()
            model.vit.vision_tower.vision_model.post_layernorm.to_empty(
                device='cpu').half()
            if not self.with_llm:
                del model.model
                del model.output
            else:
                self.vl_model = model

        # additional components.
        with add_sys_path(self.model_path):
            try:
                # internlm-xcomposer2-4khd-7b
                from ixc_utils import HD_transform
                self.HD_transform = HD_transform
                self._forward_func = self._forward_4khd_7b
            except:  # noqa
                # internlm-xcomposer2-7b
                self._forward_func = self._forward_7b

        from accelerate.utils import get_balanced_memory, infer_auto_device_map
        max_memory = get_balanced_memory(
            model,
            max_memory=self.max_memory,
            dtype=torch.half,
            no_split_module_classes=['CLIPEncoderLayer'])
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=['CLIPEncoderLayer'],
            max_memory=max_memory,
            dtype=torch.half)
        # make all tensor on same device for postprocess
        if 'plora_glb_GN' in device_map:
            device_map['plora_sub_GN'] = device_map['plora_glb_GN']

        from accelerate import load_checkpoint_and_dispatch
        with disable_logging():
            load_checkpoint_and_dispatch(
                model=model,
                checkpoint=self.model_path,
                device_map=device_map if not self.with_llm else {'': 'cpu'},
                no_split_module_classes=['CLIPEncoderLayer'],
                dtype=torch.half)

        if 'plora_glb_GN' in device_map:
            add_device_hook(
                model.vit.vision_tower.vision_model.encoder.layers[-1],
                device_map['plora_glb_GN'], lambda x:
                (x[0].to(device=device_map['plora_glb_GN']), ))

        self.model = model.eval()

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
        embeds, split = self.model.vit(outputs, self.model.plora_glb_GN,
                                       self.model.plora_sub_GN)
        embeds = self.model.vision_proj(embeds)
        embeds = torch.split(embeds, split, dim=1)
        embeds = [x.squeeze() for x in embeds]
        return embeds

    @torch.no_grad()
    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        return self._forward_func(images)
