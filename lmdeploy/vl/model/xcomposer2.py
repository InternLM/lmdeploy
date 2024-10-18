# Copyright (c) OpenMMLab. All rights reserved.

import enum
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Any, List, Tuple

import torch
from PIL.Image import Image
from transformers import AutoConfig, AutoModelForCausalLM

from lmdeploy.utils import get_logger
from lmdeploy.vl.model.base import VISION_MODELS, VisonModel
from lmdeploy.vl.model.utils import (add_device_hook, disable_logging,
                                     rewrite_ctx)

logger = get_logger('lmdeploy')


class ModelType(enum.Enum):
    """Request type."""
    XCOMPOSER2 = enum.auto()
    XCOMPOSER2_4KHD = enum.auto()
    XCOMPOSER2D5 = enum.auto()


def get_xcomposer_type(model_path: str) -> Tuple[ModelType, Any]:
    """get xcomposer type."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    match_modules = {
        'ixc_utils.Image_transform': ModelType.XCOMPOSER2D5,
        'ixc_utils.HD_transform': ModelType.XCOMPOSER2_4KHD
    }
    for key, value in match_modules.items():
        try:
            module = get_class_from_dynamic_module(key, model_path)
            return value, module
        except Exception:
            pass
    return ModelType.XCOMPOSER2, None


def _CLIPVisionModel_from_pretrained(vision_tower_name):
    from transformers import CLIPVisionConfig, CLIPVisionModel
    config = CLIPVisionConfig.from_pretrained(vision_tower_name)
    model = CLIPVisionModel._from_config(config)
    return model


@contextmanager
def init_empty_vit(model_path):
    """skip download vision model."""
    origin_func_path = [
        'transformers.CLIPVisionModel.from_pretrained',
    ]
    rewrite_func = [
        _CLIPVisionModel_from_pretrained,
    ]

    model_type, _ = get_xcomposer_type(model_path)
    if model_type == ModelType.XCOMPOSER2D5:
        from transformers.dynamic_module_utils import \
            get_class_from_dynamic_module
        from transformers.utils import TRANSFORMERS_DYNAMIC_MODULE_NAME
        _ = get_class_from_dynamic_module(
            'modeling_internlm_xcomposer2.get_font', model_path)
        folder = model_path.rstrip(os.sep).split(os.sep)[-1]
        module_path = '.'.join([
            TRANSFORMERS_DYNAMIC_MODULE_NAME, folder,
            'modeling_internlm_xcomposer2'
        ])
        origin_get_font_func = getattr(sys.modules[module_path], 'get_font')
        origin_func_path.append(origin_get_font_func)
        rewrite_func.append(lambda: None)

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
        with init_empty_weights(), warnings.catch_warnings(), \
                init_empty_vit(self.model_path):
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
        model_type, module = get_xcomposer_type(self.model_path)
        logger.info(f'matching type of {model_type}')
        if model_type == ModelType.XCOMPOSER2D5:
            self.HD_transform = module
            self._forward_func = self._forward_2d5
        elif model_type == ModelType.XCOMPOSER2_4KHD:
            self.HD_transform = module
            self._forward_func = self._forward_4khd_7b
        else:
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

    def _forward_2d5(self, images: List[Image]) -> List[torch.Tensor]:
        """internlm-xcomposer2d5-7b vit forward."""
        outputs = [x.convert('RGB') for x in images]
        hd_num = 6 if len(images) > 1 else 24
        outputs = [self.HD_transform(x, hd_num=hd_num) for x in outputs]
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
