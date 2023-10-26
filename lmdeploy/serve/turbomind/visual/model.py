# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from abc import abstractmethod
from typing import List

import torch
from mmengine import Registry
from transformers import AutoConfig

VISUAL_MODELS = Registry('visual model',
                         locations=['lmdeploy.serve.turbomind.visual.model'])


class BaseModel:
    """Base model."""

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def encode(self, image_paths: List[str]):
        pass


@VISUAL_MODELS.register_module(name='qwen-7b')
class QwenVL(BaseModel):

    def __init__(self, model_folder, **kwargs):
        vison_model_folder = os.path.join(model_folder, 'triton_models',
                                          'visual')
        cfg_path = os.path.join(model_folder, 'triton_models', 'tokenizer')
        cfg = AutoConfig.from_pretrained(cfg_path, trust_remote_code=True)
        sys.path.append(vison_model_folder)
        from visual import VisionTransformer
        model = VisionTransformer(**cfg.visual)
        sys.path.pop()
        model_bin = os.path.join(vison_model_folder, 'model.bin')
        model.load_state_dict(torch.load(model_bin))
        self.model = model.cuda().eval()

    def encode(self, image_paths: List[str]):
        return self.model.encode(image_paths)
