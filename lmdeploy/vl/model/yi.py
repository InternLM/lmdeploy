# Copyright (c) OpenMMLab. All rights reserved.
import os
from contextlib import contextmanager
from typing import MutableSequence

import torch.nn as nn

from lmdeploy.vl.model.llava import LlavaVisionModel, check_llava_install

_model_path = None


def _build_vision_projector(config, delay_load=False, **kwargs):
    """build yi projector."""
    # copy from https://github.com/01-ai/Yi/blob/main/VL/llava/model/multimodal_projector/builder.py # noqa: E501
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    import re
    use_norm = False
    if '_Norm' in projector_type:
        use_norm = True
        projector_type = projector_type.replace('_Norm', '')
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        if use_norm:
            modules = [
                nn.Linear(config.mm_hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
            ]
        else:
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            if use_norm:
                modules.append(
                    nn.Linear(config.hidden_size, config.hidden_size))
                modules.append(nn.LayerNorm(config.hidden_size))
            else:
                modules.append(
                    nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return nn.Identity()

    raise ValueError(f'Unknown projector type: {projector_type}')


def _build_vision_tower(vision_tower_cfg, **kwargs):
    """build yi vision tower."""
    cfg = vision_tower_cfg
    vision_tower = getattr(cfg, 'mm_vision_tower',
                           getattr(cfg, 'vision_tower', None))
    if os.path.exists(os.path.join(_model_path, vision_tower)):
        vision_tower = os.path.join(_model_path, vision_tower)

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith(
            'openai') or vision_tower.startswith(
                'laion') or 'ShareGPT4V' in vision_tower:
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')


def _set_function(old_func, new_func):
    import gc
    refs = gc.get_referrers(old_func)
    obj_id = id(old_func)
    for ref in refs:
        if isinstance(ref, dict):
            for x, y in ref.items():
                if id(y) == obj_id:
                    ref[x] = new_func
        elif isinstance(ref, MutableSequence):
            for i, v in enumerate(ref):
                if id(v) == obj_id:
                    ref[i] = new_func


@contextmanager
def init_yi_model():
    import llava  # noqa: F401
    old_projector = eval(
        'llava.model.multimodal_projector.builder.build_vision_projector')
    _set_function(old_projector, _build_vision_projector)
    old_vision_tower = eval(
        'llava.model.multimodal_encoder.builder.build_vision_tower')
    _set_function(old_vision_tower, _build_vision_tower)
    yield
    _set_function(_build_vision_projector, old_projector)
    _set_function(_build_vision_tower, old_vision_tower)


@contextmanager
def disable_transformers_logging():
    import transformers
    from transformers.utils import logging
    previous_level = logging.get_verbosity()
    logging.set_verbosity(transformers.logging.ERROR)
    yield
    logging.set_verbosity(previous_level)


class YiVisionModel(LlavaVisionModel):
    """Yi visual model."""

    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device
        self.build_model()

    def build_model(self):
        """build model & load weights."""
        check_llava_install()

        global _model_path
        _model_path = self.model_path

        with init_yi_model(), disable_transformers_logging():
            super().build_model()
