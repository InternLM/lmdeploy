# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
from PIL.Image import Image

from lmdeploy.vl.model.utils import load_model_from_weight_files

from .llava import LlavaVLModelWrapper


def _build_yi_projector(config):
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


class YiVLModelWrapper(LlavaVLModelWrapper):
    """Yi visual model wrapper."""

    def _build_vision_projector(self):
        """build projector."""
        return _build_yi_projector(self.config)


class YiVLModel(nn.Module):
    """Yi visual model."""

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.model = YiVLModelWrapper(model_path)
        self.model.eval().half()
        load_model_from_weight_files(self, self.model_path)

    def forward(self, images: List[Image]) -> List[torch.Tensor]:
        """forward."""
        return self.model.forward(images)
