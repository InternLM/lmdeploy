# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import PIL
import torch
from mmengine import Registry

VISION_MODELS = Registry('vision_model')


class VisonModel(ABC):
    """Visual model which extract image feature."""
    _arch: Union[str, List[str]] = None

    def __init__(self,
                 model_path: str,
                 with_llm: bool = False,
                 max_memory: Dict[int, int] = None):
        """init."""
        self.model_path = model_path
        self.with_llm = with_llm
        self.max_memory = max_memory

    @abstractmethod
    def forward(self, images: List[PIL.Image.Image]) -> List[torch.Tensor]:
        """extract image feature.

        Args:
            images (List[PIL.Image.Image]): input images

        Return:
            List[torch.Tensor]: extract image feature for each input image
        """
        raise NotImplementedError()

    @classmethod
    def match(cls, config: dict):
        """check whether the config match the model."""
        arch = config['architectures'][0]
        if arch == cls._arch or arch in cls._arch:
            return True
        return False
