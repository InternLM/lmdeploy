# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

import PIL
import torch


class VisonModel(ABC):
    """Visual model which extract image feature."""

    @abstractmethod
    def forward(self, images: List[PIL.Image.Image]) -> List[torch.Tensor]:
        """extract image feature.

        Args:
            images (List[PIL.Image.Image]): input images

        Return:
            List[torch.Tensor]: extract image feature for each input image
        """
        raise NotImplementedError()
