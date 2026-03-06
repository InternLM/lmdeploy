# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Any, ClassVar

from PIL import Image

from .data_type import MultiModalData


@dataclass
class ImageData(MultiModalData):
    data: Image
    loc: int
    meta: dict[str, Any] = None
    type: ClassVar[str] = 'image'
