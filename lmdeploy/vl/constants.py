# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum, auto

IMAGE_TOKEN = '<IMAGE_TOKEN>'


class Modality(Enum):
    IMAGE = auto()
    VIDEO = auto()
    AUDIO = auto()
    TIME_SERIES = auto()
