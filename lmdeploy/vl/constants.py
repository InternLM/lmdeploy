# Copyright (c) OpenMMLab. All rights reserved.
from enum import Enum

IMAGE_TOKEN = '<IMAGE_TOKEN>'


class Modality(str, Enum):
    IMAGE = 'image'
    VIDEO = 'video'
    AUDIO = 'audio'
    TIME_SERIES = 'time_series'
