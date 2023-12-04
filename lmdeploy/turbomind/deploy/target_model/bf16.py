# Copyright (c) OpenMMLab. All rights reserved.
from .base import OUTPUT_MODELS
from .fp import TurbomindModel

TurbomindBF16Model = OUTPUT_MODELS.register_module(name='bf16')(TurbomindModel)
