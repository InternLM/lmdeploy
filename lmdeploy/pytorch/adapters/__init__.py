# Copyright (c) OpenMMLab. All rights reserved.

import logging

from .basic import BasicAdapter
from .internlm import InternLMAdapter

import torch.nn as nn

logger = logging.getLogger(__name__)


def init_adapter(model: nn.Module, tokenizer):
    for v in model.modules():
        if "InternLMModel" in v.__class__.__name__:
            Adapter = InternLMAdapter
            break
    else:
        Adapter = BasicAdapter

    logger.info(f"Using adapter {Adapter.__name__}")

    return Adapter(tokenizer)
