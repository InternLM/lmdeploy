# Copyright (c) OpenMMLab. All rights reserved.

import logging

import torch.nn as nn

from .base import BasicAdapter, BasicAdapterFast
from .internlm import InternLMAdapter
from .llama2 import Llama2Adapter

logger = logging.getLogger(__name__)


def _get_default_adapter(tokenizer):
    if tokenizer.is_fast:
        return BasicAdapterFast
    else:
        return BasicAdapter


def init_adapter(model: nn.Module, tokenizer, adapter=None):
    if adapter is None:
        for v in model.modules():
            if 'InternLMModel' in v.__class__.__name__:
                Adapter = InternLMAdapter
                break
            elif 'LlamaModel' in v.__class__.__name__:
                Adapter = Llama2Adapter
                break
        else:
            Adapter = _get_default_adapter(tokenizer)
    elif adapter == 'llama1':
        Adapter = _get_default_adapter(tokenizer)
    else:
        raise ValueError(f'Adapter {adapter} is not allowed.')

    logger.info(f'Using adapter {Adapter.__name__}')

    return Adapter(tokenizer)
