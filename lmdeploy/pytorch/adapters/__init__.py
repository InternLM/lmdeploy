# Copyright (c) OpenMMLab. All rights reserved.

from .basic import BasicAdapter


def init_adapter(model, tokenizer):
    Adapter = {}.get(model.__class__.__name__, BasicAdapter)
    return Adapter(tokenizer)
