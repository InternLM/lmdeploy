# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
from typing import Any

import torch

from lmdeploy.utils import get_logger

MODULE_MAP = {
    'transformers.models.llama.modeling_llama.LlamaAttention':
    'lmdeploy.pytorch_poc.patch.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaModel':
    'lmdeploy.pytorch_poc.patch.llama.LlamaModel'
}


def _class_from_qualname(qualname):
    last_dot = qualname.rfind('.')
    modname = qualname[:last_dot]
    clsname = qualname[last_dot + 1:]

    # get class at runtime
    mod = importlib.import_module(modname)
    assert mod is not None, f'failed to import module: {modname}'
    cls_type = getattr(mod, clsname)
    return cls_type


def patch(model: torch.nn.Module, context: Any = None):
    global MODULE_MAP
    logger = get_logger('lmdeploy')

    # recursive over children
    for name, child in model.named_children():
        patched_child = patch(child)
        if patched_child != child:
            setattr(model, name, patched_child)

    # find rewrite module
    module_name = inspect.getmodule(model).__name__
    class_name = model.__class__.__name__
    origin_qualname = f'{module_name}.{class_name}'
    rewrite_qualname = MODULE_MAP.get(origin_qualname, None)

    if rewrite_qualname is None:
        origin_qualname = class_name
        rewrite_qualname = MODULE_MAP.get(origin_qualname, None)

    if rewrite_qualname is not None:
        logger.debug(
            f'Rewrite module {origin_qualname} with {rewrite_qualname}.')
        cls_type = _class_from_qualname(rewrite_qualname)

        model = cls_type(model, context)

    return model
