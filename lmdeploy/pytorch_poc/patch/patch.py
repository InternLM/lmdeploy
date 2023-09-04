# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
from typing import Sequence

import torch
from addict import Addict
from transformers.utils import (HF_MODULES_CACHE,
                                TRANSFORMERS_DYNAMIC_MODULE_NAME)

from lmdeploy.utils import get_logger

MODULE_MAP = {
    'transformers.models.llama.modeling_llama.LlamaAttention':
    'lmdeploy.pytorch_poc.patch.llama.LlamaAttention',
    'transformers.models.llama.modeling_llama.LlamaModel':
    'lmdeploy.pytorch_poc.patch.llama.LlamaModel',
    # 动态模块路径是不固定的，有点麻烦
    f'{TRANSFORMERS_DYNAMIC_MODULE_NAME}.chatglm2-6b.modeling_chatglm.SelfAttention':
    'lmdeploy.pytorch_poc.patch.chatglm2.PatchedSelfAttention',
    # f"{TRANSFORMERS_DYNAMIC_MODULE_NAME}.chatglm2-6b.modeling_chatglm.ChatGLMModel":
    # "lmdeploy.pytorch_poc.patch.chatglm2.ChatGLMModel",
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


def _patch(model: torch.nn.Module, context: Addict):
    global MODULE_MAP
    logger = get_logger('lmdeploy')

    # recursive over children
    for name, child in model.named_children():
        patched_child = _patch(child, context)
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


def patch(model: torch.nn.Module, extra_args: Sequence[str] = None):
    if extra_args is None:
        extra_args = []

    _patch_context = Addict()

    model = _patch(model, _patch_context)

    extra_args_str = ' '.join(f'{arg}=None,' for arg in extra_args)
    context_update_str = ' '.join(f'{arg}={arg},' for arg in extra_args)

    wrap_forward_src = f"""\
from functools import wraps
old_forward = model.forward
@wraps(old_forward)
def wrap_forward(*args, {extra_args_str} **kwargs):
    global _patch_context
    _patch_context.update({context_update_str})

    output = old_forward(*args, **kwargs)

    _patch_context.clear()

    return output
model.forward = wrap_forward
    """

    exec(wrap_forward_src, dict(_patch_context=_patch_context, model=model))

    return model
