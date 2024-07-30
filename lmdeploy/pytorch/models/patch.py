# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import re
from copy import copy
from typing import Any, Dict

import torch

from lmdeploy.utils import get_logger

from ..devices import get_device_manager
from .module_map import DEVICE_SPECIAL_MODULE_MAP, MODULE_MAP

logger = get_logger('lmdeploy')


def _get_rewrite_qualname(origin_qualname: str, module_map: Dict[str,
                                                                 str]) -> str:
    """get rewrite module from origin module name.

    Args:
        origin_qualname (str): The origin qualname of the module.

    Returns:
        str: The rewrite qualname.
    """
    if origin_qualname in module_map:
        return module_map[origin_qualname]
    for key, value in module_map.items():
        if re.search(key, origin_qualname):
            return value
    return None


def _class_from_qualname(qualname: str) -> Any:
    """Import class with qualname.

    Args:
        qualname (str): Qualname of the class

    Returns:
        Any: class or builder of the class
    """
    last_dot = qualname.rfind('.')
    modname = qualname[:last_dot]
    clsname = qualname[last_dot + 1:]

    # get class at runtime
    mod = importlib.import_module(modname)
    assert mod is not None, f'failed to import module: {modname}'
    cls_type = getattr(mod, clsname)
    return cls_type


def _find_rewrite_module_qualname(model, module_map: Dict[str, str]):
    """find rewrite module."""
    module_name = inspect.getmodule(model).__name__
    class_name = model.__class__.__name__

    def _find_fullname():
        origin_qualname = f'{module_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    def _find_classname():
        origin_qualname = class_name
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    def _find_submodulename():
        # name with first module
        mod_name = module_name[module_name.rfind('.') + 1:]
        origin_qualname = f'{mod_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname, module_map)
        return rewrite_qualname

    rewrite_qualname = _find_fullname()
    if rewrite_qualname is None:
        rewrite_qualname = _find_classname()
    if rewrite_qualname is None:
        rewrite_qualname = _find_submodulename()

    origin_qualname = f'{module_name}.{class_name}'
    if rewrite_qualname is not None:
        logger.debug('Find rewrite of module\n'
                     f'{origin_qualname} <=> {rewrite_qualname}')
    return rewrite_qualname


def _update_module_type(model: Any, cls_type: type, custom_attrs: dict = None):
    """Update class type of model."""
    # directly return origin model is not cool
    # origin model would be registered as a submodule
    old_type = type(model)

    @property
    def get_origin_mod(self):
        origin_mod = copy(self)
        origin_mod.__class__ = old_type
        return origin_mod

    attrs = dict(cls_type.__dict__)
    custom_attrs = custom_attrs or dict()
    custom_attrs['origin_mod'] = get_origin_mod
    attrs.update(custom_attrs)
    new_type = type(cls_type.__name__, (type(model), ), attrs)
    model = copy(model)
    model.__class__ = new_type

    return model


def get_rewrite_cls(model: torch.nn.Module, module_map: Dict[str, str] = None):
    """get rewrite cls."""
    if module_map is None:
        module_map = _get_module_map()
    rewrite_qualname = _find_rewrite_module_qualname(model,
                                                     module_map=module_map)
    if rewrite_qualname is None:
        return None
    return _class_from_qualname(rewrite_qualname)


def _patch(model: torch.nn.Module, module_map: Dict[str,
                                                    str]) -> torch.nn.Module:
    """patch the model with rewrite module.

    Args:
        model (Module): model to be patched.

    Returns:
        Module: The patched model
    """

    def _recursive_children(named_children):
        """recursive children."""
        for name, child in named_children:
            _patch(child, module_map=module_map)

    _recursive_children(model.named_children())
    rewrite_qualname = _find_rewrite_module_qualname(model,
                                                     module_map=module_map)

    if rewrite_qualname is not None:
        cls_type = _class_from_qualname(rewrite_qualname)
        if hasattr(cls_type, '_load_weights'):
            setattr(model, '_load_weights', cls_type._load_weights)

    return model


def _get_module_map():
    """get module map."""
    module_map = MODULE_MAP.copy()
    device_type = get_device_manager().current_context().device_type
    if device_type != 'cuda':
        device_map = DEVICE_SPECIAL_MODULE_MAP.get(device_type, dict())
        module_map.update(device_map)
    return module_map


@torch.inference_mode()
def patch(model: torch.nn.Module, ):
    """Patch the model with rewrite modules.

    Extra arguments will be patched in forward of model, weights on each rank
    will be partitioned.

    Args:
        model (Module): Model to be patched.

    Returns:
        Module: The patched model.
    """
    module_map = _get_module_map()
    model = _patch(model, module_map=module_map)
    return model


def update_model(model: torch.nn.Module):
    """build model."""
    from lmdeploy.pytorch.model_inputs import StepContextManager
    ctx_mgr = StepContextManager()
    module_map = _get_module_map()

    rewrite_qualname = _find_rewrite_module_qualname(model,
                                                     module_map=module_map)

    if rewrite_qualname is not None:
        model_cls = _class_from_qualname(rewrite_qualname)

    return model_cls(model, ctx_mgr)
