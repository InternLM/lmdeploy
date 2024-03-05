# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import re
from copy import copy
from typing import Any, Dict, Sequence

import torch
from addict import Addict
from torch.distributed._tensor import DeviceMesh

from lmdeploy.utils import get_logger

from ..dist_utils import partition_module, replicate_module
from .module_map import MODULE_MAP

logger = get_logger('lmdeploy')


def _get_rewrite_qualname(origin_qualname: str) -> str:
    """get rewrite module from origin module name.

    Args:
        origin_qualname (str): The origin qualname of the module.

    Returns:
        str: The rewrite qualname.
    """
    if origin_qualname in MODULE_MAP:
        return MODULE_MAP[origin_qualname]
    for key, value in MODULE_MAP.items():
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


def _find_rewrite_module_qualname(model):
    """find rewrite module."""
    module_name = inspect.getmodule(model).__name__
    class_name = model.__class__.__name__

    def _find_fullname():
        origin_qualname = f'{module_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname)
        return rewrite_qualname

    def _find_classname():
        origin_qualname = class_name
        rewrite_qualname = _get_rewrite_qualname(origin_qualname)
        return rewrite_qualname

    def _find_submodulename():
        # name with first module
        mod_name = module_name[module_name.rfind('.') + 1:]
        origin_qualname = f'{mod_name}.{class_name}'
        rewrite_qualname = _get_rewrite_qualname(origin_qualname)
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


def _patch(model: torch.nn.Module, context: Addict) -> torch.nn.Module:
    """patch the model with rewrite module.

    Args:
        model (Module): model to be patched.
        context (Addict): The environment info to patched in model

    Returns:
        Module: The patched model
    """

    def _recursive_children(context, named_children):
        """recursive children."""
        for name, child in named_children:
            patched_child = _patch(child, context)
            if patched_child != child:
                model.register_module(name, patched_child)

    _recursive_children(context, model.named_children())
    rewrite_qualname = _find_rewrite_module_qualname(model)

    if rewrite_qualname is not None:
        cls_type = _class_from_qualname(rewrite_qualname)
        model = _update_module_type(model, cls_type, dict(context=context))

    return model


def _update_model(model: torch.nn.Module):
    """Update model after patch and load.

    Args:
        model (Module): The model to be updated.
    """
    # recursive over children
    for _, child in model.named_children():
        _update_model(child)

    if hasattr(model, '_update_model_fn'):
        model._update_model_fn()


def _dist_model(model: torch.nn.Module,
                rank: int = 0,
                device_mesh: DeviceMesh = None):
    """distribute model parameters."""

    def _init_params():
        """init params."""
        device = torch.device(f'cuda:{rank}')
        for name, param in model.named_parameters(recurse=False):
            if device != param.device:
                if rank == 0:
                    new_param = param.to(device)
                    model.register_parameter(
                        name, torch.nn.Parameter(new_param,
                                                 requires_grad=False))
                else:
                    new_param = torch.empty_like(param, device=device)
                    model.register_parameter(
                        name, torch.nn.Parameter(new_param,
                                                 requires_grad=False))

        for name, param in model.named_buffers(recurse=False):
            if device != param.device:
                if rank == 0:
                    new_param = param.to(device)
                    model.register_buffer(name, new_param)
                else:
                    new_param = torch.empty_like(param, device=device)
                    model.register_buffer(name, new_param)

    def _dist_params():
        """dist params."""
        if hasattr(model, '_distribute_partition_fn'):
            partition_module(
                model,
                device_mesh=device_mesh,
                func=model._distribute_partition_fn,
                to_local=True,
            )
        else:
            replicate_module(model, device_mesh=device_mesh)
        torch.cuda.empty_cache()

    def _register_hooks():
        """register hooks."""
        if hasattr(model, '_distribute_input_fn'):
            input_fn = model._distribute_input_fn
            model.register_forward_pre_hook(
                lambda _, inputs, inputs_dict: input_fn(
                    inputs, inputs_dict, device_mesh),
                with_kwargs=True,
            )

        if hasattr(model, '_distribute_output_fn'):
            output_fn = model._distribute_output_fn
            model.register_forward_hook(
                lambda mod, inputs, outputs: output_fn(outputs, device_mesh))

    for name, child in model.named_children():
        if rank == 0:
            logger.debug(f'Distribute module: <{name}>')
        new_child = _dist_model(child, rank, device_mesh)
        if new_child != child:
            model.register_module(name, child)

    _init_params()
    _dist_params()
    _register_hooks()

    return model


class PatchedForward:
    """patched forward."""

    def __init__(self, model, context, extra_args):
        self._model = model
        self._patch_context: Dict = context
        self._extra_args: list = extra_args

    def __call__(self, *args, **kwargs):
        for arg_name in self._extra_args:
            extra_arg = kwargs.pop(arg_name, None)
            self._patch_context[arg_name] = extra_arg

        output = self._model(*args, **kwargs)

        self._patch_context.clear()

        return output


def patch(
    model: torch.nn.Module,
    extra_args: Sequence[str] = None,
    rank: int = 0,
    world_size: int = 1,
):
    """Patch the model with rewrite modules.

    Extra arguments will be patched in forward of model, weights on each rank
    will be partitioned.

    Args:
        model (Module): Model to be patched.
        extra_args (Sequence[str]): Extra arguments of model forward.
        rank (int): Distribution rank.
        world_size (int): Distribution world size.

    Returns:
        Module: The patched model.
    """

    if extra_args is None:
        extra_args = []

    _patch_context = Addict()

    model = _patch(model, _patch_context)

    if world_size > 1:
        if rank == 0:
            logger.info('distribute model parameters.')
        device_mesh = DeviceMesh('cuda', list(range(world_size)))
        model = _dist_model(model, rank, device_mesh=device_mesh)

    _update_model(model)

    patched_forward = PatchedForward(model,
                                     _patch_context,
                                     extra_args=extra_args)
    model.patched_forward = patched_forward

    return model
