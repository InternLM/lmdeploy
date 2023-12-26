# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
import re
from copy import copy
from typing import Any, Dict, Sequence

import torch
import torch.distributed as dist
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
    logger.debug(
        f'Find rewrite of module {origin_qualname}: {rewrite_qualname}.')
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


def _params_to_meta(model: torch.nn.Module):
    """move parameters to meta device."""
    # recursive over children
    for _, child in model.named_children():
        _params_to_meta(child)

    for k, v in model.named_parameters(recurse=False):
        model.register_parameter(
            k, torch.nn.Parameter(v.to('meta'), requires_grad=False))


def _load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor] = None,
    rank: int = 0,
    world_size: int = 1,
    device_mesh: DeviceMesh = None,
    state_prefix: str = '',
):
    """Load state dict by rank.

    Load full state dict into device memory is not possible in LLM.
    This method load shard and partition weights for different
    distribution rank

    Args:
        model (Module): Model to load weight.
        state_dict (Dict[str, Tensor]): State dict object.
        rank (int): Distribution rank.
        world_size (int): Distribution world size.
        device_mesh (DeviceMesh): Distribution device mesh.
        state_prefix (str): The prefix of state dict.

    Returns:
        Module: Updated model
    """

    def _recursive_children():
        """recursive children."""
        for name, child in model.named_children():
            loaded_child = _load_state_dict(
                child,
                state_dict,
                rank,
                world_size,
                device_mesh=device_mesh,
                state_prefix=f'{state_prefix}{name}.',
            )
            if loaded_child != child:
                model.register_module(name, loaded_child)

    def _init_parameters():
        """init parameters."""
        model_state_dict = model.state_dict()
        device = torch.device(f'cuda:{rank}')
        for k, v in model_state_dict.items():
            if '.' in k or not v.is_meta:
                # only process weight that directly owned by module
                # already initialized
                continue

            full_k = state_prefix + k
            if rank == 0:
                objs = [full_k in state_dict]
            else:
                objs = [None]
            dist.broadcast_object_list(objs, 0)
            in_state_dict = objs[0]

            if not in_state_dict:
                continue

            param_names = [
                name for name, _ in model.named_parameters(recurse=False)
            ]
            if k in param_names:
                if rank == 0:
                    new_param = torch.nn.Parameter(
                        state_dict[full_k].to(v.dtype),
                        requires_grad=False).to(device)
                else:
                    new_param = torch.nn.Parameter(torch.empty_like(
                        v, device=device),
                                                   requires_grad=False)
                model.register_parameter(k, new_param)

            buffer_names = [
                name for name, _ in model.named_buffers(recurse=False)
            ]
            if k in buffer_names:
                if rank == 0:
                    new_buffer = state_dict[full_k].to(v.dtype).to(device)
                else:
                    new_buffer = torch.empty_like(v, device=device)
                model.register_buffer(k, new_buffer)

    def _check_need_dist(model):
        """check need dist."""
        need_dist = not getattr(model, '__tp_distributed__', False)
        finish_param_init = all(not v.is_meta
                                for v in model.state_dict().values())
        return need_dist and finish_param_init

    _recursive_children()
    _init_parameters()

    # distribute module
    if world_size > 1 and _check_need_dist(model):
        model.__tp_distributed__ = True

        if hasattr(model, '_distribute_partition_fn'):
            partition_module(
                model,
                device_mesh=device_mesh,
                func=model._distribute_partition_fn,
                to_local=True,
            )
        else:
            replicate_module(model, device_mesh=device_mesh)

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

    return model


def patch(
    model: torch.nn.Module,
    extra_args: Sequence[str] = None,
    rank: int = 0,
    world_size: int = 1,
    checkpoints: Sequence[str] = None,
):
    """Patch the model with rewrite modules.

    Extra arguments will be patched in forward of model, weights on each rank
    will be partitioned.

    Args:
        model (Module): Model to be patched.
        extra_args (Sequence[str]): Extra arguments of model forward.
        rank (int): Distribution rank.
        world_size (int): Distribution world size.
        checkpoints (Sequence[str]): checkpoints of the model.

    Returns:
        Module: The patched model.
    """

    def _load_checkpoints(model, checkpoints, rank, world_size):
        """load checkpoints."""
        _params_to_meta(model)
        device_mesh = DeviceMesh('cuda', list(range(world_size)))
        for ckpt in checkpoints:
            if rank == 0:
                logger = get_logger('lmdeploy')
                logger.info(f'loading checkpoint from: {ckpt}')
                state_dict = torch.load(ckpt, map_location=f'cuda:{rank}')
            else:
                state_dict = None

            with torch.cuda.device(rank):
                _load_state_dict(
                    model,
                    state_dict,
                    rank=rank,
                    world_size=world_size,
                    device_mesh=device_mesh,
                )

    if extra_args is None:
        extra_args = []

    _patch_context = Addict()

    model = _patch(model, _patch_context)

    # load checkpoint
    if checkpoints is not None:
        _load_checkpoints(model, checkpoints, rank, world_size)

    _update_model(model)
    extra_args_str = ' '.join(f'{arg}=None,' for arg in extra_args)
    context_update_str = ' '.join(f'{arg}={arg},' for arg in extra_args)

    wrap_forward_src = f"""
from functools import wraps
# old_forward = model.forward
old_forward = type(model).forward
@wraps(old_forward)
def wrap_forward(self, *args, {extra_args_str} **kwargs):
    global _patch_context
    _patch_context.update({context_update_str})

    output = old_forward(self, *args, **kwargs)

    _patch_context.clear()

    return output
# model.forward = wrap_forward

attrs = dict(type(model).__dict__)
attrs.update(dict(forward=wrap_forward))
class_name  = model.__class__.__name__
new_type = type(class_name, (type(model), ), attrs)
model.__class__ = new_type
"""

    exec(wrap_forward_src, dict(_patch_context=_patch_context, model=model))

    return model
