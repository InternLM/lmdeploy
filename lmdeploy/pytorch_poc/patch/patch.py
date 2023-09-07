# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
from copy import copy
from typing import Dict, Sequence

import torch
import torch.distributed as dist
from addict import Addict
from torch.distributed._tensor import DeviceMesh

from lmdeploy.pytorch_poc.dist_utils import partition_module, replicate_module
from lmdeploy.utils import get_logger

MODULE_MAP = {
    "transformers.models.llama.modeling_llama.LlamaAttention": "lmdeploy.pytorch_poc.patch.llama.LlamaAttention",
    "transformers.models.llama.modeling_llama.LlamaModel": "lmdeploy.pytorch_poc.patch.llama.LlamaModel",
    "transformers.models.llama.modeling_llama.LlamaMLP": "lmdeploy.pytorch_poc.patch.llama.LlamaMLP",
}


def _class_from_qualname(qualname):
    last_dot = qualname.rfind(".")
    modname = qualname[:last_dot]
    clsname = qualname[last_dot + 1 :]

    # get class at runtime
    mod = importlib.import_module(modname)
    assert mod is not None, f"failed to import module: {modname}"
    cls_type = getattr(mod, clsname)
    return cls_type


def _patch(model: torch.nn.Module, context: Addict):
    global MODULE_MAP
    logger = get_logger("lmdeploy")

    # recursive over children
    for name, child in model.named_children():
        patched_child = _patch(child, context)
        if patched_child != child:
            model.register_module(name, patched_child)

    # find rewrite module
    module_name = inspect.getmodule(model).__name__
    class_name = model.__class__.__name__
    origin_qualname = f"{module_name}.{class_name}"
    # full qualname
    rewrite_qualname = MODULE_MAP.get(origin_qualname, None)

    if rewrite_qualname is None:
        # class name only
        origin_qualname = class_name
        rewrite_qualname = MODULE_MAP.get(origin_qualname, None)

    if rewrite_qualname is None:
        # name with first module
        mod_name = module_name[module_name.rfind(".") + 1 :]
        origin_qualname = f"{mod_name}.{class_name}"
        rewrite_qualname = MODULE_MAP.get(origin_qualname, None)

    if rewrite_qualname is not None:
        logger.debug(f"Rewrite module {origin_qualname} with {rewrite_qualname}.")
        cls_type = _class_from_qualname(rewrite_qualname)

        # directly return origin model is not cool
        # origin model would be registered as a submodule

        old_type = type(model)

        @property
        def get_origin_mod(self):
            origin_mod = copy(self)
            origin_mod.__class__ = old_type
            return origin_mod

        attrs = dict(cls_type.__dict__)
        attrs.update(dict(context=context, origin_mod=get_origin_mod))
        new_type = type(class_name, (type(model),), attrs)
        model = copy(model)
        model.__class__ = new_type

        if hasattr(model, "_update_model_fn"):
            model._update_model_fn()

    return model


def _update_model(model: torch.nn.Module):
    # recursive over children
    for _, child in model.named_children():
        _update_model(child)

    if hasattr(model, "_update_model_fn"):
        model._update_model_fn()


def _load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor] = None,
    rank: int = 0,
    world_size: int = 1,
    device_mesh: DeviceMesh = None,
    state_prefix: str = "",
):
    # post order
    for name, child in model.named_children():
        loaded_child = _load_state_dict(
            child,
            state_dict,
            rank,
            world_size,
            device_mesh=device_mesh,
            state_prefix=f"{state_prefix}{name}.",
        )
        if loaded_child != child:
            model.register_module(name, loaded_child)

    # try load states
    model_state_dict = model.state_dict()

    # init model on device
    device = torch.device(f"cuda:{rank}")
    for k, v in model_state_dict.items():
        if "." in k:
            # only process weight that directly owned by module
            continue

        if not v.is_meta:
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

        if rank == 0:
            new_param = torch.nn.Parameter(state_dict[full_k], requires_grad=False).to(
                device
            )
        else:
            new_param = torch.nn.Parameter(
                torch.empty_like(v, device=device), requires_grad=False
            )
        model.register_parameter(k, new_param)

    # distribute module
    if world_size > 1:
        # check if the model require dist
        need_dist = not getattr(model, "__tp_distributed__", False)
        for v in model.state_dict().values():
            # model has been disted or weight has not been initialized
            if v.is_meta:
                need_dist = False
                break

        # dist
        if need_dist:
            model.__tp_distributed__ = True

            if hasattr(model, "_distribute_partition_fn"):
                partition_module(
                    model,
                    device_mesh=device_mesh,
                    func=model._distribute_partition_fn,
                    to_local=True,
                )
            else:
                replicate_module(model, device_mesh=device_mesh)

            if hasattr(model, "_distribute_input_fn"):
                input_fn = model._distribute_input_fn
                model.register_forward_pre_hook(
                    lambda _, inputs, inputs_dict: input_fn(
                        inputs, inputs_dict, device_mesh
                    ),
                    with_kwargs=True,
                )

            if hasattr(model, "_distribute_output_fn"):
                output_fn = model._distribute_output_fn
                model.register_forward_hook(
                    lambda mod, inputs, outputs: output_fn(outputs, device_mesh)
                )

    return model


def patch(
    model: torch.nn.Module,
    extra_args: Sequence[str] = None,
    rank: int = 0,
    world_size: int = 1,
    checkpoints: Sequence[str] = None,
):
    if extra_args is None:
        extra_args = []

    _patch_context = Addict()

    # patch model
    model = _patch(model, _patch_context)

    # load checkpoint
    if checkpoints is not None:
        device_mesh = DeviceMesh("cuda", list(range(world_size)))
        for ckpt in checkpoints:
            if rank == 0:
                logger = get_logger("lmdeploy")
                logger.info(f"loading checkpoint from: {ckpt}")
                state_dict = torch.load(ckpt, map_location=f"cuda:{rank}")
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

    _update_model(model)
    extra_args_str = " ".join(f"{arg}=None," for arg in extra_args)
    context_update_str = " ".join(f"{arg}={arg}," for arg in extra_args)

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
