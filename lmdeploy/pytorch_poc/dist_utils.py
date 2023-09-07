# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable

import torch
from torch import nn
from torch.distributed._tensor import (DeviceMesh, DTensor, Replicate, Shard,
                                       distribute_tensor)


def try_to_local(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor


def module_to_local(module: nn.Module):
    for name, mod in module.named_children():
        module_to_local(mod)

    for name, param in module.named_parameters(recurse=False):
        module.register_parameter(name, nn.Parameter(try_to_local(param)))

    for name, buf in module.named_buffers(recurse=False):
        module.register_buffer(name, try_to_local(buf))


def rowwise_parallelize_linear_fn(module: nn.Module,
                                  device_mesh: DeviceMesh,
                                  to_local: bool = False) -> None:
    """
    This function parallelizes the input :class:`nn.Linear` module in
    :class:`RowwiseParallel` style.

    Args:
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """
    for name, param in module.named_parameters():
        dist_spec = ([Shard(1)] if name == 'weight' else
                     [Replicate()]  # type: ignore[list-item]
                     )

        dist_tensor = distribute_tensor(param, device_mesh, dist_spec)
        if to_local:
            dist_tensor = try_to_local(dist_tensor)
        dist_param = torch.nn.Parameter(dist_tensor)
        module.register_parameter(name, dist_param)


def colwise_parallelize_linear_fn(module: nn.Module,
                                  device_mesh: DeviceMesh,
                                  to_local: bool = False) -> None:
    """
    This function parallelizes the input :class:`nn.Linear` module in
    :class:`ColwiseParallel` style.

    Args:
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """

    for name, param in module.named_parameters():
        dist_tensor = distribute_tensor(param, device_mesh, [Shard(0)])
        if to_local:
            dist_tensor = try_to_local(dist_tensor)
        dist_param = torch.nn.Parameter(dist_tensor)
        module.register_parameter(name, dist_param)


def _partition_module(mod_name: str, prefix: str, module: nn.Module,
                      device_mesh: DeviceMesh, func: Callable):
    for name, mod in module.named_children():
        child_name = f'{prefix}{name}'
        _partition_module(child_name,
                          child_name + '.',
                          module=mod,
                          device_mesh=device_mesh,
                          func=func)

    func(mod_name, module, device_mesh)


def partition_module(module: nn.Module,
                     device_mesh: DeviceMesh,
                     func: Callable,
                     to_local: bool = False):
    _partition_module('',
                      '',
                      module=module,
                      device_mesh=device_mesh,
                      func=func)

    if to_local:
        module_to_local(module)


def replicate_module(model: nn.Module, device_mesh: DeviceMesh):
    for name, param in model.named_parameters(recurse=False):
        param = distribute_tensor(param,
                                  device_mesh=device_mesh,
                                  placements=[Replicate()]).to_local()
        param = nn.Parameter(param)
        model.register_parameter(name, param)

    for name, buf in model.named_buffers(recurse=False):
        buf = distribute_tensor(buf,
                                device_mesh=device_mesh,
                                placements=[Replicate()]).to_local()
        model.register_buffer(name, buf)
