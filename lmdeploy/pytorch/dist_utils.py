# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Union

import torch
from torch import Tensor, nn
from torch.distributed._tensor import (DeviceMesh, DTensor, Replicate, Shard,
                                       distribute_tensor)

try:
    from peft.tuners.lora import Linear as LoRALinear
except ImportError:

    class LoRALinear:
        pass


def try_to_local(tensor: Union[Tensor, DTensor]):
    """Try to convert DTensor to Tensor.

    Args:
        tensor (Tensor|DTensor): Tensor to convert.
    """
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor


def module_to_local(module: nn.Module):
    """convert all DTensor parameters to Tensor parameters in module.

    Args:
        module (Module): Module to convert.
    """
    for name, mod in module.named_children():
        module_to_local(mod)

    for name, param in module.named_parameters(recurse=False):
        module.register_parameter(name, nn.Parameter(try_to_local(param)))

    for name, buf in module.named_buffers(recurse=False):
        module.register_buffer(name, try_to_local(buf))


def rowwise_parallelize_linear(module: nn.Module,
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
            if name == 'bias':
                # rowwise linear would add bias more than ones.
                dist_tensor /= device_mesh.size()
        dist_param = torch.nn.Parameter(dist_tensor)
        module.register_parameter(name, dist_param)

    # Weight, bias and scale are registered as buffer in QLinear
    for name, buffer in module.named_buffers():
        dist_spec = ([Shard(1)] if name == 'weight' else
                     [Replicate()]  # type: ignore[list-item]
                     )

        dist_tensor = distribute_tensor(buffer, device_mesh, dist_spec)
        if to_local:
            dist_tensor = try_to_local(dist_tensor)
            if name == 'bias':
                # rowwise linear would add bias more than ones.
                dist_tensor /= device_mesh.size()
        module.register_buffer(name, dist_tensor)

        dist_tensor = distribute_tensor(buffer, device_mesh, dist_spec)
        if to_local:
            dist_tensor = try_to_local(dist_tensor)
        module.register_buffer(name, dist_tensor)


def rowwise_parallelize_loralinear(module: LoRALinear,
                                   device_mesh: DeviceMesh,
                                   to_local: bool = False) -> None:
    """rowwize parallelize lora linear.

    Read S-LoRA for more detail.
    """
    rowwise_parallelize_linear(module.base_layer,
                               device_mesh=device_mesh,
                               to_local=to_local)
    for mod in module.lora_A.values():
        rowwise_parallelize_linear(mod,
                                   device_mesh=device_mesh,
                                   to_local=to_local)
    for mod in module.lora_B.values():
        colwise_parallelize_linear(mod,
                                   device_mesh=device_mesh,
                                   to_local=to_local)
    module._tp_mode = 'rowwise'


def rowwise_parallelize_linear_fn(module: nn.Module,
                                  device_mesh: DeviceMesh,
                                  to_local: bool = False) -> None:
    """
    This function parallelizes the input :Linear module in
    :class:`RowwiseParallel` style.

    Args:
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """
    if isinstance(module, torch.nn.Linear):
        return rowwise_parallelize_linear(module,
                                          device_mesh=device_mesh,
                                          to_local=to_local)
    elif isinstance(module, LoRALinear):
        return rowwise_parallelize_loralinear(module,
                                              device_mesh=device_mesh,
                                              to_local=to_local)
    else:
        raise TypeError(f'Unsupported module: {type(module)}')


def colwise_parallelize_linear(module: nn.Module,
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
    # Weight, bias and scale are registered as buffer in QLinear
    for name, buffer in module.named_buffers():
        dist_tensor = distribute_tensor(buffer, device_mesh, [Shard(0)])
        if to_local:
            dist_tensor = try_to_local(dist_tensor)
        module.register_buffer(name, dist_tensor)


def colwise_parallelize_loralinear(module: nn.Module,
                                   device_mesh: DeviceMesh,
                                   to_local: bool = False) -> None:
    """colwise parallelize lora linear."""
    colwise_parallelize_linear(module.base_layer,
                               device_mesh=device_mesh,
                               to_local=to_local)
    for mod in module.lora_A.values():
        colwise_parallelize_linear(mod,
                                   device_mesh=device_mesh,
                                   to_local=to_local)
    for mod in module.lora_B.values():
        colwise_parallelize_linear(mod,
                                   device_mesh=device_mesh,
                                   to_local=to_local)
    module._tp_mode = 'colwise'


def colwise_parallelize_linear_fn(module: nn.Module,
                                  device_mesh: DeviceMesh,
                                  to_local: bool = False) -> None:
    """
    This function parallelizes the input :Linear module in
    :class:`ColwiseParallel` style.

    Args:
        module (:class:`nn.Module`):
            The :class:`nn.Linear` module to be parallelized.
        device_mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices.

    Returns:
        None
    """
    if isinstance(module, torch.nn.Linear):
        return colwise_parallelize_linear(module,
                                          device_mesh=device_mesh,
                                          to_local=to_local)
    elif isinstance(module, LoRALinear):
        return colwise_parallelize_loralinear(module,
                                              device_mesh=device_mesh,
                                              to_local=to_local)
    else:
        raise TypeError(f'Unsupported module: {type(module)}')


def _partition_module(
    mod_name: str,
    prefix: str,
    module: nn.Module,
    device_mesh: DeviceMesh,
    func: Callable,
):
    """partition module.

    Parameters in module won't be force Replicated.

    Args:
        mod_name (str): module name.
        prefix (str): Parameter prefix.
        module (Module): Module to be partitioned.
        device_mesh (DeviceMesh): The device mesh.
        func (Callable): partition callback
    """
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
    """partition module.

    Parameters in module won't be force Replicated.

    Args:
        module (Module): Module to be partitioned.
        device_mesh (DeviceMesh): The device mesh.
        func (Callable): partition callback.
        to_local (bool): Convert all DTensor parameters to Tensor parameters.
    """
    _partition_module('',
                      '',
                      module=module,
                      device_mesh=device_mesh,
                      func=func)

    if to_local:
        module_to_local(module)


def replicate_module(model: nn.Module, device_mesh: DeviceMesh):
    """Replicate all parameters in module.

    Args:
        model (Module): Module to perform replicate.
        device_mesh (DeviceMesh): The distribution device mesh.
    """
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
