# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn
from torch.distributed._tensor import (DeviceMesh, DTensor, Replicate, Shard,
                                       distribute_tensor)


def try_to_local(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor


def rowwise_parallelize_linear_fn(
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
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
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, dist_spec))
        module.register_parameter(name, dist_param)


def colwise_parallelize_linear_fn(
    module: nn.Module,
    device_mesh: DeviceMesh,
) -> None:
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
        dist_param = torch.nn.Parameter(
            distribute_tensor(param, device_mesh, [Shard(0)]))
        module.register_parameter(name, dist_param)
