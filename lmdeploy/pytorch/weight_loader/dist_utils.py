# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch

from lmdeploy.pytorch.models.q_modules import QLinear

from .model_weight_loader import ModelWeightLoader

try:
    from peft.tuners.lora import Linear as LoRALinear
except ImportError:

    class LoRALinear:
        pass


def get_prefixed_name(name: str, prefix: str):
    """get prefixed name."""
    if len(prefix) == 0:
        return name
    else:
        return f'{prefix}.{name}'


def cast_dtype(param: torch.Tensor, dtype: torch.dtype):
    """cast dtype."""
    if param.dtype != dtype:
        param = param.to(dtype)
    return param


def colwise_parallelize_linear_naive(mod: torch.nn.Module,
                                     loader: ModelWeightLoader,
                                     rank: int,
                                     world_size: int,
                                     prefix: str = ''):
    """colwise parallelize linear."""
    for name, param in mod.named_parameters():
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name).chunk(world_size)[rank]
        param = cast_dtype(param, dtype)
        param = torch.nn.Parameter(param, requires_grad=False)
        mod.register_parameter(name, param)
    for name, param in mod.named_buffers():
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name).chunk(world_size)[rank]
        param = cast_dtype(param, dtype)
        mod.register_buffer(name, param)


def colwise_parallelize_loralinear(module: torch.nn.Module,
                                   loader: ModelWeightLoader,
                                   rank: int,
                                   world_size: int,
                                   prefix: str = ''):
    """colwise parallelize loralinear."""
    colwise_parallelize_linear_naive(module.base_layer,
                                     loader,
                                     rank=rank,
                                     world_size=world_size,
                                     prefix=prefix)
    for key, mod in module.lora_A.items():
        ada_loader = loader.adapter(key)
        colwise_parallelize_linear_naive(mod,
                                         ada_loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix=get_prefixed_name(
                                             'lora_A', prefix))
    for key, mod in module.lora_B.items():
        ada_loader = loader.adapter(key)
        colwise_parallelize_linear_naive(mod,
                                         ada_loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix=get_prefixed_name(
                                             'lora_B', prefix))
    module._tp_mode = 'colwise'


def colwise_parallelize_linear(module: torch.nn.Module,
                               loader: ModelWeightLoader,
                               rank: int,
                               world_size: int,
                               prefix: str = ''):
    """colwise parallelize linear."""
    if isinstance(module, (torch.nn.Linear, QLinear)):
        return colwise_parallelize_linear_naive(module,
                                                loader,
                                                rank=rank,
                                                world_size=world_size,
                                                prefix=prefix)
    elif isinstance(module, LoRALinear):
        return colwise_parallelize_loralinear(module,
                                              loader,
                                              rank=rank,
                                              world_size=world_size,
                                              prefix=prefix)
    else:
        raise TypeError(f'Unsupported module: {type(module)}')


def rowwise_parallelize_linear_naive(mod: torch.nn.Module,
                                     loader: ModelWeightLoader,
                                     rank: int,
                                     world_size: int,
                                     prefix: str = ''):
    """rowwise parallelize linear."""
    for name, param in mod.named_parameters():
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        if name == 'weight':
            param = param.chunk(world_size, 1)[rank]
        if name == 'bias':
            param /= world_size
        param = cast_dtype(param, dtype)
        param = torch.nn.Parameter(param, requires_grad=False)
        mod.register_parameter(name, param)
    for name, param in mod.named_buffers():
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        if name == 'weight':
            param = param.chunk(world_size, 1)[rank]
        if name == 'bias':
            param /= world_size
        param = cast_dtype(param, dtype)
        mod.register_buffer(name, param)


def rowwise_parallelize_loralinear(module: LoRALinear,
                                   loader: ModelWeightLoader,
                                   rank: int,
                                   world_size: int,
                                   prefix: str = ''):
    """colwise parallelize loralinear."""
    rowwise_parallelize_linear_naive(module.base_layer,
                                     loader,
                                     rank=rank,
                                     world_size=world_size,
                                     prefix=prefix)
    for key, mod in module.lora_A.items():
        ada_loader = loader.adapter(key)
        rowwise_parallelize_linear_naive(mod,
                                         ada_loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix=get_prefixed_name(
                                             'lora_A', prefix))
    for key, mod in module.lora_B.items():
        ada_loader = loader.adapter(key)
        colwise_parallelize_linear_naive(mod,
                                         ada_loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix=get_prefixed_name(
                                             'lora_B', prefix))
    module._tp_mode = 'colwise'


def rowwise_parallelize_linear(module: torch.nn.Module,
                               loader: ModelWeightLoader,
                               rank: int,
                               world_size: int,
                               prefix: str = ''):
    """colwise parallelize linear."""
    if isinstance(module, (torch.nn.Linear, QLinear)):
        return rowwise_parallelize_linear_naive(module,
                                                loader,
                                                rank=rank,
                                                world_size=world_size,
                                                prefix=prefix)
    elif isinstance(module, LoRALinear):
        return rowwise_parallelize_loralinear(module,
                                              loader,
                                              rank=rank,
                                              world_size=world_size,
                                              prefix=prefix)
    else:
        raise TypeError(f'Unsupported module: {type(module)}')


def colwise_split_parallelize_linear_naive(module: torch.nn.Module,
                                           sections: List[int],
                                           loader: ModelWeightLoader,
                                           rank: int,
                                           world_size: int,
                                           prefix: str = ''):
    """colwise split linear naive."""
    for name, param in module.named_parameters():
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        splited_param = param.split(sections, dim=0)
        updated_param = []
        for p in splited_param:
            p = p.chunk(world_size)[rank]
            p = cast_dtype(p, dtype)
            updated_param.append(p)
        param = torch.cat(updated_param)
        param = torch.nn.Parameter(param, requires_grad=False)
        module.register_parameter(name, param)
    for name, dtype in module.named_buffers():
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        splited_param = param.split(sections, dim=0)
        updated_param = []
        for p in splited_param:
            p = p.chunk(world_size)[rank]
            p = cast_dtype(p, dtype)
            updated_param.append(p)
        module.register_buffer(name, param)


def colwise_split_parallelize_loralinear(module: LoRALinear,
                                         sections: List[int],
                                         loader: ModelWeightLoader,
                                         rank: int,
                                         world_size: int,
                                         prefix: str = ''):
    """colwise split linear naive."""
    colwise_split_parallelize_linear_naive(module.base_layer,
                                           sections,
                                           loader,
                                           rank=rank,
                                           world_size=world_size,
                                           prefix=prefix)
    for key, mod in module.lora_A.items():
        ada_loader = loader.adapter(key)
        colwise_parallelize_linear_naive(mod,
                                         ada_loader,
                                         rank=rank,
                                         world_size=world_size,
                                         prefix=get_prefixed_name(
                                             'lora_A', prefix))
    for key, mod in module.lora_B.items():
        ada_loader = loader.adapter(key)
        colwise_split_parallelize_linear_naive(mod,
                                               sections,
                                               ada_loader,
                                               rank=rank,
                                               world_size=world_size,
                                               prefix=get_prefixed_name(
                                                   'lora_B', prefix))
    module._tp_mode = 'colwise'


def colwise_split_parallelize_linear(module: torch.nn.Module,
                                     sections: List[int],
                                     loader: ModelWeightLoader,
                                     rank: int,
                                     world_size: int,
                                     prefix: str = ''):
    """colwise split linear."""
    if isinstance(module, (torch.nn.Linear, QLinear)):
        return colwise_split_parallelize_linear_naive(module,
                                                      sections,
                                                      loader,
                                                      rank=rank,
                                                      world_size=world_size,
                                                      prefix=prefix)
    elif isinstance(module, LoRALinear):
        return colwise_split_parallelize_loralinear(module,
                                                    sections,
                                                    loader,
                                                    rank=rank,
                                                    world_size=world_size,
                                                    prefix=prefix)
    else:
        raise TypeError(f'Unsupported module: {type(module)}')
