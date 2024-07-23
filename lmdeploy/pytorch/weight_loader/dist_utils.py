# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import List

import torch

from lmdeploy.pytorch.models.q_modules import QLinear
from lmdeploy.utils import get_logger

from .model_weight_loader import ModelWeightLoader

logger = get_logger('lmdeploy')

try:
    from peft.tuners.lora import Linear as LoRALinear
except ImportError:
    logger.debug('load peft.tuner.lora.Linear failed.')

    class LoRALinear:
        pass


try:
    from peft.tuners.lora.awq import AwqLoraLinear
except ImportError:
    logger.debug('load peft.tuners.lora.awq.AwqLoraLinear failed.')

    class AwqLoraLinear:
        pass


try:
    from awq.modules.linear.gemm import WQLinear_GEMM
except ImportError:
    logger.debug('load awq.modules.linear.gemm.WQLinearGEMM failed.')

    class WQLinear_GEMM:
        pass


def _div_up(a, b):
    """div up."""
    return (a + b - 1) // b


def _math_lcm(*args):
    """lcm."""
    return int(math.prod(args) / math.gcd(*args))


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

    def __update_param(name, param):
        """update_param."""
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name).chunk(world_size)[rank]
        param = cast_dtype(param, dtype)
        return param

    for name, param in mod.named_parameters():
        param = __update_param(name, param)
        param = torch.nn.Parameter(param, requires_grad=False)
        mod.register_parameter(name, param)
    for name, param in mod.named_buffers():
        param = __update_param(name, param)
        mod.register_buffer(name, param)


def colwise_parallelize_loralinear(module: torch.nn.Module,
                                   loader: ModelWeightLoader,
                                   rank: int,
                                   world_size: int,
                                   prefix: str = ''):
    """colwise parallelize loralinear."""
    if isinstance(module.base_layer, WQLinear_GEMM):
        parallel_base_func = colwise_parallelize_wqlinear
    else:
        parallel_base_func = colwise_parallelize_linear_naive
    parallel_base_func(module.base_layer,
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


def _get_split_size_with_align(size: int, align: int, num_chunk: int):
    """get split size with align."""
    assert size % align == 0
    num_aligned = size // align
    split_size = _div_up(num_aligned, num_chunk) * align
    return split_size


def colwise_parallelize_wqlinear(mod: torch.nn.Module,
                                 loader: ModelWeightLoader,
                                 rank: int,
                                 world_size: int,
                                 prefix: str = ''):
    """colwise parallelize wqlinear."""
    elem_per_word = 32 // mod.w_bit
    group_size = mod.group_size
    lcm = _math_lcm(elem_per_word, group_size)
    num_out = mod.scales.size(1)

    split_size = _get_split_size_with_align(num_out, lcm, world_size)
    qsplit_size = split_size // elem_per_word

    def __update_param(name, param):
        """update_param."""
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        if name == 'bias':
            ssize = split_size
            dim = 0
        elif name == 'scales':
            ssize = split_size
            dim = 1
        else:
            ssize = qsplit_size
            dim = 1
        param = loader.pop(prefixed_name)
        param = param.split(ssize, dim)[rank]
        param = cast_dtype(param, dtype)
        return param

    for name, param in mod.named_parameters():
        param = __update_param(name, param)
        param = torch.nn.Parameter(param, requires_grad=False)
        mod.register_parameter(name, param)
    for name, param in mod.named_buffers():
        param = __update_param(name, param)
        mod.register_buffer(name, param)
    mod.in_features = mod.qweight.size(0)
    mod.out_features = mod.scales.size(1)


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
    elif isinstance(module, (LoRALinear, AwqLoraLinear)):
        return colwise_parallelize_loralinear(module,
                                              loader,
                                              rank=rank,
                                              world_size=world_size,
                                              prefix=prefix)
    elif isinstance(module, WQLinear_GEMM):
        return colwise_parallelize_wqlinear(module,
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

    def __update_param(name: str, param: torch.Tensor):
        """update_param."""
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        if name == 'weight':
            param = param.chunk(world_size, 1)[rank]
        if name == 'bias':
            param /= world_size
        param = cast_dtype(param, dtype)
        return param

    for name, param in mod.named_parameters():
        param = __update_param(name, param)
        param = torch.nn.Parameter(param, requires_grad=False)
        mod.register_parameter(name, param)
    for name, param in mod.named_buffers():
        param = __update_param(name, param)
        mod.register_buffer(name, param)


def rowwise_parallelize_loralinear(module: LoRALinear,
                                   loader: ModelWeightLoader,
                                   rank: int,
                                   world_size: int,
                                   prefix: str = ''):
    """colwise parallelize loralinear."""
    if isinstance(module.base_layer, WQLinear_GEMM):
        parallel_base_func = rowwise_parallelize_wqlinear
    else:
        parallel_base_func = rowwise_parallelize_linear_naive
    parallel_base_func(module.base_layer,
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


def rowwise_parallelize_wqlinear(mod: torch.nn.Module,
                                 loader: ModelWeightLoader,
                                 rank: int,
                                 world_size: int,
                                 prefix: str = ''):
    """rowwise parallelize linear."""
    elem_per_word = 32 // mod.w_bit
    group_size = mod.group_size
    lcm = _math_lcm(elem_per_word, group_size)
    num_in = mod.qweight.size(0)

    split_size = _get_split_size_with_align(num_in, lcm, world_size)
    qsplit_size = split_size // group_size

    def __update_param(name: str, param: torch.Tensor):
        """update_param."""
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        if name == 'bias':
            param /= world_size
        elif name == 'qweight':
            param = param.split(split_size)[rank]
        else:
            param = param.split(qsplit_size)[rank]
        param = cast_dtype(param, dtype)
        return param

    for name, param in mod.named_parameters():
        param = __update_param(name, param)
        param = torch.nn.Parameter(param, requires_grad=False)
        mod.register_parameter(name, param)
    for name, param in mod.named_buffers():
        param = __update_param(name, param)
        mod.register_buffer(name, param)
    mod.in_features = mod.qweight.size(0)
    mod.out_features = mod.scales.size(1)


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
    elif isinstance(module, (LoRALinear, AwqLoraLinear)):
        return rowwise_parallelize_loralinear(module,
                                              loader,
                                              rank=rank,
                                              world_size=world_size,
                                              prefix=prefix)
    elif isinstance(module, WQLinear_GEMM):
        return rowwise_parallelize_wqlinear(module,
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

    def __update_param(name: str, param: torch.Tensor):
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
        return param

    for name, param in module.named_parameters():
        param = __update_param(name, param)
        param = torch.nn.Parameter(param, requires_grad=False)
        module.register_parameter(name, param)
    for name, param in module.named_buffers():
        param = __update_param(name, param)
        module.register_buffer(name, param)


def colwise_split_parallelize_loralinear(module: LoRALinear,
                                         sections: List[int],
                                         loader: ModelWeightLoader,
                                         rank: int,
                                         world_size: int,
                                         prefix: str = ''):
    """colwise split loralinear."""
    if isinstance(module.base_layer, WQLinear_GEMM):
        parallel_base_func = colwise_split_parallelize_wqlinear
    else:
        parallel_base_func = colwise_split_parallelize_linear_naive
    parallel_base_func(module.base_layer,
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


def colwise_split_parallelize_wqlinear(module: torch.nn.Module,
                                       sections: List[int],
                                       loader: ModelWeightLoader,
                                       rank: int,
                                       world_size: int,
                                       prefix: str = ''):
    """colwise split wqlinear."""
    elem_per_word = 32 // module.w_bit
    group_size = module.group_size
    lcm = _math_lcm(elem_per_word, group_size)

    for s in sections:
        assert s % lcm == 0

    def __update_param(name: str, param: torch.Tensor):
        dtype = param.dtype
        prefixed_name = get_prefixed_name(name, prefix)
        param = loader.pop(prefixed_name)
        if name == 'bias':
            dim = 0
            sec = sections
        elif name == 'scales':
            dim = 1
            sec = sections
        else:
            dim = 1
            sec = [s // elem_per_word for s in sections]
        splited_param = param.split(sec, dim=dim)
        updated_param = []
        for p in splited_param:
            if name == 'bias':
                p = p.chunk(world_size)[rank]
            else:
                p = p.chunk(world_size, 1)[rank]
            p = cast_dtype(p, dtype)
            updated_param.append(p)
        if name == 'bias':
            param = torch.cat(updated_param)
        else:
            param = torch.cat(updated_param, 1)
        return param

    for name, param in module.named_parameters():
        param = __update_param(name, param)
        param = torch.nn.Parameter(param, requires_grad=False)
        module.register_parameter(name, param)
    for name, param in module.named_buffers():
        param = __update_param(name, param)
        module.register_buffer(name, param)
    module.in_features = module.qweight.size(0)
    module.out_features = module.scales.size(1)


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
    elif isinstance(module, (LoRALinear, AwqLoraLinear)):
        return colwise_split_parallelize_loralinear(module,
                                                    sections,
                                                    loader,
                                                    rank=rank,
                                                    world_size=world_size,
                                                    prefix=prefix)
    elif isinstance(module, WQLinear_GEMM):
        return colwise_split_parallelize_wqlinear(module,
                                                  sections,
                                                  loader,
                                                  rank=rank,
                                                  world_size=world_size,
                                                  prefix=prefix)
    else:
        raise TypeError(f'Unsupported module: {type(module)}')


def load_no_recursive(mod: torch.nn.Module,
                      loader: ModelWeightLoader,
                      rank: int = 0,
                      prefix: str = ''):
    """default load linear naive."""
    for name, param in mod.named_parameters(recurse=False):
        prefixed_name = get_prefixed_name(name, prefix)
        dtype = param.dtype
        if not loader.has(prefixed_name):
            logger.debug(f'rank [{rank}]'
                         f' failed to find weight: {name}.')
            param = torch.empty_like(param, device='cpu')
        else:
            param = loader.pop(prefixed_name)
        if param.dtype != dtype:
            param = param.to(dtype)
        mod.register_parameter(name,
                               torch.nn.Parameter(param, requires_grad=False))
    for name, param in mod.named_buffers(recurse=False):
        prefixed_name = get_prefixed_name(name, prefix)
        dtype = param.dtype
        if not loader.has(prefixed_name):
            logger.debug(f'rank [{rank}]'
                         f' failed to find weight: {name}.')
            param = torch.empty_like(param, device='cpu')
        else:
            param = loader.pop(prefixed_name)
        if param.dtype != dtype:
            param = param.to(dtype)
        mod.register_buffer(name, param)


def default_load_linear(module: torch.nn.Module,
                        loader: ModelWeightLoader,
                        rank: int = 0,
                        prefix: str = ''):
    """default load linear."""
    if isinstance(module, (torch.nn.Linear, QLinear, WQLinear_GEMM)):
        load_no_recursive(module, loader, rank=rank, prefix=prefix)
    elif isinstance(module, (LoRALinear, AwqLoraLinear)):
        raise NotImplementedError('Not implemented, please contact us.')
    else:
        raise TypeError(f'Unsupported module: {type(module)}')
