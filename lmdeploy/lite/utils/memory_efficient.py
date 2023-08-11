# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import re
import warnings
from contextlib import contextmanager
from functools import partial
from typing import List

import torch
from torch import nn

from lmdeploy.lite.defaults import KV_CACHE_SIGNATURE, OFFLOAD_MOD


def extract_return_values(module: nn.Module) -> List[str]:
    """Extracts return values from given module's forward method.

    Args:
        module (nn.Module): Module to inspect

    Returns:
        list[str]: List of return values
    """

    last_line = inspect.getsource(module.forward).rstrip('\n').split('\n')[-1]
    pattern = r'return ([\w\s,]+)'
    match = re.search(pattern, last_line)

    if match:
        return_values = match.group(1).split(',')
        return [value.strip() for value in return_values]
    else:
        return []


def find_kv_cache_idx(module: nn.Module) -> int:
    """Finds index of kv cache signature in module's forward parameters."""

    signatures = list(inspect.signature(module.forward).parameters.keys())
    if KV_CACHE_SIGNATURE not in signatures:
        raise ValueError(f'{KV_CACHE_SIGNATURE} not in signatures of '
                         f'{type(module)} forward.')
    return signatures.index(KV_CACHE_SIGNATURE)


def find_modules_by_return_value(model: nn.Module,
                                 value: str) -> List[nn.Module]:
    """Finds modules in model that return given value.

    Args:
        model (nn.Module): Model to inspect
        value (str): Return value to search for

    Returns:
        list[nn.Module]: List of matching modules

    Raises:
        ValueError: If no matching modules found
    """

    modules = []
    for name, module in model.named_modules():
        returns = extract_return_values(module)
        if value in returns:
            print(f'Found {name} returning {value}')
            modules.append(module)

    if not modules:
        error_msg = f'No modules found returning {value}. '
        error_msg += 'Please check if the default KV_CACHE_SIGNATURE  '
        error_msg += f"'{KV_CACHE_SIGNATURE}' matches what is used in your "
        error_msg += 'model code. If not, you can modify KV_CACHE_SIGNATURE '
        error_msg += 'in `lmdeploy.lite.defaults`.'
        raise ValueError(error_msg)

    return modules


@contextmanager
def offload_kv_cache(model: nn.Module, device: str = 'cuda') -> None:
    """Offloads kv cache to given device during forward pass.

    Args:
        model (nn.Module): Model for inference
        device (str): Device to offload to

    Yields:
        None
    """

    modules = find_modules_by_return_value(model, KV_CACHE_SIGNATURE)

    original_forwards = {mod: mod.forward for mod in modules}
    input_idxs = {mod: find_kv_cache_idx(mod) for mod in modules}
    output_idxs = {
        mod: extract_return_values(mod).index(KV_CACHE_SIGNATURE)
        for mod in modules
    }

    def wrap_forward(module, *args, **kwargs):

        idx = input_idxs[module]
        if idx >= len(args):
            # kv cache in kwargs
            if KV_CACHE_SIGNATURE in kwargs:
                if kwargs[KV_CACHE_SIGNATURE]:
                    kwargs[KV_CACHE_SIGNATURE] = kwargs[KV_CACHE_SIGNATURE].to(
                        device)
            else:
                raise ValueError(f'No kv cache input found at index {idx}')
        else:
            # kv cache in args
            args = list(args)
            args[idx] = args[idx].to(device)
            args = tuple(args)

        result = original_forwards[module](*args, **kwargs)

        result = list(result)
        idx = output_idxs[module]

        # Move kv cache outputs back to CPU
        key = result[idx][0].to('cpu')
        value = result[idx][1].to('cpu')
        torch.cuda.empty_cache()

        result[idx] = (key, value)
        result = tuple(result)

        return result

    try:
        for module in modules:
            original_forwards[module] = module.forward
            module.forward = partial(wrap_forward, module)

        yield

    finally:
        for module in modules:
            module.forward = original_forwards[module]
            del original_forwards[module]


@contextmanager
def offload_weights(model: nn.Module, device: str = 'cuda') -> None:
    """Offloads specified modules to given device during forward pass.

    Args:
        model (nn.Module): Model for inference
        device (str): Device to offload to

    Yields:
        None
    """

    target_modules = OFFLOAD_MOD

    def before_forward(module: nn.Module, inp: torch.Tensor):
        module.to(device)

    def after_forward(module: nn.Module, inp: torch.Tensor, out: torch.Tensor):
        module.to('cpu')
        torch.cuda.empty_cache()

    def _to_device(m, spec_modules, dev):
        if len(spec_modules) == 0 or len(list(m.children())) == 0:
            m.to(dev)
            return

        for child in m.children():
            if isinstance(child, spec_modules):
                child.to('cpu')
            else:
                _to_device(child, spec_modules, dev)
                # m.to(dev)

    warnings.warn('By default, offloading will be done on '
                  '`nn.Linear`. You can add modules which want offload to '
                  'the `lmdeploy.lite.defaults.OFFLOAD_MOD`.')
    target = OFFLOAD_MOD

    _to_device(model, target, device)

    handles = []
    for module in model.modules():
        if isinstance(module, target_modules):
            handle1 = module.register_forward_pre_hook(before_forward)
            handle2 = module.register_forward_hook(after_forward)
            handles.extend([handle1, handle2])

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()

        model.to('cpu')
        torch.cuda.empty_cache()


@contextmanager
def memory_efficient_inference(model: nn.Module,
                               offload: bool = True,
                               device: str = 'cuda') -> None:
    """Memory efficient inference context manager.

    Moves model to device for inference, with option to offload
    specific modules.

    Args:
        model (nn.Module): Model for inference
        offload (bool): Whether to offload modules
        device (str): Device for inference

    Yields:
        None
    """

    if offload:
        warnings.warn('Using offload mode - modules defined in OFFLOAD_MOD '
                      'will be moved to GPU during forward pass only.')
        warnings.warn(
            'Using offload mode will incur performance penalty due to '
            'frequent CPU-GPU data transfers.')
        with torch.inference_mode():
            with offload_kv_cache(model, device):
                with offload_weights(model, device):
                    yield
    else:
        model.to(device)
        with torch.inference_mode():
            yield
