# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import re
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Generator, List, Optional

import torch
from torch import nn

from lmdeploy.lite.defaults import KV_CACHE_SIGNATURE, OFFLOAD_MOD
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def extract_return_values(module: nn.Module) -> List[str]:
    """Extracts return values from given module's forward method.

    Args:
        module: Module to inspect

    Returns:
        List of return values
    """
    try:
        source = inspect.getsource(module.forward)
        last_line = source.rstrip('\n').split('\n')[-1]
        pattern = r'return ([\w\s,]+)'
        match = re.search(pattern, last_line)

        if match:
            return_values = match.group(1).split(',')
            return [value.strip() for value in return_values]
        return []
    except Exception as e:
        logger.warning(f'Failed to extract return values from {type(module).__name__}: {e}')
        return []


def find_kv_cache_idx(module: nn.Module) -> int:
    """Finds index of kv cache signature in module's forward parameters.

    Args:
        module: Module to inspect

    Returns:
        Index of kv cache parameter

    Raises:
        ValueError: If KV_CACHE_SIGNATURE not found in module's forward parameters
    """
    signatures = list(inspect.signature(module.forward).parameters.keys())
    if KV_CACHE_SIGNATURE not in signatures:
        raise ValueError(f'{KV_CACHE_SIGNATURE} not in signatures of '
                         f'{type(module).__name__} forward. Found: {signatures}')
    return signatures.index(KV_CACHE_SIGNATURE)


def find_modules_by_return_value(model: nn.Module, value: str) -> List[nn.Module]:
    """Finds modules in model that return given value.

    Args:
        model: Model to inspect
        value: Return value to search for

    Returns:
        List of matching modules

    Raises:
        ValueError: If no matching modules found
    """
    modules: List[nn.Module] = []
    for name, module in model.named_modules():
        returns = extract_return_values(module)
        if value in returns:
            logger.debug(f'Found {name} returning {value}')
            modules.append(module)

    if not modules:
        error_msg = (
            f'No modules found returning {value}. '
            f'Please check if the default KV_CACHE_SIGNATURE '
            f"'{KV_CACHE_SIGNATURE}' matches what is used in your "
            f'model code. If not, you can modify KV_CACHE_SIGNATURE '
            f'in `lmdeploy.lite.defaults`.'
        )
        raise ValueError(error_msg)

    return modules


@contextmanager
def offload_kv_cache(model: nn.Module, device: str = 'cuda') -> Generator[None, None, None]:
    """Offloads kv cache to given device during forward pass.

    Args:
        model: Model for inference
        device: Device to offload to

    Yields:
        None
    """
    if device not in ('cpu', 'cuda'):
        raise ValueError(f'device must be "cpu" or "cuda", got {device}')

    modules = find_modules_by_return_value(model, KV_CACHE_SIGNATURE)

    original_forwards: Dict[nn.Module, Any] = {}
    input_idxs: Dict[nn.Module, int] = {}
    output_idxs: Dict[nn.Module, int] = {}

    for mod in modules:
        original_forwards[mod] = mod.forward
        input_idxs[mod] = find_kv_cache_idx(mod)
        return_vals = extract_return_values(mod)
        if KV_CACHE_SIGNATURE not in return_vals:
            raise ValueError(f'{KV_CACHE_SIGNATURE} not in return values of {type(mod).__name__}')
        output_idxs[mod] = return_vals.index(KV_CACHE_SIGNATURE)

    def wrap_forward(module: nn.Module, *args: Any, **kwargs: Any) -> Any:
        idx = input_idxs[module]
        if idx >= len(args):
            if KV_CACHE_SIGNATURE in kwargs and kwargs[KV_CACHE_SIGNATURE]:
                kwargs[KV_CACHE_SIGNATURE] = kwargs[KV_CACHE_SIGNATURE].to(device)
            else:
                raise ValueError(f'No kv cache input found at index {idx}')
        else:
            args_list = list(args)
            args_list[idx] = args_list[idx].to(device)
            args = tuple(args_list)

        result = original_forwards[module](*args, **kwargs)

        if not isinstance(result, tuple):
            return result

        result_list = list(result)
        idx = output_idxs[module]

        key = result_list[idx][0].to('cpu')
        value = result_list[idx][1].to('cpu')
        torch.cuda.empty_cache()

        result_list[idx] = (key, value)
        return tuple(result_list)

    try:
        for module in modules:
            module.forward = partial(wrap_forward, module)
        yield
    finally:
        for module in modules:
            module.forward = original_forwards[module]
        original_forwards.clear()


@contextmanager
def offload_weights(model: nn.Module, device: str = 'cuda') -> Generator[None, None, None]:
    """Offloads specified modules to given device during forward pass.

    Args:
        model: Model for inference
        device: Device to offload to

    Yields:
        None
    """
    if device not in ('cpu', 'cuda'):
        raise ValueError(f'device must be "cpu" or "cuda", got {device}')

    target_modules = OFFLOAD_MOD

    def before_forward(module: nn.Module, inp: Any) -> None:
        module.to(device)

    def after_forward(module: nn.Module, inp: Any, out: Any) -> None:
        module.to('cpu')
        torch.cuda.empty_cache()

    def _to_device(m: nn.Module, spec_modules: tuple, dev: str) -> None:
        if len(spec_modules) == 0 or len(list(m.children())) == 0:
            m.to(dev)
            return

        for child in m.children():
            if isinstance(child, spec_modules):
                child.to('cpu')
            else:
                _to_device(child, spec_modules, dev)

    warnings.warn('By default, offloading will be done on '
                  '`nn.Linear`. You can add modules which want offload to '
                  'the `lmdeploy.lite.defaults.OFFLOAD_MOD`.')

    _to_device(model, target_modules, device)

    handles: List[Any] = []
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
        handles.clear()
        model.to('cpu')
        torch.cuda.empty_cache()


@contextmanager
def memory_efficient_inference(model: nn.Module, offload: bool = True, device: str = 'cuda') -> Generator[None, None, None]:
    """Memory efficient inference context manager.

    Moves model to device for inference, with option to offload
    specific modules.

    Args:
        model: Model for inference
        offload: Whether to offload modules
        device: Device for inference

    Yields:
        None
    """
    if device not in ('cpu', 'cuda'):
        raise ValueError(f'device must be "cpu" or "cuda", got {device}')

    if offload:
        warnings.warn('Using offload mode - modules defined in OFFLOAD_MOD '
                      'will be moved to GPU during forward pass only.')
        warnings.warn('Using offload mode will incur performance penalty due to '
                      'frequent CPU-GPU data transfers.')
        try:
            with torch.inference_mode():
                with offload_kv_cache(model, device):
                    with offload_weights(model, device):
                        yield
        except Exception as e:
            logger.error(f'Memory efficient inference failed with offload: {e}')
            raise
    else:
        try:
            model.to(device)
            with torch.inference_mode():
                yield
        except Exception as e:
            logger.error(f'Memory efficient inference failed: {e}')
            raise
