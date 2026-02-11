# Copyright (c) OpenMMLab. All rights reserved.

import inspect
from contextlib import contextmanager
from typing import Callable, List, MutableSequence, Union

import torch


@contextmanager
def disable_transformers_logging():
    import transformers
    from transformers.utils import logging
    previous_level = logging.get_verbosity()
    logging.set_verbosity(transformers.logging.ERROR)
    yield
    logging.set_verbosity(previous_level)


@contextmanager
def disable_logging():
    import logging
    previous_level = logging.root.manager.disable
    logging.disable(logging.ERROR)
    yield
    logging.disable(previous_level)


def _set_func(origin_func_path: Union[str, None], rewrite_func: Callable, origin_func: Callable = None):
    """Replace old function with the new function.

    Args:
        origin_func_path (str): original function path
        rewrite_func (Callable): function to replace with
        origin_func (Callable): function to replace
    """
    # import module
    if isinstance(origin_func_path, str):
        split_path = origin_func_path.split('.')
        for i in range(len(split_path), 0, -1):
            try:
                exec('import {}'.format('.'.join(split_path[:i])))
                break
            except Exception:
                continue

        origin_func = eval(origin_func_path) \
            if origin_func is None else origin_func

    method_class = inspect.ismethod(origin_func)

    # replace method
    if not method_class:
        import gc
        refs = gc.get_referrers(origin_func)
        obj_id = id(origin_func)
        for ref in refs:
            if isinstance(ref, dict):
                for x, y in ref.items():
                    if id(y) == obj_id:
                        ref[x] = rewrite_func
            elif isinstance(ref, MutableSequence):
                for i, v in enumerate(ref):
                    if id(v) == obj_id:
                        ref[i] = rewrite_func
    if isinstance(origin_func_path, str):
        exec(f'{origin_func_path} = rewrite_func')
    elif method_class:
        raise NotImplementedError

    return origin_func


@contextmanager
def rewrite_ctx(origin_func_path: List[Union[str, Callable]], rewrite_func: List[Callable]):
    """Rewrite context."""
    assert len(origin_func_path) == len(rewrite_func)
    origin_func_list = []
    for (func_path, dst_func) in zip(origin_func_path, rewrite_func):
        if isinstance(func_path, Callable):
            origin_func = _set_func(None, dst_func, func_path)
        else:
            origin_func = _set_func(func_path, dst_func)
        origin_func_list.append(origin_func)
    yield
    for (func_path, dst_func, origin_func) in zip(origin_func_path, rewrite_func, origin_func_list):
        if isinstance(func_path, Callable):
            _set_func(None, origin_func, dst_func)
        else:
            _set_func(func_path, origin_func, dst_func)


def add_device_hook(module: torch.nn.Module, device: torch.device, fn: Callable = None):
    """Add device hook."""
    from accelerate.hooks import ModelHook, add_hook_to_module

    class ToDevice(ModelHook):
        """ToDevice hook."""

        def __init__(self, device):
            self.device = device

        def post_forward(self, module, output):
            if fn is not None:
                output = fn(output)
            else:
                output = output.to(device=self.device)
            return output

    add_hook_to_module(module=module, hook=ToDevice(device=device), append=True)
