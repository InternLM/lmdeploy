# Copyright (c) OpenMMLab. All rights reserved.
import os
from contextlib import contextmanager
from typing import Callable, Dict, List, MutableSequence

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME,
                                WEIGHTS_INDEX_NAME, is_safetensors_available)
from transformers.utils.hub import get_checkpoint_shard_files


def load_weight_ckpt(ckpt: str) -> Dict[str, torch.Tensor]:
    """load checkpoint."""
    if ckpt.endswith('.safetensors'):
        return load_file(ckpt)
    else:
        return torch.load(ckpt)


def get_used_weight_files(folder: str,
                          state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """get used checkpoint which contains keys in state_dict."""
    _index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    _safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(_index_file):
        index_file = _index_file
    elif os.path.exists(_safe_index_file):
        index_file = _safe_index_file
    elif is_safetensors_available() and os.path.isfile(
            os.path.join(folder, SAFE_WEIGHTS_NAME)):  # Single safetensor file
        return [os.path.join(folder, SAFE_WEIGHTS_NAME)]
    else:
        raise FileNotFoundError
    _, sharded_metadata = get_checkpoint_shard_files(folder, index_file)
    potential_keys = set(state_dict.keys())
    supplied_keys = set(sharded_metadata['weight_map'].keys())
    shared_keys = potential_keys & supplied_keys
    valid_files = set(sharded_metadata['weight_map'][k] for k in shared_keys)
    return valid_files


def load_model_from_weight_files(model: nn.Module, folder: str) -> None:
    """load nn.Module weight from folder."""
    valid_files = get_used_weight_files(folder, model.state_dict())
    for file_name in valid_files:
        ckpt = os.path.join(folder, file_name)
        state_dict = load_weight_ckpt(ckpt)
        model.load_state_dict(state_dict, strict=False)


@contextmanager
def disable_transformers_logging():
    import transformers
    from transformers.utils import logging
    previous_level = logging.get_verbosity()
    logging.set_verbosity(transformers.logging.ERROR)
    yield
    logging.set_verbosity(previous_level)


@contextmanager
def hack_import_with(src: List[str], dst: str = 'torch'):
    """Replace wanted and uninstalled package with a dummy one.

    Args:
        src (List): a list of package name
        dst (str): dummy package name. Default to 'torch'.
    """
    import sys
    from importlib.util import find_spec
    not_installed = []
    for item in src:
        if not find_spec(item):
            not_installed.append(item)
            sys.modules[item] = __import__(dst)
    yield
    for item in not_installed:
        sys.modules.pop(item, None)


def _set_func(origin_func_path: str,
              rewrite_func: Callable,
              origin_func: Callable = None):
    """Replace old function with the new function.

    Args:
        origin_func_path (str): original function path
        rewrite_func (Callable): function to replace with
        origin_func (Callable): function to replace
    """
    # import module
    split_path = origin_func_path.split('.')
    for i in range(len(split_path), 0, -1):
        try:
            exec('import {}'.format('.'.join(split_path[:i])))
            break
        except Exception:
            continue

    method_class = False
    if len(split_path) > 1:
        module_or_class = eval('.'.join(split_path[:-1]))
        if isinstance(module_or_class, type):
            method_class = True

    origin_func = eval(origin_func_path) \
        if origin_func is None else origin_func

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
    exec(f'{origin_func_path} = rewrite_func')
    return origin_func


@contextmanager
def rewrite_ctx(origin_func_path: List[str], rewrite_func: List[Callable]):
    """rewrite context."""
    assert len(origin_func_path) == len(rewrite_func)
    origin_func_list = []
    for (func_path, dst_func) in zip(origin_func_path, rewrite_func):
        origin_func = _set_func(func_path, dst_func)
        origin_func_list.append(origin_func)
    yield
    for (func_path, dst_func, origin_func) in zip(origin_func_path,
                                                  rewrite_func,
                                                  origin_func_list):
        _set_func(func_path, origin_func, dst_func)
