# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from contextlib import contextmanager
from typing import Dict, Iterator, List, Union

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
def add_sys_path(path: Union[str, os.PathLike]) -> Iterator[None]:
    """Temporarily add the given path to `sys.path`."""
    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


@contextmanager
def disable_transformers_logging():
    import transformers
    from transformers.utils import logging
    previous_level = logging.get_verbosity()
    logging.set_verbosity(transformers.logging.ERROR)
    yield
    logging.set_verbosity(previous_level)
