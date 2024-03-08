# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
from safetensors.torch import load_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME
from transformers.utils.hub import get_checkpoint_shard_files


def load_weight_ckpt(ckpt: str):
    """load checkpoint."""
    if ckpt.endswith('.safetensors'):
        return load_file(ckpt)
    else:
        return torch.load(ckpt)


def get_used_weight_files(folder, state_dict):
    """get used checkpoint which contains keys in state_dict."""
    _index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    _safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.exists(_index_file):
        index_file = _index_file
    elif os.path.exists(_safe_index_file):
        index_file = _safe_index_file
    else:
        raise FileNotFoundError
    _, sharded_metadata = get_checkpoint_shard_files(folder, index_file)
    potential_keys = set(state_dict.keys())
    supplied_keys = set(sharded_metadata['weight_map'].keys())
    shared_keys = potential_keys & supplied_keys
    valid_files = set(sharded_metadata['weight_map'][k] for k in shared_keys)
    return valid_files


def load_model_from_weight_files(model, folder):
    """load nn.Module weight from folder."""
    valid_files = get_used_weight_files(folder, model.state_dict())
    for file_name in valid_files:
        ckpt = os.path.join(folder, file_name)
        state_dict = load_weight_ckpt(ckpt)
        model.load_state_dict(state_dict, strict=False)
