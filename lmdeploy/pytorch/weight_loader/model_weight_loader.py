# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from contextlib import ExitStack, contextmanager
from typing import Dict

import torch
from transformers.modeling_utils import load_state_dict
from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME,
                                WEIGHTS_INDEX_NAME, WEIGHTS_NAME)

from lmdeploy.utils import get_logger

from .adapter_weight_loader import AdapterWeightLoader

logger = get_logger('lmdeploy')


def _get_weight_type(model_path: str, use_safetensors: bool = None):
    """get weight type."""
    weight_type = None
    is_sharded = False
    if use_safetensors is not False and osp.isfile(
            osp.join(model_path, SAFE_WEIGHTS_NAME)):
        # Load from a safetensors checkpoint
        weight_type = 'safetensors'
    elif use_safetensors is not False and osp.isfile(
            osp.join(model_path, SAFE_WEIGHTS_INDEX_NAME)):
        # Load from a sharded safetensors checkpoint
        weight_type = 'safetensors'
        is_sharded = True
    elif osp.isfile(osp.join(model_path, WEIGHTS_NAME)):
        # Load from a PyTorch checkpoint
        weight_type = 'pytorch'
    elif osp.isfile(osp.join(model_path, WEIGHTS_INDEX_NAME)):
        # Load from a sharded PyTorch checkpoint
        weight_type = 'pytorch'
        is_sharded = True
    else:
        raise RuntimeError('Unknown weight type.')

    return (weight_type, is_sharded)


def _get_weight_map(model_path: str, weight_type: str):
    """get weight index."""
    if weight_type == 'safetensors':
        load_index = osp.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    elif weight_type == 'pytorch':
        load_index = osp.join(model_path, WEIGHTS_INDEX_NAME)
    else:
        raise RuntimeError('Unknown weight type.')

    with open(load_index, mode='r', encoding='utf-8') as f:
        index = json.load(f)

    weight_map = index['weight_map']
    return weight_map


def _get_weight_path(model_path: str, weight_type: str):
    """get weight path."""
    if weight_type == 'safetensors':
        weight_name = SAFE_WEIGHTS_NAME
    elif weight_type == 'pytorch':
        weight_name = WEIGHTS_NAME
    else:
        raise RuntimeError('Unknown weight type.')

    weight_path = osp.join(model_path, weight_name)
    return weight_path, weight_name


class ModelWeightLoader:
    """model weight loader for sharded weights."""

    def __init__(self, model_path: str, adapters: Dict[str, str] = None):
        self.model_path = model_path
        self._state_dict = dict()
        weight_type, is_sharded = _get_weight_type(model_path)

        self._weight_type = weight_type
        self._is_sharded = is_sharded
        self._prefix = ''
        if is_sharded:
            self._weight_map = _get_weight_map(model_path, weight_type)
        else:
            weight_path, weight_name = _get_weight_path(
                model_path, weight_type)
            self._load_shard(weight_path)
            keys = list(self._state_dict.keys())
            self._weight_map = dict((k, weight_name) for k in keys)

        if adapters is None:
            adapters = dict()

        self._adapter_loaders: Dict[str, AdapterWeightLoader] = dict()
        for ada_name, ada_path in adapters.items():
            ada_loader = AdapterWeightLoader(ada_name, ada_path)
            self._adapter_loaders[ada_name] = ada_loader

    def _load_shard(self, path: str):
        """load shards."""
        self._state_dict.update(load_state_dict(path))

    def _load_shard_for_key(self, key: str):
        """load shard for key."""
        if key in self._state_dict:
            return
        if key not in self._weight_map:
            raise RuntimeError(f'Unknown weight: {key}.')

        shard_file = osp.join(self.model_path, self._weight_map[key])
        self._load_shard(shard_file)
        if key not in self._state_dict:
            raise RuntimeError(f'Can not found "{key}" in "{shard_file}"')

    def pop(self, key: str):
        """pop weight."""
        key = self._prefix + key
        self._load_shard_for_key(key)
        return self._state_dict.pop(key)

    def get(self, key: str):
        """get weight."""
        key = self._prefix + key
        self._load_shard_for_key(key)
        return self._state_dict.get(key)

    def adapter(self, key: str):
        """get adapter loader."""
        if key not in self._adapter_loaders:
            raise RuntimeError(f'Unknown adapter: {key}')
        return self._adapter_loaders[key]

    @contextmanager
    def prefix_context(self, mod_name: str):
        """update prefix by mod name."""
        old_prefix = self._prefix
        if len(old_prefix) == 0:
            new_prefix = f'{mod_name}.'
        else:
            new_prefix = f'{old_prefix}{mod_name}.'
        self._prefix = new_prefix

        with ExitStack() as stack:
            for ada in self._adapter_loaders.values():
                stack.enter_context(ada.prefix_context(mod_name))
            yield new_prefix
        self._prefix = old_prefix


def _load_model_weights_impl(model: torch.nn.Module,
                             loader: ModelWeightLoader,
                             rank: int = 0,
                             world_size: int = 1,
                             device: torch.device = 'cpu'):
    """load model weights implementation."""

    def __load_no_recursive(mod: torch.nn.Module):
        """load no recursive."""
        for name, _ in mod.named_parameters(recurse=False):
            param = loader.pop(name)
            mod.register_parameter(
                name, torch.nn.Parameter(param, requires_grad=False))
        for name, _ in mod.named_buffers(recurse=False):
            buf = loader.pop(name)
            mod.register_buffer(name, buf)

    if hasattr(model, '_load_weights'):
        model._load_weights(loader, rank, world_size)
    else:
        __load_no_recursive(model)
        for name, child in model.named_children():
            with loader.prefix_context(name):
                _load_model_weights_impl(child,
                                         loader,
                                         rank=rank,
                                         world_size=world_size,
                                         device=device)

    model.to(device)


@torch.inference_mode()
def load_model_weights(model: torch.nn.Module,
                       checkpoint_path: str,
                       adapters: Dict[str, str] = None,
                       rank: int = 0,
                       world_size: int = 1,
                       device: torch.device = 'cpu'):
    """Loading model weights.

    Please waiting.
    """
    if rank == 0:
        logger.info('Loading model weights.')
    if adapters is None:
        adapters = dict()
    loader = ModelWeightLoader(checkpoint_path, adapters=adapters)
    _load_model_weights_impl(model,
                             loader,
                             rank=rank,
                             world_size=world_size,
                             device=device)
