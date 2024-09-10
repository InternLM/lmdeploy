# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp

import torch
import torch.distributed as dist
from transformers.modeling_utils import load_state_dict
from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME,
                                WEIGHTS_INDEX_NAME, WEIGHTS_NAME)

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _get_world_rank():
    """get rank."""
    rank = 0
    world_size = 1
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return world_size, rank


def load_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor,
                **kwargs):
    """load weight."""
    if hasattr(param, 'weight_loader'):
        param.weight_loader(param, loaded_weight, **kwargs)
    else:
        assert len(kwargs) == 0
        default_weight_loader(param, loaded_weight)


def default_weight_loader(param: torch.nn.Parameter,
                          loaded_weight: torch.Tensor):
    """default weight loader."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        assert param.size() == loaded_weight.size(), (
            f'Attempted to load weight ({loaded_weight.size()}) '
            f'into parameter ({param.size()})')
        param.data.copy_(loaded_weight)


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
        raise RuntimeError(f'Unsupported weight type: {weight_type}.')

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

    def __init__(self, model_path: str, prefix: str = None):
        self.model_path = model_path
        weight_type, is_sharded = _get_weight_type(model_path)

        self._weight_type = weight_type
        self._is_sharded = is_sharded
        self._prefix = prefix
        self._shard_paths = self._get_shard_paths(model_path, is_sharded,
                                                  weight_type)

    @staticmethod
    def _get_shard_paths(model_path: str, is_sharded: bool, weight_type: str):
        """get shard paths."""
        if is_sharded:
            weight_map = _get_weight_map(model_path, weight_type)
            paths = set(weight_map.values())
            paths = tuple(f'{model_path}/{path}' for path in paths)
            return paths
        else:
            path, _ = _get_weight_path(model_path, weight_type)
            return (path, )

    def _load_shard(self, path: str):
        """load shards."""
        state_dict = load_state_dict(path)
        if self._prefix is not None:
            state_dict = dict(
                (f'{self._prefix}{k}', v) for k, v in state_dict.items())
        return state_dict

    def load_model_weights(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
    ):
        """load model weights implementation."""
        assert hasattr(model, 'load_weights')
        paths = self._shard_paths
        world_size, rank = _get_world_rank()
        for path in paths:

            # log
            file_name = osp.split(path)[1]
            msg = f'loading weights - "{file_name}"'
            if world_size > 1:
                msg = f'rank[{rank}] {msg}'
            logger.info(msg)

            # process
            state_dict = self._load_shard(path)
            model.load_weights(state_dict.items())
        if device is not None:
            device = model.to(device)


@torch.inference_mode()
def load_model_weights(model: torch.nn.Module,
                       checkpoint_path: str,
                       prefix: str = None,
                       device: torch.device = None):
    """Loading model weights."""
    loader = ModelWeightLoader(checkpoint_path, prefix=prefix)
    loader.load_model_weights(model, device=device)
    model.eval()
    for _, mod in model.named_modules():
        if not hasattr(mod, 'update_weights'):
            continue
        mod.update_weights()
