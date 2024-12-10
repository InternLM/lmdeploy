# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
from typing import List

import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm
from transformers.utils import (SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME,
                                WEIGHTS_INDEX_NAME, WEIGHTS_NAME)

from lmdeploy.pytorch.distributed import get_world_rank
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


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


def _get_safetensors_weights_iterator(hf_files: List[str], disable_tqdm: bool):
    """get safeternsors weights iterator."""
    for file in tqdm(hf_files,
                     desc='Loading weights from safetensors',
                     disable=disable_tqdm):
        with safe_open(file, framework='pt') as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param


def _get_pt_weights_iterator(hf_files: List[str], disable_tqdm: bool):
    """get pt weights iterator."""
    for file in tqdm(hf_files,
                     desc='Loading weights from pt ckpt',
                     disable=disable_tqdm):
        state = torch.load(file, weights_only=True, map_location='cpu')
        yield from state.items()
        del state
        torch.cuda.empty_cache()


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

    def _get_weights_iterator(self, paths: List[str], disable_tqdm: bool):
        """get weights iterator."""
        if self._weight_type == 'safetensors':
            weights_iterator = _get_safetensors_weights_iterator(
                paths, disable_tqdm)
        else:
            weights_iterator = _get_pt_weights_iterator(paths, disable_tqdm)
        if self._prefix is not None:
            weights_iterator = ((self._prefix + name, tensor)
                                for name, tensor in weights_iterator)
        return weights_iterator

    def load_model_weights(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
    ):
        """load model weights implementation."""
        assert hasattr(model, 'load_weights')
        paths = self._shard_paths
        _, rank = get_world_rank()
        disable_tqdm = rank != 0
        weights_iterator = self._get_weights_iterator(paths, disable_tqdm)
        model.load_weights(weights_iterator)
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
