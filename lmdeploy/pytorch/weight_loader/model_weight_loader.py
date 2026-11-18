# Copyright (c) OpenMMLab. All rights reserved.

import json
import os.path as osp

import numpy as np
import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.distributed import get_world_rank
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def load_weight(param: torch.nn.Parameter, loaded_weight: torch.Tensor, **kwargs):
    """Load weight."""
    if hasattr(param, 'weight_loader'):
        param.weight_loader(param, loaded_weight, **kwargs)
    else:
        assert len(kwargs) == 0
        default_weight_loader(param, loaded_weight)


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    """Default weight loader."""
    if param.numel() == 1 and loaded_weight.numel() == 1:
        param.data.fill_(loaded_weight.item())
    else:
        assert param.size() == loaded_weight.size(), (f'Attempted to load weight ({loaded_weight.size()}) '
                                                      f'into parameter ({param.size()})')
        param.data.copy_(loaded_weight)


# compressed-tensors pack-quantized checkpoint suffixes mapped to the
# canonical AutoAWQ names the AWQ linear family consumes.
_CT_TO_AWQ_SUFFIX = {
    'weight_packed': 'qweight',
    'weight_scale': 'scales',
    'weight_zero_point': 'qzeros',
}


def normalize_ct_weights(iterator):
    """Normalize compressed-tensors W4A16 tensors to the canonical AWQ layout.

    Detection works on checkpoint key suffixes, so checkpoints that carry no
    ct keys pass through untouched. Tensors of one linear are staged until the
    mandatory ``weight_packed``/``weight_scale`` pair is complete, then
    converted in one ``convert_shard`` call. Peak staging memory stays at the
    single-module bound documented in ``ct_repack``.

    A ``weight_zero_point`` that arrives after its module converted must
    carry the symmetric constant; anything else means an asymmetric scheme
    the conversion does not support. Staging leftovers at the end mean a
    corrupt checkpoint and raise.
    """
    staged: dict[str, dict[str, torch.Tensor]] = {}
    converted: set[str] = set()
    # Import stays local: this module loads early during engine startup, and
    # the ct path needs the conversion library only.
    from lmdeploy.turbomind import ct_repack

    for name, tensor in iterator:
        suffix = next((s for s in _CT_TO_AWQ_SUFFIX if name.endswith(s)), None)
        if suffix is None:
            yield name, tensor
            continue
        prefix = name[:-len(suffix)]
        if prefix in converted:
            # The module already converted. The symmetric constant zero
            # point carries no extra information, so it may still arrive.
            if suffix == 'weight_zero_point' and bool((tensor == ct_repack.CT_SYMMETRIC_ZERO).all()):
                continue
            raise ValueError(f'Duplicate compressed-tensors key {name!r} after {prefix} converted.')
        shard = staged.setdefault(prefix, {})
        shard[suffix] = tensor
        if 'weight_packed' not in shard or 'weight_scale' not in shard:
            continue

        out_shard, _ = ct_repack.convert_shard(shard)
        staged.pop(prefix)
        converted.add(prefix)
        for awq_name, awq_tensor in out_shard.items():
            yield f'{prefix}{awq_name}', awq_tensor
    if staged:
        raise ValueError('Incomplete compressed-tensors shards for: '
                         f'{sorted(staged)}; check the checkpoint integrity.')


def _get_weight_type(model_path: str, use_safetensors: bool = None):
    """Get weight type."""
    weight_type = None
    is_sharded = False
    if use_safetensors is not False and osp.isfile(osp.join(model_path, SAFE_WEIGHTS_NAME)):
        # Load from a safetensors checkpoint
        weight_type = 'safetensors'
    elif use_safetensors is not False and osp.isfile(osp.join(model_path, SAFE_WEIGHTS_INDEX_NAME)):
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
    """Get weight index."""
    if weight_type == 'safetensors':
        load_index = osp.join(model_path, SAFE_WEIGHTS_INDEX_NAME)
    elif weight_type == 'pytorch':
        load_index = osp.join(model_path, WEIGHTS_INDEX_NAME)
    else:
        raise RuntimeError(f'Unsupported weight type: {weight_type}.')

    with open(load_index, encoding='utf-8') as f:
        index = json.load(f)

    weight_map = index['weight_map']
    return weight_map


def _get_weight_path(model_path: str, weight_type: str):
    """Get weight path."""
    if weight_type == 'safetensors':
        weight_name = SAFE_WEIGHTS_NAME
    elif weight_type == 'pytorch':
        weight_name = WEIGHTS_NAME
    else:
        raise RuntimeError('Unknown weight type.')

    weight_path = osp.join(model_path, weight_name)
    return weight_path, weight_name


def _get_safetensors_weights_iterator(file: str, prefix: str):
    """Get safeternsors weights iterator."""
    with safe_open(file, framework='pt') as f:
        for name in f.keys():
            param = f.get_tensor(name)
            if prefix is not None:
                name = f'{prefix}{name}'
            yield name, param


def _get_pt_weights_iterator(file: str, prefix: str):
    """Get pt weights iterator."""
    state = torch.load(file, weights_only=True, map_location='cpu')
    try:
        if prefix is None:
            yield from state.items()
        else:
            for k, v in state.items():
                yield f'{prefix}{k}', v
    finally:
        del state
        torch.cuda.empty_cache()


class ModelWeightLoader:
    """Model weight loader for sharded weights."""

    def __init__(self, model_path: str, prefix: str = None):
        self.model_path = model_path
        weight_type, is_sharded = _get_weight_type(model_path)

        self._weight_type = weight_type
        self._is_sharded = is_sharded
        self._prefix = prefix
        self._shard_paths = self._get_shard_paths(model_path, is_sharded, weight_type)

    @staticmethod
    def _get_shard_paths(model_path: str, is_sharded: bool, weight_type: str):
        """Get shard paths."""
        if is_sharded:
            weight_map = _get_weight_map(model_path, weight_type)
            paths = set(weight_map.values())
            paths = tuple(f'{model_path}/{path}' for path in paths)
            return paths
        else:
            path, _ = _get_weight_path(model_path, weight_type)
            return (path, )

    def _get_weights_iterator(self, path: str):
        """Get weights iterator."""
        if self._weight_type == 'safetensors':
            weights_iterator = _get_safetensors_weights_iterator(path, self._prefix)
        else:
            weights_iterator = _get_pt_weights_iterator(path, self._prefix)
        return weights_iterator

    @staticmethod
    def _skip_dummy_iterator(iterator, dummy_prefix: list):
        """Wrap iterator to skip dummy weights."""
        for name, param in iterator:
            if not any(name.startswith(prefix) for prefix in dummy_prefix):
                yield name, param

    @staticmethod
    def _rename_weights_iterator(iterator, model: torch.nn.Module):
        """Wrap iterator to rename weights."""
        rename_func = getattr(model, 'rename_weight', lambda x: x)
        for name, param in iterator:
            new_name = rename_func(name)
            yield new_name, param

    def load_model_weights(
        self,
        model: torch.nn.Module,
        device: torch.device = None,
        normalize_ct: bool = False,
    ):
        """Load model weights implementation."""
        assert hasattr(model, 'load_weights')
        paths = self._shard_paths
        _, rank = get_world_rank()
        disable_tqdm = rank != 0

        # get dummy prefix
        dummy_prefix = []
        for name, mod in model.named_modules():
            if getattr(mod, '_is_dummy_mod', False):
                dummy_prefix.append(f'{name}.')

        paths = sorted(paths)
        if _envs.random_load_weight:
            np.random.shuffle(paths)
        for path in tqdm(paths, desc='Loading weights from safetensors', disable=disable_tqdm):
            weights_iterator = self._get_weights_iterator(path)
            if normalize_ct:
                weights_iterator = normalize_ct_weights(weights_iterator)
            weights_iterator = self._rename_weights_iterator(weights_iterator, model)
            if len(dummy_prefix) > 0:
                weights_iterator = self._skip_dummy_iterator(weights_iterator, dummy_prefix)
            model.load_weights(weights_iterator)
        if device is not None:
            model.to(device)


@torch.inference_mode()
def process_weights_after_loading(model: torch.nn.Module):
    """Run module weight updates followed by dependent post-load hooks."""
    for _, mod in model.named_modules():
        if not hasattr(mod, 'update_weights'):
            continue
        mod.update_weights()

    # Keep this as a second pass. Standard loading creates online-FP8 weights
    # and scales, then update_weights() finalizes their backend representation.
    # MLA can only absorb kv_b_proj into kc/vc after both steps are complete.
    for _, mod in model.named_modules():
        if not hasattr(mod, 'process_weights_after_loading'):
            continue
        mod.process_weights_after_loading()


@torch.inference_mode()
def load_model_weights(model: torch.nn.Module,
                       checkpoint_path: str,
                       prefix: str = None,
                       device: torch.device = None,
                       normalize_ct: bool = False):
    """Loading model weights."""
    loader = ModelWeightLoader(checkpoint_path, prefix=prefix)
    loader.load_model_weights(model, device=device, normalize_ct=normalize_ct)
    model.eval()
    process_weights_after_loading(model)
