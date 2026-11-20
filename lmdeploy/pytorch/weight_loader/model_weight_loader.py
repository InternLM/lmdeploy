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


def new_ct_normalize_state() -> dict:
    """Create staging state that lets one linear's tensors split across
    streams.

    Pass the returned dict to consecutive ``normalize_ct_weights`` calls so a
    module whose ``weight_packed``/``weight_scale`` land in different
    safetensors shards (or update buckets) still stage together. Finish the
    sequence with ``finalize_ct_weights``; a state passed this way does NOT
    validate leftovers at the end of a single stream.
    """
    return {'staged': {}, 'converted': set()}


def finalize_ct_weights(state: dict):
    """Reject staged leftovers once every stream of a state was consumed."""
    if state['staged']:
        raise ValueError('Incomplete compressed-tensors shards for: '
                         f'{sorted(state["staged"])}; check the checkpoint integrity.')


def _validate_weight_shape(prefix: str, shape_tensor: torch.Tensor, packed: torch.Tensor):
    """Reject a ``weight_shape`` record inconsistent with the packed tensor.

    ct exports carry the unpacked weight geometry as ``weight_shape``; its
    orientation varies by exporter, so either dimension order may match the
    packed int32 layout ``[out_features, in_features // 8]``.
    """
    try:
        dims = (int(shape_tensor[0]), int(shape_tensor[1]))
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError(f'{prefix}.weight_shape must hold the unpacked [k, n] geometry') from e
    out_features, words = packed.shape
    k_total = words * 8
    if dims not in ((k_total, out_features), (out_features, k_total)):
        raise ValueError(f'{prefix}: weight_shape {dims} does not match packed weights {tuple(packed.shape)}')


def normalize_ct_weights(iterator, state: dict | None = None):
    """Normalize compressed-tensors W4A16 tensors to the canonical AWQ layout.

    Detection works on checkpoint key suffixes, so checkpoints that carry no
    ct keys pass through untouched. Tensors of one linear are staged until the
    mandatory ``weight_packed``/``weight_scale`` pair is complete, then
    converted in one ``convert_shard`` call. Peak staging memory stays at the
    single-module bound documented in ``ct_repack``.

    Per-linear ``weight_shape`` metadata is validated and consumed: the AWQ
    modules have no such parameter, and forwarding it would fail the load.

    A ``weight_zero_point`` that arrives after its module converted must
    carry the symmetric constant; anything else means an asymmetric scheme
    the conversion does not support. Staging leftovers at the end mean a
    corrupt checkpoint and raise — with a locally owned state. Pass a shared
    ``state`` (see ``new_ct_normalize_state``) to keep staging alive across
    calls; the caller then owns validation through ``finalize_ct_weights``.
    Staged tensors always own their storage: hot-update buckets may deliver
    views into reused IPC buffers.
    """
    if state is None:
        state = new_ct_normalize_state()
        owns_state = True
    else:
        owns_state = False
    staged = state['staged']
    converted = state['converted']
    # Import stays local: this module loads early during engine startup, and
    # the ct path needs the conversion library only.
    from lmdeploy.turbomind import ct_repack

    for name, tensor in iterator:
        suffix = next((s for s in _CT_TO_AWQ_SUFFIX if name.endswith(s)), None)
        if suffix is None:
            if name.endswith('weight_shape'):
                prefix = name[:-len('weight_shape')]
                if prefix in converted:
                    # Metadata trailing a converted module carries nothing new.
                    continue
                if not owns_state:
                    tensor = tensor.clone()
                shard = staged.setdefault(prefix, {})
                packed = shard.get('weight_packed')
                if packed is not None:
                    _validate_weight_shape(prefix, tensor, packed)
                shard['weight_shape'] = tensor
                continue
            yield name, tensor
            continue
        prefix = name[:-len(suffix)]
        if prefix in converted:
            # The module already converted. The symmetric constant zero
            # point carries no extra information, so it may still arrive.
            if suffix == 'weight_zero_point' and bool((tensor == ct_repack.CT_SYMMETRIC_ZERO).all()):
                continue
            raise ValueError(f'Duplicate compressed-tensors key {name!r} after {prefix} converted.')
        if not owns_state:
            tensor = tensor.clone()
        shard = staged.setdefault(prefix, {})
        if suffix == 'weight_packed' and 'weight_shape' in shard:
            _validate_weight_shape(prefix, shard['weight_shape'], tensor)
        shard[suffix] = tensor
        if 'weight_packed' not in shard or 'weight_scale' not in shard:
            continue

        shard.pop('weight_shape', None)
        out_shard, _ = ct_repack.convert_shard(shard)
        staged.pop(prefix)
        converted.add(prefix)
        for awq_name, awq_tensor in out_shard.items():
            yield f'{prefix}{awq_name}', awq_tensor
    if owns_state and staged:
        raise ValueError('Incomplete compressed-tensors shards for: '
                         f'{sorted(staged)}; check the checkpoint integrity.')


_GPTQ_SUFFIXES = ('qweight', 'scales', 'qzeros')


def new_gptq_normalize_state() -> dict:
    """Create staging state that lets one linear's tensors split across
    streams.

    Pass the returned dict to consecutive ``normalize_gptq_weights`` calls so
    a module whose quant tensors land in different safetensors shards still
    stages together. Finish the sequence with ``finalize_gptq_weights``; a
    state passed this way does NOT flush or validate at the end of a single
    stream.
    """
    return {'staged': {}, 'converted': set()}


def _gptq_convert_one(prefix, shard):
    """Decide the layout of one module and emit canonical AWQ tensors."""
    from lmdeploy.turbomind import gptq_repack

    qw, sc = shard['qweight'], shard['scales']
    is_gptq = qw.shape[-1] == sc.shape[-1]
    is_awq = qw.shape[-1] * 8 == sc.shape[-1]
    if not is_gptq and not is_awq:
        raise ValueError(
            f'Cannot classify quant tensors at prefix {prefix!r}: qweight {tuple(qw.shape)} vs scales '
            f'{tuple(sc.shape)}; neither the GPTQ nor the AWQ pack geometry matches.')
    if is_awq:
        for s in _GPTQ_SUFFIXES:
            if s in shard:
                yield f'{prefix}{s}', shard[s]
        return
    out_shard, _ = gptq_repack.convert_shard(shard)
    for s, t in out_shard.items():
        yield f'{prefix}{s}', t


def _gptq_flush_staged(staged, converted, keep):
    """Flush every staged module except ``keep``; raise on strays."""
    for prefix, shard in list(staged.items()):
        if prefix == keep:
            continue
        if 'scales' not in shard:
            raise ValueError(f'Quant shard for {prefix!r} has no scales; '
                             'check the checkpoint integrity.')
        staged.pop(prefix)
        converted.add(prefix)
        yield from _gptq_convert_one(prefix, shard)


def finalize_gptq_weights(state: dict):
    """Flush staged modules once every stream of the state was consumed.

    Mirrors the end-of-stream behavior of a locally owned normalizer:
    modules missing ``qzeros`` flush with the synthesized symmetric pad,
    strays without ``scales`` raise. The flushed tensors must still be
    loaded, so this is a generator; consume it fully — consuming is also
    what triggers the stray validation.
    """
    yield from _gptq_flush_staged(state['staged'], state['converted'], keep=None)


def normalize_gptq_weights(iterator, state: dict | None = None):
    """Normalize GPTQ W4A16 tensors to the canonical AWQ layout.

    GPTQ and AWQ checkpoints share the ``qweight``/``scales``/``qzeros``
    suffixes but pack differently: GPTQ packs along K with sequential
    nibbles and a ``zero_point - 1`` offset, AWQ packs along N with the
    ``[0, 4, 1, 5, 2, 6, 3, 7]`` nibble order. The discriminator is the
    qwidth/scales relation, which is decisive for any non-empty layer:
    GPTQ ``qweight.shape[-1] == scales.shape[-1]``, AWQ
    ``qweight.shape[-1] * 8 == scales.shape[-1]``.

    Tensors of one linear are staged until the full triple arrives, then
    GPTQ modules convert in one ``gptq_to_awq`` call; AWQ modules pass
    through in arrival order. A module that never ships ``qzeros`` flushes
    with a synthesized symmetric zero word at the end of the complete
    stream — exporters only drop qzeros for symmetric groups, where the
    AWQ symmetric pad is the exact value. Staging never synthesizes early:
    a module whose tensors straddle shard or bucket boundaries stays staged
    until its components arrive. Peak staging memory stays at the
    interleaving bound of the stream.

    With a locally owned state the stream end flushes and validates.
    Pass a shared ``state`` (see ``new_gptq_normalize_state``) to keep
    staging alive across calls; the caller then flushes the leftovers
    through ``finalize_gptq_weights``.
    """
    if state is None:
        state = new_gptq_normalize_state()
        owns_state = True
    else:
        owns_state = False
    staged = state['staged']
    converted = state['converted']
    from lmdeploy.turbomind import gptq_repack
    from lmdeploy.turbomind.gptq_repack import _is_sequential_g_idx

    for name, tensor in iterator:
        if name.endswith('g_idx'):
            # AutoGPTQ ships an identity group-index buffer even without
            # act-order; consume it. A reordered mapping is real act-order
            # and cannot be served by the AWQ layout.
            if _is_sequential_g_idx(tensor):
                continue
            raise ValueError(f'{name}: desc_act (act-order) GPTQ checkpoints are not supported; '
                             're-quantize the checkpoint without desc_act.')
        suffix = next((s for s in _GPTQ_SUFFIXES if name.endswith(s)), None)
        if suffix is None:
            yield name, tensor
            continue
        prefix = name[:-len(suffix)]
        if prefix in converted:
            # The module already converted. A symmetric GPTQ checkpoint
            # stores the constant 7 (= zero point 8 minus one), which the
            # synthesized pad already represents.
            if suffix == 'qzeros' and bool((gptq_repack.unpack_u4_lsb(tensor) == 7).all()):
                continue
            raise ValueError(f'Duplicate quant key {name!r} after {prefix} converted.')
        shard = staged.setdefault(prefix, {})
        if not owns_state:
            # Update buckets deliver views into reused IPC buffers; a module
            # whose scales land in a later bucket would convert from a stale
            # view. The ct normalizer makes the same defensive copy.
            tensor = tensor.clone()
        shard[suffix] = tensor
        if len(shard) == len(_GPTQ_SUFFIXES):
            staged.pop(prefix)
            converted.add(prefix)
            yield from _gptq_convert_one(prefix, shard)
    if owns_state:
        yield from _gptq_flush_staged(staged, converted, keep=None)


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
        normalize_quant: bool = False,
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
        # One staging state per normalizer spans all shard files: a linear's
        # quant tensors may split across shard boundaries.
        ct_state = new_ct_normalize_state()
        gptq_state = new_gptq_normalize_state()
        for path in tqdm(paths, desc='Loading weights from safetensors', disable=disable_tqdm):
            weights_iterator = self._get_weights_iterator(path)
            if normalize_quant:
                weights_iterator = normalize_gptq_weights(
                    normalize_ct_weights(weights_iterator, ct_state), gptq_state)
            weights_iterator = self._rename_weights_iterator(weights_iterator, model)
            if len(dummy_prefix) > 0:
                weights_iterator = self._skip_dummy_iterator(weights_iterator, dummy_prefix)
            model.load_weights(weights_iterator)
        if normalize_quant:
            # Strays raise here; flushed leftovers still feed the model
            # through the same rename/dummy pipeline as the shard streams.
            leftover = list(finalize_gptq_weights(gptq_state))
            if leftover:
                leftover = self._rename_weights_iterator(iter(leftover), model)
                if len(dummy_prefix) > 0:
                    leftover = self._skip_dummy_iterator(leftover, dummy_prefix)
                model.load_weights(leftover)
            finalize_ct_weights(ct_state)
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
                       normalize_quant: bool = False):
    """Loading model weights."""
    loader = ModelWeightLoader(checkpoint_path, prefix=prefix)
    loader.load_model_weights(model, device=device, normalize_quant=normalize_quant)
    model.eval()
    process_weights_after_loading(model)
