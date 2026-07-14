# Copyright (c) OpenMMLab. All rights reserved.
"""Checkpoint storage abstraction and Prefix path navigation.

Replaces the flat ``dict[str, Tensor]`` interface that older turbomind
source-model code passed around. Source models receive a ``Prefix``,
do arithmetic on it (``+``, ``append``, ``slices``), and call
``get`` / ``has`` to read individual tensors.
"""
from __future__ import annotations

import json
import os.path as osp
from abc import ABC, abstractmethod
from collections.abc import Iterator
from glob import glob

import torch
from safetensors import safe_open

# https://github.com/huggingface/transformers/blob/53fad641cfdb5105e2470bcf3ef17ea8e25cc300/src/transformers/modeling_utils.py#L372
WEIGHT_INDEX_NAME = 'pytorch_model.bin.index.json'
WEIGHT_PATTERN = 'pytorch_model*.bin'
SAFE_WEIGHT_INDEX_NAME = 'model.safetensors.index.json'
SAFE_WEIGHT_PATTERN = 'model*.safetensors'
EXTRA_WEIGHT_PATTERNS = ['*.pt', '*.bin']
EXTRA_SAFE_WEIGHT_PATTERN = '*.safetensors'


class Prefix:
    """Path navigation overlay on a :class:`Checkpoint`.

    A ``Prefix`` carries a checkpoint reference plus a fully-qualified
    key prefix string (no trailing dot). Path arithmetic via ``+`` /
    ``append`` returns new ``Prefix`` objects. Tensor reads via
    ``get`` / ``has`` go through the underlying checkpoint.
    """

    __slots__ = ('ckpt', 'prefix')

    def __init__(self, ckpt: Checkpoint, prefix: str = ''):
        self.ckpt = ckpt
        self.prefix = prefix

    # ----- path navigation -----

    def __add__(self, key) -> Prefix:
        """``pfx + 'foo'`` -> Prefix at ``'parent.foo'`` (default '.'
        separator).

        ``key`` may be ``str`` or ``int``; ints are stringified.
        """
        return self.append(str(key))

    def append(self, name: str, sep: str = '.') -> Prefix:
        """Return a new Prefix with ``name`` appended via ``sep``.

        Empty current prefix or empty ``name`` skip the separator entirely.
        """
        return Prefix(self.ckpt, self._joined(name, sep))

    # ----- tensor access -----

    def get(self, name: str = '', sep: str = '.', *, index=None) -> torch.Tensor:
        """Read the tensor at ``self.prefix + sep + name``.

        Empty ``name`` reads the tensor at the exact prefix. Raises
        ``KeyError`` on miss (delegates to checkpoint).

        If ``index`` is not None, the checkpoint slices the tensor along
        dim 0 on CPU before transferring to GPU.
        """
        return self.ckpt.get(self._joined(name, sep), index=index)

    def has(self, name: str = '', sep: str = '.') -> bool:
        return self.ckpt.has(self._joined(name, sep))

    def pop(self, name: str = '', sep: str = '.', *, index=None) -> torch.Tensor:
        """Read and remove the tensor at ``self.prefix + sep + name``.

        Raises ``KeyError`` on miss.

        If ``index`` is not None, the checkpoint slices the tensor along
        dim 0 on CPU before transferring to GPU.
        """
        return self.ckpt.pop(self._joined(name, sep), index=index)

    # ----- slice enumeration -----

    def slices(self, begin: int, end: int) -> Iterator[tuple[int, Prefix]]:
        """Yield ``(index, Prefix(...))`` for ``index`` in ``[begin, end)``.

        Constructs ``self.prefix + '.' + str(index)`` directly — no regex,
        no checkpoint key scan.  A ``tqdm`` progress bar is shown.

        ``begin`` and ``end`` are required: checkpoints may include
        drafter (speculative-decoding) layers whose indices exceed
        ``num_hidden_layers``, and forcing the bound to be explicit
        keeps drafter weights from silently leaking into a non-drafter
        load. The same parameters double as the future pipeline-parallel
        slice.
        """
        from tqdm import tqdm

        for i in tqdm(range(begin, end), desc='Loading', leave=False):
            yield i, self + str(i)

    # ----- diagnostics -----

    def __repr__(self) -> str:
        return f'Prefix({self.prefix!r})'

    # ----- internal -----

    def _joined(self, name: str, sep: str) -> str:
        if not name:
            return self.prefix
        if not self.prefix:
            return name
        return self.prefix + sep + name


def _apply_mappings(key: str, mappings) -> str:
    for fn in mappings:
        key = fn(key)
    return key


def _gather_shards(model_path: str, index_name: str | None,
                   file_pattern: str | None) -> tuple[list[str], dict]:
    if index_name:
        with open(osp.join(model_path, index_name)) as f:
            index = json.load(f)['weight_map']
        shards = sorted(
            osp.join(model_path, name) for name in set(index.values()))
    else:
        index = {}
        shards = sorted(glob(osp.join(model_path, file_pattern)))
    if not shards:
        raise RuntimeError(
            f'failed to locate weight files under {model_path!r}')
    return shards, index


class Checkpoint(ABC):
    """Abstract storage backend for a flat-keyed tensor store."""

    @abstractmethod
    def get(self, key: str, index=None) -> torch.Tensor:
        """Return tensor at ``key``. Raises ``KeyError`` on miss.

        If ``index`` is not None, the tensor is sliced along dim 0
        (``tensor[index]``) before being returned.
        """

    @abstractmethod
    def has(self, key: str) -> bool:
        """Return whether ``key`` exists in this checkpoint."""

    @abstractmethod
    def pop(self, key: str, index=None) -> torch.Tensor:
        """Return tensor at ``key`` and remove it. Raises ``KeyError`` on miss.

        If ``index`` is not None, the tensor is sliced along dim 0
        (``tensor[index]``) before being returned.
        """

    @abstractmethod
    def keys(self) -> Iterator[str]:
        """Iterate over fully-qualified keys (post-mappings)."""

    def close(self) -> None:
        """Release any open resources.

        Default no-op; idempotent.
        """


class SafetensorsCheckpoint(Checkpoint):
    """Safetensors-backed checkpoint.

    Reads shards directly (no longer wrapping SafetensorsLoader).
    The dict is mmap-backed, so no host-RAM copy happens up front.

    ``mappings`` is a list of regex/string functions applied to every
    checkpoint key on load (the per-model ``_loader_mappings``).
    """

    def __init__(self, model_path: str, *,
                 mappings=(),
                 index_name: str | None = None,
                 file_pattern: str | None = None):
        self._mappings = list(mappings)
        shards, index = _gather_shards(model_path, index_name, file_pattern)
        if not index:
            for shard in shards:
                filename = osp.basename(shard)
                with safe_open(shard, 'pt') as f:
                    index.update({k: filename for k in f.keys()})
        self._data: dict[str, torch.Tensor] = {}
        for shard in shards:
            with safe_open(shard, 'pt') as f:
                filename = osp.basename(shard)
                for k in f.keys():
                    if k not in index or index[k] != filename:
                        continue
                    self._data[_apply_mappings(k, self._mappings)] = (
                        f.get_tensor(k))

    def get(self, key: str, index=None):
        t = self._data[key]
        if index is not None:
            t = t[index]
        return t.cuda()

    def pop(self, key: str, index=None):
        t = self._data.pop(key)
        if index is not None:
            t = t[index]
        return t.cuda()

    def has(self, key: str) -> bool: return key in self._data
    def keys(self): return iter(self._data.keys())
    def close(self) -> None: self._data = {}


class PytorchCheckpoint(Checkpoint):
    """torch.load-backed checkpoint over ``*.bin`` / ``*.pt`` shards."""

    def __init__(self, model_path: str, *,
                 mappings=(),
                 index_name: str | None = None,
                 file_pattern: str | None = None):
        self._mappings = list(mappings)
        shards, _ = _gather_shards(model_path, index_name, file_pattern)
        self._data: dict[str, torch.Tensor] = {}
        for shard in shards:
            tmp = torch.load(shard, map_location='cpu', weights_only=True)
            for k, v in tmp.items():
                self._data[_apply_mappings(k, self._mappings)] = v

    def get(self, key: str, index=None):
        t = self._data[key]
        if index is not None:
            t = t[index]
        return t.cuda()

    def pop(self, key: str, index=None):
        t = self._data.pop(key)
        if index is not None:
            t = t[index]
        return t.cuda()

    def has(self, key: str) -> bool: return key in self._data
    def keys(self): return iter(self._data.keys())
    def close(self) -> None: self._data = {}


def create_checkpoint(model_path: str, *, mappings=()) -> Checkpoint:
    """Pick the right :class:`Checkpoint` subclass for ``model_path``.

    Precedence matches the legacy ``create_loader``:

    1. ``model.safetensors.index.json``  -> SafetensorsCheckpoint (sharded)
    2. ``model*.safetensors``            -> SafetensorsCheckpoint (single-file or unsharded)
    3. ``pytorch_model.bin.index.json``  -> PytorchCheckpoint
    4. ``pytorch_model*.bin``            -> PytorchCheckpoint
    5. ``*.safetensors``                 -> SafetensorsCheckpoint (extra pattern)
    6. ``*.pt`` / ``*.bin``              -> PytorchCheckpoint (extra pattern)
    """
    if osp.exists(osp.join(model_path, SAFE_WEIGHT_INDEX_NAME)):
        return SafetensorsCheckpoint(
            model_path, mappings=mappings,
            index_name=SAFE_WEIGHT_INDEX_NAME)
    if glob(osp.join(model_path, SAFE_WEIGHT_PATTERN)):
        return SafetensorsCheckpoint(
            model_path, mappings=mappings,
            file_pattern=SAFE_WEIGHT_PATTERN)
    if osp.exists(osp.join(model_path, WEIGHT_INDEX_NAME)):
        return PytorchCheckpoint(
            model_path, mappings=mappings,
            index_name=WEIGHT_INDEX_NAME)
    if glob(osp.join(model_path, WEIGHT_PATTERN)):
        return PytorchCheckpoint(
            model_path, mappings=mappings,
            file_pattern=WEIGHT_PATTERN)
    if glob(osp.join(model_path, EXTRA_SAFE_WEIGHT_PATTERN)):
        return SafetensorsCheckpoint(
            model_path, mappings=mappings,
            file_pattern=EXTRA_SAFE_WEIGHT_PATTERN)
    for p in EXTRA_WEIGHT_PATTERNS:
        if glob(osp.join(model_path, p)):
            return PytorchCheckpoint(
                model_path, mappings=mappings, file_pattern=p)
    raise RuntimeError(f'Failed to find valid checkpoint under {model_path!r}')
