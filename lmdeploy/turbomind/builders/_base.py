# Copyright (c) OpenMMLab. All rights reserved.

import enum

import _turbomind as _tm
import torch

from ..linear import Linear

# ---------------------------------------------------------------------------
# SplitSide enum (internal -- not exposed to specs)
# ---------------------------------------------------------------------------


class SplitSide(enum.Enum):
    """Semantic TP split direction for commit operations.

    OUTPUT -- column-parallel: split along the output dimension (axis -1)
    INPUT  -- row-parallel:    split along the input dimension  (axis  0)
    """

    OUTPUT = 'output'
    INPUT = 'input'


# ---------------------------------------------------------------------------
# Canonical dtype mappings (moved from commit.py)
# ---------------------------------------------------------------------------

_STR_TO_DTYPE: dict[str, _tm.DataType] = {
    'float32':  _tm.DataType.TYPE_FP32,
    'float16':  _tm.DataType.TYPE_FP16,
    'bfloat16': _tm.DataType.TYPE_BF16,
}

_TORCH_TO_CPP: dict[torch.dtype, _tm.DataType] = {
    torch.float32:  _tm.DataType.TYPE_FP32,
    torch.float16:  _tm.DataType.TYPE_FP16,
    torch.bfloat16: _tm.DataType.TYPE_BF16,
    torch.int32:    _tm.DataType.TYPE_INT32,
    torch.int64:    _tm.DataType.TYPE_INT64,
    torch.int8:     _tm.DataType.TYPE_INT8,
    torch.uint8:    _tm.DataType.TYPE_UINT8,
}

_CPP_TO_TORCH: dict[_tm.DataType, torch.dtype] = {v: k for k, v in _TORCH_TO_CPP.items()}

_SPLIT_SIDE_TO_DIM: dict[SplitSide, int] = {SplitSide.OUTPUT: -1, SplitSide.INPUT: 0}


# ---------------------------------------------------------------------------
# Dtype / format helpers (moved from commit.py)
# ---------------------------------------------------------------------------


def _cpp_dtype(dtype_str: str):
    """Convert a model-config data_type string to C++ DataType enum."""
    return _STR_TO_DTYPE[dtype_str]


def _act_type_id(act_str: str) -> int:
    """Convert activation_type string to C++ ActivationType enum value."""
    return {'silu': 0, 'gpt-oss': 1}.get(act_str, 0)


def _torch_dtype_to_cpp(dtype: torch.dtype):
    """Convert a torch dtype to the C++ ``DataType`` enum, or ``None``."""
    return _TORCH_TO_CPP.get(dtype)


def _cast_shard_for_tm(shard: torch.Tensor, tm_tensor) -> torch.Tensor:
    """Cast *shard* dtype to match *tm_tensor*'s C++ dtype when needed."""
    if tm_tensor.type == _tm.DataType.TYPE_FP32 and shard.dtype in (torch.float16, torch.bfloat16):
        return shard.float()
    if tm_tensor.type == _tm.DataType.TYPE_FP16 and shard.dtype != torch.float16:
        return shard.half()
    if tm_tensor.type == _tm.DataType.TYPE_BF16 and shard.dtype != torch.bfloat16:
        return shard.to(torch.bfloat16)
    return shard



def _copy_shard_to_param(handle, param_name: str, shard: torch.Tensor, *,
                         alloc_shape: list[int] | None = None,
                         alloc_dtype=None) -> None:
    """Move shard to GPU, allocate the C++ param slot, cast, and copy.

    Invariant: ``dst.byte_size == shard.nbytes`` after the cast.  Upstream
    is responsible for any padding/reshape needed to satisfy this.  A
    mismatch raises immediately.

    ``alloc_shape`` / ``alloc_dtype`` default to the shard's own shape /
    dtype.  Override only to express shape/dtype *relabels* where byte
    size is preserved (e.g. quantized weight: physical int32
    [in, out/8] stored in a logical UINT4 [in, out] C++ slot).
    """
    if not shard.is_cuda:
        shard = shard.cuda(0).contiguous()
    elif not shard.is_contiguous():
        shard = shard.contiguous()

    if alloc_shape is None:
        alloc_shape = list(shard.shape)
    if alloc_dtype is None:
        alloc_dtype = _torch_dtype_to_cpp(shard.dtype)

    dst = handle.param(param_name).alloc(alloc_shape, alloc_dtype)
    shard = _cast_shard_for_tm(shard, dst)
    assert dst.byte_size == shard.nbytes, (
        f'{param_name}: alloc byte_size={dst.byte_size} != '
        f'shard.nbytes={shard.nbytes}')
    dst.copy_from(shard)


def _shard(tensor: torch.Tensor, split_dim: int | None, tp: int,
           rank: int) -> torch.Tensor:
    """Return the ``rank``-th split along ``split_dim``, or the tensor
    unchanged.

    Used wherever a TP shard is selected from a broadcast-by-default
    tensor.  A ``split_dim`` of ``None`` or ``tp <= 1`` returns the tensor
    untouched.
    """
    if split_dim is None or tp <= 1:
        return tensor
    return tensor.split(tensor.shape[split_dim] // tp, dim=split_dim)[rank]


# ---------------------------------------------------------------------------
# Builder base class
# ---------------------------------------------------------------------------


class BuiltModule:
    """Opaque handle bundle returned by ``Builder.build()``.

    Wraps a list of per-GPU C++ module handles.  Iteration and len delegate
    to the underlying list so callers can ``zip(BuiltModule, contexts)`` etc.
    """

    __slots__ = ('handles',)

    def __init__(self, handles):
        self.handles = handles

    def __iter__(self):
        return iter(self.handles)

    def __len__(self):
        return len(self.handles)


class Builder:
    """Wraps N GPU handles for a single logical module.

    Distributes module creation, child binding, and weight commits
    across all GPUs with bound TP configuration.

    Subclasses specialize for particular module types (e.g. attention,
    FFN, MoE).

    Lifecycle: stage commits -> build() -> BuiltModule (frozen).
    After ``build()`` the Builder is inert — further commits or child
    attachments raise.
    """

    def __init__(self, config, contexts, tp=1, ranks=None):
        """Initialise the builder with staging dicts.

        Parameters
        ----------
        config : C++ config struct
            Config with ``clone()`` method and optionally ``tp_rank`` field.
        contexts : list
            GPU context managers (one per GPU).
        tp : int
            Tensor parallelism degree.
        ranks : list[int] | None
            Per-GPU TP ranks.
        """
        # `_built` must be set first: __setattr__ reads it inside the
        # BuiltModule branch.  Bool is not a BuiltModule, so the normal
        # fall-through assigns it via object.__setattr__ at the end of
        # __setattr__.
        self._built = False
        self._contexts = contexts
        self._tp = tp
        self._ranks = ranks
        self.config = config
        self._pending_tensors = {}
        self._pending_children = {}
        self._handles = None

    # ------------------------------------------------------------------
    # Child binding via attribute assignment
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value):
        if isinstance(value, Builder):
            raise TypeError(
                f'{type(self).__name__}.{name}: assign .build() output '
                f'(BuiltModule), not the Builder itself')
        if isinstance(value, BuiltModule):
            if self._built:
                raise RuntimeError(
                    f'{type(self).__name__} is built; '
                    f'cannot assign {name!r}')
            self._add_child(name, value.handles)
            return
        object.__setattr__(self, name, value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def tp_size(self):
        return self._tp

    def _rank_for(self, gpu_idx: int) -> int:
        if self._ranks and self._tp > 1:
            return self._ranks[gpu_idx]
        return 0

    # ------------------------------------------------------------------
    # Add methods — stage into pending dicts (pre-build only)
    # ------------------------------------------------------------------

    def _add_linear(self, name: str, linear: Linear,
                       split_side: SplitSide | None = None):
        """Create standalone LinearWeight modules and copy tensor data.

        Creates per-GPU LinearWeight modules via ``_tm.create_module``
        at commit time.  Attachment to the parent module is deferred to
        ``build()`` via ``_commit_child``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")

        w = linear.tensors.get('weight')
        if w is None:
            return

        # --- GPU-invariant preparation -------------------------------------
        fmt = linear.weight_format

        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        in_dim, out_dim = w.shape[0], w.shape[-1]
        if split_side == SplitSide.OUTPUT:
            out_dim //= tp
        elif split_side == SplitSide.INPUT:
            in_dim //= tp

        compute_dtype = self.config.data_type
        lin_cfg = _tm.LinearConfig()
        lin_cfg.input_dim  = in_dim
        lin_cfg.output_dim = out_dim
        lin_cfg.data_type  = compute_dtype or _tm.DataType.TYPE_INVALID
        lin_cfg.format     = linear.weight_format.make_data_format(compute_dtype)
        lin_cfg.has_bias   = 'bias' in linear.tensors

        packed = {k: fmt.pack(t, k) for k, t in linear.tensors.items()}
        tensors = {k: p.tensor for k, p in packed.items()}

        kind_split_dims = {
            kind: None if (kind == 'bias' and split_side == SplitSide.INPUT)
                  else split_dim
            for kind in tensors
        }

        if tp > 1 and split_dim is not None:
            for kind, tensor in tensors.items():
                kind_split_dim = kind_split_dims[kind]
                if kind_split_dim is not None:
                    d = tensor.shape[kind_split_dim]
                    assert d % tp == 0, (
                        f'TP split: {name}.{kind} dim {kind_split_dim} '
                        f'has size {d}, not divisible by tp={tp}.')

        # --- Per-GPU: standalone creation + tensor copy --------------------
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                rank = self._rank_for(i) if tp > 1 else 0

                mod = _tm.create_module(lin_cfg)

                for kind, tensor in tensors.items():
                    shard = _shard(tensor, kind_split_dims[kind], tp, rank)

                    alloc_shape, alloc_dtype = packed[kind].alloc_shape, \
                                               packed[kind].alloc_dtype
                    if alloc_shape is not None and split_dim is not None \
                            and tp > 1:
                        alloc_shape = list(alloc_shape)
                        alloc_shape[split_dim] //= tp
                    if alloc_dtype is None and kind == 'weight':
                        alloc_dtype = self.config.data_type

                    _copy_shard_to_param(mod, kind, shard,
                                         alloc_shape=alloc_shape,
                                         alloc_dtype=alloc_dtype)

                handles.append(mod)

        self._add_child(name, handles)

    def _add_tensor(self, name: str, tensor: torch.Tensor | None,
                       split_side: SplitSide | None = None):
        """Stage a raw-tensor commit under ``name``.

        Applied during
        ``build()`` in ``_commit_tensor``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        if tensor is not None:
            self._pending_tensors[name] = (tensor, split_side)

    # ------------------------------------------------------------------
    # Add helpers
    # ------------------------------------------------------------------

    def _add_child(self, name: str, handles: list):
        """Stage pre-created per-GPU ``Module*`` handles under ``name``.

        Applied during ``build()`` in ``_commit_child``.
        """
        assert not self._built, (
            f"{type(self).__name__} is built; commit '{name}' rejected")
        assert name not in self._pending_children, (
            f"{type(self).__name__}: duplicate child commit '{name}'")
        self._pending_children[name] = handles

    # ------------------------------------------------------------------
    # build() — create handles, drain staged state, return BuiltModule
    # ------------------------------------------------------------------

    def build(self) -> BuiltModule:
        """Create C++ module handles and drain all staged state.

        Idempotent on second call — returns the same ``BuiltModule``.
        """
        if self._built:
            return BuiltModule(self._handles)

        self._create_handles()

        # True is not BuiltModule; falls through to plain assignment.
        self._built = True

        # Drain staged children (linear weights + sub-builder output)
        for name, handles in self._pending_children.items():
            self._commit_child(name, handles)

        # Drain staged tensors
        for name, (tensor, split_side) in self._pending_tensors.items():
            self._commit_tensor(name, tensor, split_side)

        return BuiltModule(self._handles)

    def _create_handles(self):
        """Create one C++ module per context via ``_tm.create_module(cfg)``."""
        handles = []
        for i, ctx in enumerate(self._contexts):
            with ctx:
                cfg = self._cfg_for_rank(i)
                handle = _tm.create_module(cfg)
                handles.append(handle)
        self._handles = handles

    def _cfg_for_rank(self, gpu_idx: int):
        """Clone config and set tp_rank if tp > 1."""
        if self._tp > 1 and hasattr(self.config, 'tp_rank'):
            cfg = self.config.clone()
            cfg.tp_rank = self._ranks[gpu_idx]
            return cfg
        return self.config

    def _commit_child(self, name: str, handles: list):
        """Attach pre-created per-GPU child handles to parent handles."""
        for i, (parent_h, child_h) in enumerate(
                zip(self._handles, handles)):
            with self._contexts[i]:
                parent_h.add_child_raw(name, child_h)

    # ------------------------------------------------------------------
    # Commit methods — drain pending dicts to C++ modules
    # ------------------------------------------------------------------

    def _commit_tensor(self, name: str, tensor: torch.Tensor,
                      split_side: SplitSide | None = None):
        """Commit a raw tensor to a named parameter on all GPUs.

        Parameters
        ----------
        name : str
            Parameter name within the module.
        tensor : torch.Tensor
            The tensor data.
        split_side : SplitSide | None
            TP split semantics.  ``None`` means broadcast.
        """
        tp = self._tp if split_side else 1
        split_dim = _SPLIT_SIDE_TO_DIM.get(split_side) if split_side else None

        for i, handle in enumerate(self._handles):
            with self._contexts[i]:
                rank = self._rank_for(i) if tp > 1 else 0
                shard = _shard(tensor, split_dim, tp, rank)
                _copy_shard_to_param(handle, name, shard,
                                     alloc_dtype=None)
