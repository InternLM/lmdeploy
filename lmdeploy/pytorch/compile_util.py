# Copyright (c) OpenMMLab. All rights reserved.
# Two levels of custom op registration exist in this codebase:
#
# 1. `custom_op()` (this module) — for ops that act as **graph split points**
#    in the piecewise torch.compile backend. These are ops that cannot be
#    captured into a CUDA graph (e.g. attention with dynamic shapes, EP MoE
#    with NCCL comms). The `split_prefill` / `split_decoding` flags control
#    which compilation phase treats them as split boundaries.
#
# 2. `torch.library.custom_op()` (PyTorch built-in) — for ops that run
#    **inside** captured CUDA graph subgraphs (e.g. Triton kernel launchers,
#    FP8 GEMM). These don't need split metadata and are invisible to the
#    piecewise backend's graph splitter.
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal, overload

import torch
from torch._library.custom_ops import CustomOpDef, device_types_t

from lmdeploy.pytorch.utils import singleton


@overload
def custom_op(
    name: str,
    fn: Literal[None] = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = None,
    schema: str | None = None,
    split_prefill: bool = False,
    split_decoding: bool = False,
) -> Callable[[Callable[..., object]], 'CustomOpDef']:
    ...


@overload
def custom_op(
    name: str,
    fn: Callable[..., object],
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = None,
    schema: str | None = None,
    split_prefill: bool = False,
    split_decoding: bool = False,
) -> 'CustomOpDef':
    ...


def custom_op(
    name: str,
    fn: Callable | None = None,
    /,
    *,
    mutates_args: str | Iterable[str],
    device_types: device_types_t = None,
    schema: str | None = None,
    tags: Sequence[Any] = None,
    split_prefill: bool = False,
    split_decoding: bool = False,
):
    """A decorator to mark a function as a custom op for torch.compile."""

    def decorator(fn: Callable[..., object]) -> 'CustomOpDef':
        ret = torch.library.custom_op(name,
                                      fn,
                                      mutates_args=mutates_args,
                                      device_types=device_types,
                                      schema=schema,
                                      tags=tags)
        get_custom_op_manager().add_custom_op(name, ret, split_prefill=split_prefill, split_decoding=split_decoding)
        return ret

    if fn is None:
        return decorator
    return decorator(fn)


@singleton
class CustomOpManager:
    """Manager for custom ops."""

    def __init__(self):
        from itertools import count
        from weakref import WeakValueDictionary
        self._custom_ops = dict()
        self._split_prefill_ops = set()
        self._split_decoding_ops = set()

        # register mod instances
        self._mod_instances = WeakValueDictionary()
        self.counter = count()

    def register_mod_instance(self, instance: Any):
        """Register mod instance."""
        key = next(self.counter)
        self._mod_instances[key] = instance
        return key

    def get_mod_instance(self, key: int):
        """Get instance."""
        return self._mod_instances.get(key, None)

    def add_custom_op(self, name: str, custom_op: CustomOpDef,
                      split_prefill: bool = False, split_decoding: bool = False):
        """Add custom op."""
        self._custom_ops[name] = custom_op
        if split_prefill:
            self._split_prefill_ops.add(name)
        if split_decoding:
            self._split_decoding_ops.add(name)

    def get_split_prefill_ops(self):
        """Get split prefill ops."""
        return self._split_prefill_ops

    def get_split_decoding_ops(self):
        """Get split decoding ops."""
        return self._split_decoding_ops


# Eagerly create the singleton so it exists before any torch.compile tracing.
# The @singleton decorator returns a factory function; calling it here ensures
# the instance is created at import time (in eager Python), not during tracing
# where the constructor call could be captured by dynamo.
_custom_op_manager = CustomOpManager()


def get_custom_op_manager():
    """Get custom op manager."""
    return _custom_op_manager
