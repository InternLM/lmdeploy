# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Iterable, Literal, Sequence, overload

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
    split_prefill: bool = True,
    split_decoding: bool = True,
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
    split_prefill: bool = True,
    split_decoding: bool = True,
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
    split_prefill: bool = True,
    split_decoding: bool = True,
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
        self._custom_ops = dict()
        self._split_prefill_ops = set()
        self._split_decoding_ops = set()

    def add_custom_op(self, name: str, custom_op: CustomOpDef, split_prefill: bool = True, split_decoding: bool = True):
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


def get_custom_op_manager():
    """Get custom op manager."""
    return CustomOpManager()
