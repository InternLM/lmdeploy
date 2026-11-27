# Copyright (c) OpenMMLab. All rights reserved.
"""Private FLA call bindings with explicit FP32 dot launch precision."""

import inspect
from copy import copy
from functools import lru_cache, update_wrapper
from types import FunctionType, ModuleType

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction, KernelInterface


_MODULES = frozenset({
    "fla.ops.kda.chunk_intra", "fla.ops.kda.chunk_intra_token_parallel",
    "fla.ops.kda.wy_fast", "fla.ops.kda.chunk_bwd",
    "fla.ops.common.chunk_delta_h", "fla.ops.gla.chunk",
})


class _DotKernel(KernelInterface):
    def __init__(self, kernel, precision):
        self.kernel, self.precision = kernel, precision

    def run(self, *args, **kwargs):
        # Compiler launch options are in the Triton kernel cache key. Explicit
        # per-dot choices (such as allow_tf32=False) retain their original value.
        kwargs["default_dot_input_precision"] = self.precision
        return self.kernel.run(*args, **kwargs)


@triton.jit
def _selected_dot(input, other, acc, input_precision: tl.constexpr, allow_tf32: tl.constexpr,
                   max_num_imprecise_acc: tl.constexpr, out_dtype: tl.constexpr, default: tl.constexpr):
    # Omit None-valued compatibility arguments: tl.dot checks Python identity
    # before unwrapping constexpr, so forwarding constexpr[None] is invalid.
    if allow_tf32 is None:
        precision: tl.constexpr = default if input_precision is None else input_precision
        if acc is None:
            return tl.dot(input, other, input_precision=precision,
                          max_num_imprecise_acc=max_num_imprecise_acc, out_dtype=out_dtype)
        return tl.dot(input, other, acc, input_precision=precision,
                      max_num_imprecise_acc=max_num_imprecise_acc, out_dtype=out_dtype)
    if acc is None:
        return tl.dot(input, other, allow_tf32=allow_tf32,
                      max_num_imprecise_acc=max_num_imprecise_acc, out_dtype=out_dtype)
    return tl.dot(input, other, acc, allow_tf32=allow_tf32,
                  max_num_imprecise_acc=max_num_imprecise_acc, out_dtype=out_dtype)


@triton.jit
def _dot_tf32x3(input, other, acc=None, input_precision: tl.constexpr = None,
                allow_tf32: tl.constexpr = None, max_num_imprecise_acc: tl.constexpr = None,
                out_dtype: tl.constexpr = tl.float32):
    return _selected_dot(input, other, acc, input_precision, allow_tf32,
                         max_num_imprecise_acc, out_dtype, "tf32x3")


@triton.jit
def _dot_ieee(input, other, acc=None, input_precision: tl.constexpr = None,
              allow_tf32: tl.constexpr = None, max_num_imprecise_acc: tl.constexpr = None,
              out_dtype: tl.constexpr = tl.float32):
    return _selected_dot(input, other, acc, input_precision, allow_tf32,
                         max_num_imprecise_acc, out_dtype, "ieee")


@lru_cache(maxsize=None)
def _language(precision):
    language = ModuleType(f"lmdeploy_kda_tl_{precision}")
    language.__dict__.update(vars(tl))
    language.__name__ = f"lmdeploy_kda_tl_{precision}"
    # A JIT function is a tracked compiler dependency, so the precision policy
    # is part of caller hashes as well as launch options (including disk cache).
    language.dot = _dot_tf32x3 if precision == "tf32x3" else _dot_ieee
    return language


def _namespace(function, precision):
    namespace = dict(function.__globals__)
    for name in function.__code__.co_names:
        value = namespace.get(name)
        if value is tl:
            namespace[name] = _language(precision)
        elif isinstance(value, KernelInterface):
            inner = value
            while not isinstance(inner, JITFunction) and hasattr(inner, "fn"):
                inner = inner.fn
            if isinstance(inner, JITFunction) and inner.fn.__module__ in _MODULES:
                namespace[name] = _kernel(value, precision)
        elif isinstance(value, FunctionType) and value.__module__ in _MODULES:
            namespace[name] = _bind(value, precision)
    return namespace


@lru_cache(maxsize=None)
def _kernel(kernel, precision):
    if isinstance(kernel, JITFunction):
        original = kernel.fn
        function = FunctionType(original.__code__, _namespace(original, precision), original.__name__,
                                original.__defaults__, original.__closure__)
        function.__kwdefaults__ = original.__kwdefaults__
        update_wrapper(function, original)
        return triton.jit(function, do_not_specialize=kernel.do_not_specialize,
                          do_not_specialize_on_alignment=kernel.do_not_specialize_on_alignment,
                          debug=kernel.debug, noinline=kernel.noinline)
    bound = copy(kernel)
    bound.fn = _kernel(kernel.fn, precision)
    if hasattr(bound, "cache"):
        bound.cache = {}
    return bound


@lru_cache(maxsize=None)
def _bind(function, precision):
    original = inspect.unwrap(function)
    # Bind only referenced host functions and kernels. The original FLA module
    # dictionaries, kernel source, backend registry and global flags stay intact.
    namespace = _namespace(original, precision)
    for name in original.__code__.co_names:
        if isinstance(namespace.get(name), KernelInterface):
            namespace[name] = _DotKernel(namespace[name], precision)
    bound = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    bound.__kwdefaults__ = original.__kwdefaults__
    return update_wrapper(bound, original)


def fp32_dot_function(function, precision):
    """Bind a KDA FLA call to a local Triton dot precision policy.

    Args:
        function (Callable): FLA host function in the supported KDA call graph.
        precision (str): ``tf32`` (native path), ``tf32x3`` or ``ieee``.

    Returns:
        Callable: Original function or a private host binding to Triton kernels.
    """
    if precision not in {"tf32", "tf32x3", "ieee"}:
        raise ValueError(f"Unsupported KDA FP32 dot precision: {precision}")
    if precision == "tf32":
        return function
    if inspect.unwrap(function).__module__ not in _MODULES:
        raise ValueError(f"Unsupported FLA function: {function.__module__}.{function.__name__}")
    return _bind(function, precision)
