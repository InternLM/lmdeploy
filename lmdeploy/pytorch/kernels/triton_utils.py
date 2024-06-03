# Copyright (c) OpenMMLab. All rights reserved.
import functools
import inspect
from typing import Callable, Dict, Sequence, TypeVar, Union, cast, overload

import torch
import triton
from packaging import version
from triton import JITFunction

if version.parse(triton.__version__) <= version.parse('2.2.0'):

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        from triton.runtime.jit import get_cuda_stream

        device = tensor.device
        device_idx = device.index
        device_type = device.type
        stream = get_cuda_stream(device_idx)
        return dict(device=device, device_type=device_type, stream=stream)
else:

    KERNEL_META = dict()

    def get_kernel_meta(tensor: torch.Tensor):
        """kernel meta."""
        return KERNEL_META


TRITON_DIVIIBILITY = getattr(JITFunction, 'divisibility', 16)
TRITON_DIVIIBILITY_8 = getattr(JITFunction, 'divisibility_8', 8)


def _check_type_hint(jit_func: JITFunction, type_hint: Union[Dict, Sequence]):
    """check type hint."""
    params = jit_func.params
    arg_key = tuple(p.name for p in params)

    if isinstance(type_hint, Dict):
        for key in arg_key:
            if key not in type_hint:
                type_hint[key] = None
        return type_hint
    elif type_hint is None:
        return dict((key, None) for key in arg_key)
    elif isinstance(type_hint, Sequence):
        assert len(arg_key) == len(type_hint)
        return dict(zip(arg_key, type_hint))
    else:
        raise RuntimeError(f'Unknown type_hint: {type_hint}')


class JitFunction220Wrapper:

    def __init__(self,
                 jit_func: JITFunction,
                 type_hint: Union[Dict, Sequence] = None):
        """jit func."""
        self.jit_func = jit_func
        self.type_hint = _check_type_hint(jit_func, type_hint)
        self.run = self._make_launcher(jit_func)
        self.arg_names = jit_func.arg_names

        self.__doc__ = jit_func.__doc__
        self.__name__ = jit_func.__name__
        self.__globals__ = jit_func.__globals__
        self.__module__ = jit_func.__module__

    @staticmethod
    def _specialization_key(value):
        if isinstance(value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                value % TRITON_DIVIIBILITY == 0,
                value % TRITON_DIVIIBILITY_8 == 0,
                value == 1,
            )

        if hasattr(value, 'data_ptr'):
            return (value.data_ptr() % TRITON_DIVIIBILITY == 0, )

        return (False, )

    def _make_launcher(self, jit_func: triton.JITFunction):
        """make input builder."""
        from triton.common.backend import get_backend, get_cuda_version_key
        from triton.compiler import (CompiledKernel,
                                     get_arch_default_num_stages,
                                     get_arch_default_num_warps)

        def _make_spec_key_str(key):
            anno = self.type_hint[key]
            if anno == torch.Tensor:
                return f'({key}.data_ptr() % {TRITON_DIVIIBILITY} == 0, )'
            elif anno in [int, bool, torch.int32, torch.int64, torch.uint64]:
                return (f'({key} % {TRITON_DIVIIBILITY} == 0, '
                        f'{key} % {TRITON_DIVIIBILITY_8} == 0, '
                        f'{key} == 1, )')
            elif anno is not None:
                return '(False,)'
            return f'_specialization_key({key})'

        def _make_sig_key_str(key):
            anno = self.type_hint[key]
            if anno == bool:
                return '"i1"'
            elif anno == float:
                return '"fp32"'
            elif anno == torch.int32:
                return '"i32"'
            elif anno == torch.uint64:
                return '"u64"'
            elif anno == torch.int64:
                return '"i64"'
            elif anno == torch.Tensor:
                return f'{key}.dtype'
            return f'_key_of({key})'

        fn = jit_func.fn
        params = jit_func.params

        # arg key
        arg_key = tuple(p.name for p in params)
        arg_key_str = ', '.join(arg_key)
        grid_args = ','.join([f'{arg}={arg}' for arg in arg_key])
        args_signature = ', '.join(
            p.name if p.default ==
            inspect._empty else f'{p.name} == {p.default}' for p in params)

        # constexpr key
        constexpr_key = tuple(p.name for p in params if p.is_constexpr)
        constexpr_key_str = ', '.join(constexpr_key)

        # sig key
        sig_key = tuple(p.name for p in params if not p.is_constexpr)
        sig_name_str = ', '.join(key for key in sig_key)
        sig_key_str = ', '.join(_make_sig_key_str(key) for key in sig_key)

        # spec key
        spec_key = tuple(p.name for p in params if not p.do_not_specialize)
        spec_key_str = ', '.join(_make_spec_key_str(key) for key in spec_key)

        # options
        cuda_opt_fields = dict(
            num_warps=None,
            num_ctas=1,
            num_stages=None,
            enable_warp_specialization=False,
            enable_fp_fusion=True,
            extern_libs=None,
            stream=None,
            device=None,
            device_type=None,
        )
        cuda_opt_signature = ', '.join(f'{k} = {v}'
                                       for k, v in cuda_opt_fields.items())
        cuda_opt_args = ', '.join(f'{k}={k}' for k in cuda_opt_fields)
        src = f"""
def _{fn.__name__}_launcher({args_signature}, grid=None, {cuda_opt_signature}, warmup=False, **kwargs):
    debug=jit_func.debug
    device_backend = None

    if device_type not in ["cuda"]:
        device_backend = get_backend(device_type)
        if device_backend is None:
            raise ValueError("Cannot find backend for " + device_type)

    if num_warps is None:
        num_warps = get_arch_default_num_warps(device_type)
    if num_stages is None:
        num_stages = get_arch_default_num_stages(device_type)

    if device_type in ["cuda"]:
        version_key = get_cuda_version_key()
    else:
        version_key = device_backend.get_version_key()

    sig_key = ({sig_key_str}, )
    spec_key = ({spec_key_str}, )
    constexpr_key = ({constexpr_key_str}, )
    key = (
        version_key,
        sig_key,
        constexpr_key,
        spec_key,
        num_warps,
        num_ctas,
        num_stages,
        enable_warp_specialization,
        enable_fp_fusion,
        debug,
    )
    if extern_libs is not None:
        key = (key, tuple(extern_libs.items()))

    bin = kernel_cache[device].get(key, None)
    if bin is None:
        return jit_func[grid]({arg_key_str}, {cuda_opt_args}, **kwargs)

    non_constexpr_arg_values = ({sig_name_str})
    if callable(grid):
        grid = grid(dict({grid_args}))
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    if not hasattr(bin, 'tensormaps_info'):
        bin.c_wrapper(
            grid_0,
            grid_1,
            grid_2,
            bin.num_warps,
            bin.num_ctas,
            *bin.clusterDims,
            bin.shared,
            stream,
            bin.cu_function,
            launch_enter_hook,
            launch_exit_hook,
            bin,
            {sig_name_str},
        )
    else:
        bin.c_wrapper(
            grid_0,
            grid_1,
            grid_2,
            bin.num_warps,
            bin.num_ctas,
            *bin.clusterDims,
            bin.shared,
            stream,
            bin.cu_function,
            launch_enter_hook,
            launch_exit_hook,
            bin,
            *bin.assemble_tensormap_to_arg(non_constexpr_arg_values),
        )

    return bin
"""   # noqa: E501
        scope = dict(
            get_backend=get_backend,
            get_arch_default_num_stages=get_arch_default_num_stages,
            get_arch_default_num_warps=get_arch_default_num_warps,
            _specialization_key=self._specialization_key,
            get_cuda_version_key=get_cuda_version_key,
            jit_func=jit_func,
            _key_of=JITFunction._key_of,
            kernel_cache=jit_func.cache,
            launch_enter_hook=CompiledKernel.launch_enter_hook,
            launch_exit_hook=CompiledKernel.launch_exit_hook,
        )
        exec(src, scope)
        return scope[f'_{fn.__name__}_launcher']

    def __getitem__(self, grid):
        """get item."""
        return functools.partial(cast(Callable, self.run), grid=grid)


class JitFunction230Wrapper:

    def __init__(self,
                 jit_func: JITFunction,
                 type_hint: Union[Dict, Sequence] = None):
        """jit func."""
        self.jit_func = jit_func
        self.type_hint = _check_type_hint(jit_func, type_hint)
        self.run = self._make_launcher(jit_func)
        self.arg_names = jit_func.arg_names

        self.__doc__ = jit_func.__doc__
        self.__name__ = jit_func.__name__
        self.__globals__ = jit_func.__globals__
        self.__module__ = jit_func.__module__

    @staticmethod
    def _specialization_key(value):
        if hasattr(value, 'data_ptr'):
            return (value.data_ptr() % TRITON_DIVIIBILITY == 0, )

        if isinstance(value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                value % TRITON_DIVIIBILITY == 0,
                value % TRITON_DIVIIBILITY_8 == 0,
                value == 1,
            )

        return (False, )

    def _make_launcher(self, jit_func: triton.JITFunction):
        """make input builder."""
        from dataclasses import fields

        from triton.common.backend import get_cuda_version_key
        from triton.compiler import CompiledKernel
        from triton.compiler.backends.cuda import CUDABackend, CUDAOptions
        from triton.runtime.driver import driver

        def _make_spec_key_str(key):
            anno = self.type_hint[key]
            if anno == torch.Tensor:
                return f'({key}.data_ptr() % {TRITON_DIVIIBILITY} == 0, )'
            elif anno in [int, bool, torch.int32, torch.int64, torch.uint64]:
                return (f'({key} % {TRITON_DIVIIBILITY} == 0, '
                        f'{key} % {TRITON_DIVIIBILITY_8} == 0, '
                        f'{key} == 1, )')
            elif anno is not None:
                return '(False,)'
            return f'_specialization_key({key})'

        def _make_sig_key_str(key):
            anno = self.type_hint[key]
            if anno == bool:
                return '"i1"'
            elif anno == float:
                return '"fp32"'
            elif anno == torch.int32:
                return '"i32"'
            elif anno == torch.uint64:
                return '"u64"'
            elif anno == torch.int64:
                return '"i64"'
            elif anno == torch.Tensor:
                return f'{key}.dtype'
            return f'_key_of({key})'

        fn = jit_func.fn
        params = jit_func.params

        # arg key
        arg_key = tuple(p.name for p in params)
        arg_key_str = ', '.join(arg_key)
        grid_args = ','.join([f'{arg}={arg}' for arg in arg_key])
        args_signature = ', '.join(
            p.name if p.default ==
            inspect._empty else f'{p.name} == {p.default}' for p in params)

        # constexpr key
        constexpr_key = tuple(p.name for p in params if p.is_constexpr)
        constexpr_key_str = ', '.join(constexpr_key)

        # sig key
        sig_key = tuple(p.name for p in params if not p.is_constexpr)
        sig_name_str = ', '.join(key for key in sig_key)
        sig_key_str = ', '.join(_make_sig_key_str(key) for key in sig_key)

        # spec key
        spec_key = tuple(p.name for p in params if not p.do_not_specialize)
        spec_key_str = ', '.join(_make_spec_key_str(key) for key in spec_key)

        # cuda opt key/default
        cuda_opt_fields = dict(
            (f.name, f.default) for f in fields(CUDAOptions))
        cuda_opt_fields['debug'] = jit_func.debug
        cuda_opt_signature = ', '.join(f'{k} = {v}'
                                       for k, v in cuda_opt_fields.items())
        cuda_opt_args = ', '.join(f'{k}={k}' for k in cuda_opt_fields)

        triton_version = version.parse(triton.__version__)
        if triton_version == version.parse('2.3.0'):
            mni_acc_default = '0 if target[1] >= 89 else None'
        else:
            mni_acc_default = '2**30 if target[1] == 90 else 0'

        src = f"""
def _{fn.__name__}_launcher({args_signature}, grid=None, {cuda_opt_signature}, warmup=False, **kwargs):
    device = get_current_device()
    stream = get_current_stream(device)
    target = get_current_target()
    if target[1] >= 89:
        allow_fp8e4nv = True
    max_num_imprecise_acc_default = {mni_acc_default}
    options = CUDAOptions({cuda_opt_args}, )
    sig_key = ({sig_key_str}, )
    spec_key = ({spec_key_str}, )
    constexpr_key = ({constexpr_key_str}, )
    key = (get_cuda_version_key(), sig_key, constexpr_key, spec_key, options)

    kernel = kernel_cache[device].get(key, None)
    if kernel is None:
        return jit_func[grid]({arg_key_str}, {cuda_opt_args}, **kwargs)

    args = ({sig_name_str})
    if callable(grid):
        grid = grid(dict({grid_args}))
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    if kernel.metadata["tensormaps_info"] is not None:
        kernel.run(grid_0, grid_1, grid_2, kernel.num_warps, kernel.num_ctas,
                kernel.cluster_dims[0], kernel.cluster_dims[1], kernel.cluster_dims[2],
                kernel.shared, stream, kernel.function, launch_enter_hook,
                launch_exit_hook, kernel,
                *assemble_tensormap_to_arg(kernel.metadata["tensormaps_info"], args))
    else:
        kernel.run(grid_0, grid_1, grid_2, kernel.num_warps, kernel.num_ctas,
                kernel.cluster_dims[0], kernel.cluster_dims[1], kernel.cluster_dims[2],
                kernel.shared, stream, kernel.function, launch_enter_hook,
                launch_exit_hook, kernel,
                {sig_name_str})

    return kernel
"""   # noqa: E501
        scope = dict(
            get_current_device=driver.get_current_device,
            get_current_stream=driver.get_current_stream,
            get_current_target=driver.get_current_target,
            assemble_tensormap_to_arg=driver.assemble_tensormap_to_arg,
            _specialization_key=self._specialization_key,
            get_cuda_version_key=get_cuda_version_key,
            CUDABackend=CUDABackend,
            CUDAOptions=CUDAOptions,
            jit_func=jit_func,
            _key_of=JITFunction._key_of,
            kernel_cache=jit_func.cache,
            launch_enter_hook=CompiledKernel.launch_enter_hook,
            launch_exit_hook=CompiledKernel.launch_exit_hook,
        )
        exec(src, scope)
        return scope[f'_{fn.__name__}_launcher']

    def __getitem__(self, grid):
        """get item."""
        return functools.partial(cast(Callable, self.run), grid=grid)


T = TypeVar('T')


@overload
def wrap_jit_func(func: T):
    ...


@overload
def wrap_jit_func(
    *,
    type_hint=None,
):
    ...


def wrap_jit_func(
    func: T = None,
    *,
    type_hint=None,
):
    """wrap jit func."""

    def decorator(func: T):
        triton_version = version.parse(triton.__version__)

        if triton_version == version.parse('2.2.0'):
            return JitFunction220Wrapper(func, type_hint)
        if version.parse('2.2.0') < triton_version <= version.parse('2.3.1'):
            return JitFunction230Wrapper(func, type_hint)
        return func

    if func is not None:
        return decorator(func)
    else:
        return decorator
