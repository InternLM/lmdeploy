# Copyright (c) OpenMMLab. All rights reserved.
import functools
import inspect
from typing import Callable, cast

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


class JitFunction220Wrapper:

    def __init__(self, jit_func: JITFunction):
        """jit func."""
        self.jit_func = jit_func
        self.run = self._make_launcher(jit_func)

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

        def _make_spec_key_str(anno, key):
            if 'Tensor' in anno:
                return f'({key}.data_ptr() % TRITON_DIVIIBILITY == 0, )'
            if anno == 'int' or anno == 'bool':
                return (f'({key} % TRITON_DIVIIBILITY == 0, '
                        f'{key} % TRITON_DIVIIBILITY_8 == 0, '
                        f'{key} == 1, )')
            return f'_specialization_key({key})'

        def _make_sig_key_str(anno, key):
            if anno == 'bool':
                return '"i1"'
            if anno == 'float':
                return '"fp32"'
            if 'Tensor' in anno:
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
        annotations = tuple(p.annotation for p in params if not p.is_constexpr)
        sig_key_str = ', '.join(
            _make_sig_key_str(anno, key)
            for anno, key in zip(annotations, sig_key))

        # spec key
        spec_key = tuple(p.name for p in params if not p.do_not_specialize)
        spec_annos = tuple(p.annotation for p in params
                           if not p.do_not_specialize)
        spec_key_str = ', '.join(
            _make_spec_key_str(anno, key)
            for anno, key in zip(spec_annos, spec_key))

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
def _{fn.__name__}_launcher({args_signature}, grid=None, {cuda_opt_signature}, **kwargs):
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
    bin.c_wrapper(
        grid_0,
        grid_1,
        grid_2,
        bin.num_warps,
        bin.num_ctas,
        bin.clusterDims[0],
        bin.clusterDims[1],
        bin.clusterDims[2],
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
            TRITON_DIVIIBILITY=TRITON_DIVIIBILITY,
            TRITON_DIVIIBILITY_8=TRITON_DIVIIBILITY_8,
        )
        exec(src, scope)
        return scope[f'_{fn.__name__}_launcher']

    def __getitem__(self, grid):
        """get item."""
        return functools.partial(cast(Callable, self.run), grid=grid)


class JitFunction230Wrapper:

    def __init__(self, jit_func: JITFunction):
        """jit func."""
        self.jit_func = jit_func
        self.run = self._make_launcher(jit_func)

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

        def _make_spec_key_str(anno, key):
            return f'_specialization_key({key})'

        def _make_sig_key_str(anno, key):
            if anno == 'bool':
                return '"i1"'
            if anno == 'float':
                return '"fp32"'
            if 'Tensor' in anno:
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
        annotations = tuple(p.annotation for p in params if not p.is_constexpr)
        sig_key_str = ', '.join(
            _make_sig_key_str(anno, key)
            for anno, key in zip(annotations, sig_key))

        # spec key
        spec_key = tuple(p.name for p in params if not p.do_not_specialize)
        spec_annos = tuple(p.annotation for p in params
                           if not p.do_not_specialize)
        spec_key_str = ', '.join(
            _make_spec_key_str(anno, key)
            for anno, key in zip(spec_annos, spec_key))

        # cuda opt key/default
        cuda_opt_fields = dict(
            (f.name, f.default) for f in fields(CUDAOptions))
        cuda_opt_fields['debug'] = jit_func.debug
        cuda_opt_signature = ', '.join(f'{k} = {v}'
                                       for k, v in cuda_opt_fields.items())
        cuda_opt_args = ', '.join(f'{k}={k}' for k in cuda_opt_fields)

        src = f"""
def _{fn.__name__}_launcher({args_signature}, grid=None, {cuda_opt_signature}, **kwargs):
    device = get_current_device()
    stream = get_current_stream(device)
    target = get_current_target()
    if target[1] >= 89:
        allow_fp8e4nv = True
        max_num_imprecise_acc_default = 0
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
    kernel.run(grid_0, grid_1, grid_2, kernel.num_warps, kernel.num_ctas,
               kernel.cluster_dims[0], kernel.cluster_dims[1], kernel.cluster_dims[2],
               kernel.shared, stream, kernel.function, launch_enter_hook,
               launch_exit_hook, kernel,
               *assemble_tensormap_to_arg(kernel.metadata["tensormaps_info"], args))

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


def wrap_jit_func(func):
    """wrap jit func."""
    triton_version = version.parse(triton.__version__)

    if triton_version == version.parse('2.2.0'):
        return JitFunction220Wrapper(func)
    if triton_version == version.parse('2.3.0'):
        return JitFunction230Wrapper(func)
    return func
