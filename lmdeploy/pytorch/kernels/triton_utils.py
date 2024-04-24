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
        try:
            return (value.data_ptr() % JITFunction.divisibility == 0, )
        except AttributeError:
            pass

        if isinstance(value, int):
            # bool is a subclass of int, so we don't check explicitly above.
            return (
                value % JITFunction.divisibility == 0,
                value % JITFunction.divisibility_8 == 0,
                value == 1,
            )

        return (False, )

    def _make_launcher(self, jit_func: triton.JITFunction):
        """make input builder."""
        from triton.common.backend import get_cuda_version_key
        from triton.compiler import CompiledKernel
        from triton.compiler.backends.cuda import CUDABackend
        from triton.runtime.driver import driver

        def _make_sig_key_str(anno, key):
            if anno == 'bool':
                return '"i1"'
            if anno == 'float':
                return '"fp32"'
            if 'Tensor' in anno:
                return f'{key}.dtype'
            return f'JITFunction._key_of({key})'

        fn = jit_func.fn
        params = jit_func.params

        arg_key = tuple(p.name for p in params)
        sig_key = tuple(p.name for p in params if not p.is_constexpr)
        annotations = tuple(p.annotation for p in params if not p.is_constexpr)
        constexpr_key = tuple(p.name for p in params if p.is_constexpr)
        spec_key = tuple(p.name for p in params if not p.do_not_specialize)

        arg_key_str = ', '.join(arg_key)
        args_signature = ', '.join(
            p.name if p.default ==
            inspect._empty else f'{p.name} == {p.default}' for p in params)
        constexpr_key_str = ', '.join(constexpr_key)
        sig_name_str = ', '.join(key for key in sig_key)
        sig_key_str = ', '.join(
            _make_sig_key_str(anno, key)
            for anno, key in zip(annotations, sig_key))
        spec_key_str = ', '.join(f'_specialization_key({key})'
                                 for key in spec_key)
        src = f"""
def _{fn.__name__}_launcher({args_signature}, grid=None, **kwargs):
    device = driver.get_current_device()
    stream = driver.get_current_stream(device)
    target = driver.get_current_target()
    backend = CUDABackend(target)
    kwargs["debug"] = jit_func.debug
    options = backend.parse_options(kwargs)
    sig_key = ({sig_key_str}, )
    spec_key = ({spec_key_str}, )
    constexpr_key = ({constexpr_key_str}, )
    key = (get_cuda_version_key(), sig_key, constexpr_key, spec_key, options)

    cache = jit_func.cache[device]
    if key not in cache:
        return jit_func[grid]({arg_key_str}, **kwargs)

    kernel = cache[key]
    args = ({sig_name_str})
    if callable(grid):
        grid = grid(dict(bound_args.arguments))
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    kernel.run(grid_0, grid_1, grid_2, kernel.num_warps, kernel.num_ctas,
               kernel.cluster_dims[0], kernel.cluster_dims[1], kernel.cluster_dims[2],
               kernel.shared, stream, kernel.function, CompiledKernel.launch_enter_hook,
               CompiledKernel.launch_exit_hook, kernel,
               *driver.assemble_tensormap_to_arg(kernel.metadata["tensormaps_info"], args))

    return kernel
"""   # noqa: E501
        scope = dict(
            _specialization_key=self._specialization_key,
            get_cuda_version_key=get_cuda_version_key,
            driver=driver,
            CUDABackend=CUDABackend,
            CompiledKernel=CompiledKernel,
            JITFunction=JITFunction,
            jit_func=jit_func,
        )
        exec(src, scope)
        return scope[f'_{fn.__name__}_launcher']

    def __getitem__(self, grid):
        """get item."""
        return functools.partial(cast(Callable, self.run), grid=grid)


def wrap_jit_func(func):
    """wrap jit func."""
    triton_version = version.parse(triton.__version__)
    if triton_version == version.parse('2.3.0'):
        return JitFunction230Wrapper(func)
    return func
