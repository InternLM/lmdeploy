# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import inspect
from typing import Callable

from lmdeploy.utils import get_logger

from ..devices import DeviceContext, get_device_manager

logger = get_logger('lmdeploy')


def _default_api(*args, **kwargs):
    """default api."""
    ...


class ParamParser:

    def __init__(self, param: inspect.Parameter) -> None:
        self.param = param

    def name(self):
        """name."""
        return self.param.name

    def func_arg(self):
        """func arg."""
        param = self.param
        name = self.name()
        kind = param.kind
        ret = name
        if kind == inspect.Parameter.VAR_POSITIONAL:
            ret = f'*{name}'
        elif kind == inspect.Parameter.VAR_KEYWORD:
            ret = f'**{name}'

        default = param.default
        if default != inspect._empty:
            ret = f'{ret}={default}'

        return ret

    def func_input(self):
        """func input."""
        param = self.param
        name = self.name()
        kind = param.kind
        ret = name
        if kind == inspect.Parameter.VAR_POSITIONAL:
            ret = f'*{name}'
        elif kind == inspect.Parameter.VAR_KEYWORD:
            ret = f'**{name}'
        else:
            ret = f'{name}={name}'
        return ret


class FunctionDispatcher:

    def __init__(self, func_name: str):
        self.device_manager = get_device_manager()
        self.impl_map: dict[str, Callable] = dict()
        self.func_name = func_name
        self.dispatched_func = self.load_and_call
        self.device_manager.register_context_callback(self.device_callback)

    def device_callback(self, context: DeviceContext):
        """device context callback."""
        self.dispatched_func = self.load_and_call

    def load_func(self, device: str):
        """load function."""
        try:
            mod = importlib.import_module(f'lmdeploy.pytorch.kernels.{device}')
            func = getattr(mod, self.func_name)
            self.impl_map[device] = func
        except Exception:
            logger.debug(f'Failed to load <{self.func_name}>'
                         f' for <{device}>, '
                         'try load default implementation.')
            mod = importlib.import_module('lmdeploy.pytorch.kernels.default')
            if not hasattr(mod, self.func_name):
                raise RuntimeError(f'<{self.func_name}> default and <{device}>'
                                   ' implementation not exists.')
            func = getattr(mod, self.func_name)
            self.impl_map[device] = func

    def load_and_call(self, *args, **kwargs):
        """load and call."""
        device = self.device_manager.current_context().device_type
        if device not in self.impl_map:
            self.load_func(device)
        self.dispatched_func = self.impl_map[device]
        return self.dispatched_func(*args, **kwargs)

    def make_caller(self, api: Callable = _default_api, globals=None):
        """make call function."""
        signature = inspect.signature(api)
        params = signature.parameters

        param_parsers = [ParamParser(p) for p in params.values()]
        func_args = [p.func_arg() for p in param_parsers]
        func_inputs = [p.func_input() for p in param_parsers]
        func_args = ', '.join(func_args)
        func_inputs = ', '.join(func_inputs)

        src = f"""
def {self.func_name}({func_args}):
    return dispatcher.dispatched_func({func_inputs})
"""   # noqa: E501

        scope = dict(dispatcher=self, )
        if globals is not None:
            scope.update(globals)
        exec(src, scope)
        return scope[f'{self.func_name}']
