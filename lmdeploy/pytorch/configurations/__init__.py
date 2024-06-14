# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import pkgutil

from .builder import AutoModelConfigBuilder

__all__ = []

# load all submodule
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    __all__.append(module_name)
    _module = importlib.import_module('{}.{}'.format(__name__, module_name))
    globals()[module_name] = _module

__all__ += ['AutoModelConfigBuilder']
