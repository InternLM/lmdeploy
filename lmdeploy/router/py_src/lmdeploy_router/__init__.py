# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy_router.router import ROUTER_AVAILABLE
from lmdeploy_router.version import __version__

__all__ = ['__version__']

if ROUTER_AVAILABLE:
    from lmdeploy_router.router import Router as Router

    __all__.append('Router')
