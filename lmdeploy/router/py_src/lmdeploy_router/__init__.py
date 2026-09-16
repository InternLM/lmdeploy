from lmdeploy_router.version import __version__
from lmdeploy_router.router import ROUTER_AVAILABLE

__all__ = ["__version__"]

if ROUTER_AVAILABLE:
    from lmdeploy_router.router import Router as Router

    __all__.append("Router")
