# Copyright (c) OpenMMLab. All rights reserved.
import threading
from contextlib import contextmanager
from dataclasses import dataclass

from torch import distributed as dist


@dataclass
class DistContext:
    rank: int = 0
    world_size: int = 1
    dist_group: dist.ProcessGroup = None


DefaultContext = DistContext()


class DistManager:
    """distributed context manager."""

    def __init__(self):
        self.t_local = threading.local()
        self.t_local.device_context = DefaultContext

    def current_context(self) -> DistContext:
        """get current context."""
        return getattr(self.t_local, 'device_context', DefaultContext)

    def set_context(self, context: DistContext):
        """set current context."""
        self.t_local.device_context = context

    @contextmanager
    def context(self, context: DistContext):
        """context manager."""
        origin_context = self.current_context()
        self.set_context(context)
        yield self
        self.set_context(origin_context)


_DIST_MANAGER: DistManager = None


def get_dist_manager():
    """get device manager."""
    global _DIST_MANAGER
    if _DIST_MANAGER is None:
        _DIST_MANAGER = DistManager()
    return _DIST_MANAGER


def get_world_rank():
    """get distributed world size and rank."""
    ctx = get_dist_manager().current_context()
    world_size = ctx.world_size
    rank = ctx.rank

    return world_size, rank


def get_process_group():
    """get process group."""
    ctx = get_dist_manager().current_context()
    return ctx.dist_group
