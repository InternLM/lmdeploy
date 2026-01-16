# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from typing import TYPE_CHECKING

from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.serve.core import AsyncEngine

logger = get_logger('lmdeploy')


class InferInstManager:
    """Manages inference instances."""

    def __init__(self, engine: 'AsyncEngine', size: int):
        self.engine = engine.engine
        self.size = size
        self.insts = [self.engine.create_instance() for _ in range(size)]
        # `asyncio.Queue` must be created in an async context, refer to `get` method
        self.pool: asyncio.Queue = None

    def get(self):
        """Get the instance pool."""
        # Lazy initialization: create pool on first use
        if self.pool is None:
            self.pool = asyncio.Queue()
            for inst in self.insts:
                self.pool.put_nowait(inst)

        return self.pool

    def ret(self, inst):
        """Return a generator to the pool."""
        if inst is not None and self.pool is not None:
            self.pool.put_nowait(inst)

    def rebuild(self, engine):
        """Rebuild all generator instances.

        Used after wakeup for turbomind backend.
        """
        self.insts = [engine.create_instance() for _ in range(self.size)]
        self.pool = None

    def clear(self):
        """Clear all instances."""
        self.insts = []
        self.pool = None
