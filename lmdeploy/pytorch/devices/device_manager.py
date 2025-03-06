# Copyright (c) OpenMMLab. All rights reserved.
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable


@dataclass
class DeviceContext:
    device_type: str = 'cuda'


DefaultContext = DeviceContext()


class DeviceManager:

    def __init__(self):
        self.t_local = threading.local()
        self.t_local.device_context = DefaultContext
        self._context_callback: dict[int, Callable] = dict()
        self._next_cb_handle = 0

    def register_context_callback(self, callback: Callable):
        """register callback."""
        handle = self._next_cb_handle
        self._context_callback[handle] = callback
        self._next_cb_handle += 1
        return handle

    def unregister_context_callback(self, handle: int):
        """unregister callback."""
        self._context_callback.pop(handle, None)

    def current_context(self) -> DeviceContext:
        """get current context."""
        return getattr(self.t_local, 'device_context', DefaultContext)

    def set_context(self, context: DeviceContext):
        """set current context."""
        self.t_local.device_context = context
        for callback in self._context_callback.values():
            callback(context)

    @contextmanager
    def context(self, context: DeviceContext):
        """context manager."""
        origin_context = self.current_context()
        self.set_context(context)
        yield self
        self.set_context(origin_context)


_DEVICE_MANAGER: DeviceManager = None


def get_device_manager():
    """get device manager."""
    global _DEVICE_MANAGER
    if _DEVICE_MANAGER is None:
        _DEVICE_MANAGER = DeviceManager()
    return _DEVICE_MANAGER
