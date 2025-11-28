# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
from typing import Callable

from lmdeploy.pytorch.utils import CtxMgrBase, singleton


@dataclass
class DeviceContext:
    device_type: str = 'cuda'


DefaultContext = DeviceContext()


@singleton
class DeviceManager(CtxMgrBase[DeviceContext]):

    def __init__(self):
        super().__init__(DefaultContext)
        self._context_callback: dict[int, Callable] = dict()
        self._next_cb_handle = 0

    def register_context_callback(self, callback: Callable):
        """Register callback."""
        handle = self._next_cb_handle
        self._context_callback[handle] = callback
        self._next_cb_handle += 1
        return handle

    def unregister_context_callback(self, handle: int):
        """Unregister callback."""
        self._context_callback.pop(handle, None)


def get_device_manager():
    """Get device manager."""
    return DeviceManager()
