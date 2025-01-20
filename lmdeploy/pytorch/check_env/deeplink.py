# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseChecker

deeplink_device_type_list = [
    'ascend',
    'npu',
    'maca',
    'camb',
]


class DeeplinkChecker(BaseChecker):
    """check pytorch is available."""

    def __init__(self, device_type: str, logger=None) -> None:
        super().__init__(logger=logger)
        self.device_type = device_type

    def check(self):
        """check."""
        device_type = self.device_type
        if device_type in deeplink_device_type_list:
            try:
                import dlinfer.framework.lmdeploy_ext  # noqa: F401
            except Exception as e:
                self.log_and_exit(e, 'dlinfer', 'dlinfer is not available.')
