# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.utils import try_import_deeplink

from .base import BaseChecker


class DeeplinkChecker(BaseChecker):
    """Check pytorch is available."""

    def __init__(self, device_type: str, logger=None) -> None:
        super().__init__(logger=logger)
        self.device_type = device_type

    def check(self):
        """check."""
        try_import_deeplink(self.device_type)
