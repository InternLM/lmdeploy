# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.utils import singleton


@singleton
class MoEBackend:

    def __init__(self):
        """Initialize moe backend."""
        self._use_deepep_moe_backend = False

    def set_deepep_moe_backend(self):
        """Set deepep moe backend."""
        self._use_deepep_moe_backend = True

    def use_deepep_moe_backend(self):
        """Get deepep moe backend."""
        return self._use_deepep_moe_backend


def get_moe_backend():
    return MoEBackend()
