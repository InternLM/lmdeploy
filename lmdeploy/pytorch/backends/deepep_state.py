# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.utils import singleton


@singleton
class DeepEPState:

    def __init__(self):
        """Initialize DeepEP state."""
        self._enabled = False

    def enable(self):
        """Mark DeepEP as enabled."""
        self._enabled = True

    def enabled(self):
        """Return whether DeepEP is enabled."""
        return self._enabled


def get_deepep_state():
    return DeepEPState()
