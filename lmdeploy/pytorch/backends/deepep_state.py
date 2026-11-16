# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.distributed import get_dist_manager
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
        is_ep_mode = get_dist_manager().current_context().dist_config.ep > 1
        return self._enabled and is_ep_mode


def get_deepep_state():
    return DeepEPState()
