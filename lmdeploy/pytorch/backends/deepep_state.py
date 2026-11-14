# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.utils import singleton


@singleton
class DeepEPState:

    def __init__(self):
        """Initialize DeepEP state."""
        self._enabled = False
        self.low_latency_prefill_tokens = 0

    def enable(self, low_latency_prefill_tokens: int = 0):
        """Register a layer's safe prefill capacity; unsupported layers opt
        out.

        All layers share the forward's communication mode, so the least capable layer determines whether small prefills
        can use low latency.
        """
        if self._enabled:
            low_latency_prefill_tokens = min(self.low_latency_prefill_tokens, low_latency_prefill_tokens)
        self.low_latency_prefill_tokens = low_latency_prefill_tokens
        self._enabled = True

    def enabled(self):
        """Return whether DeepEP is enabled."""
        is_ep_mode = get_dist_manager().current_context().dist_config.ep > 1
        return self._enabled and is_ep_mode


def get_deepep_state():
    return DeepEPState()
