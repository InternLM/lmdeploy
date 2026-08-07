# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """Build backend-specific cache layouts and local primitives."""

    @classmethod
    @abstractmethod
    def build_block_layout(cls, resources, num_layers: int):
        """Select the physical layout for block-cache resources."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_state_layout(cls, resources):
        """Select the physical layout for state-cache resources."""
        raise NotImplementedError
