# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod


class EngineStrategy(ABC):
    """Engine strategy."""

    @abstractmethod
    def get_prealloc_size(self, is_decoding: bool) -> int:
        """Get prealloc_size."""
        pass

    @abstractmethod
    def get_num_loops(self, is_decoding: bool) -> int:
        """Get num_loops."""
        pass

    @abstractmethod
    def get_num_decode_tokens(self) -> int:
        """Get num_decode_tokens."""
        pass

    def get_num_required_tokens(self) -> int:
        """Get num_require_tokens."""
        return self.get_num_decode_tokens()
