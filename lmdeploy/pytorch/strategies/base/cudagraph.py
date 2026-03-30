# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod


class CudagraphStrategy(ABC):

    @abstractmethod
    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        pass
