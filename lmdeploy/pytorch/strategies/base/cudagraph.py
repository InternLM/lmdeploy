# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

import torch


class CudagraphStrategy(ABC):

    @abstractmethod
    def get_max_tokens(self, batch_size: int, input_ids: torch.Tensor, q_seqlens: torch.Tensor) -> int:
        """Get max tokens."""
        pass
