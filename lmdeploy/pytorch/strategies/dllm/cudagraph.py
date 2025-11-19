# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class DLLMCudagraphStrategy(CudagraphStrategy):

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.block_size = block_size

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        return batch_size * self.block_size
