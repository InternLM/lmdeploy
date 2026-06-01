# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int, method: str):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens
        self.method = method

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""

        # only eagle3 have two sets of cudagraph due to different target_hidden_size
        if num_tokens == origin_batch_size and self.method == 'eagle3':
            return batch_size

        return batch_size * (self.num_spec_tokens + 1)
