# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        if num_tokens == origin_batch_size:
            return batch_size

        assert num_tokens % (self.num_spec_tokens + 1) == 0, 'The input_ids length must be divisible by batch_size.'
        return batch_size * (self.num_spec_tokens + 1)
