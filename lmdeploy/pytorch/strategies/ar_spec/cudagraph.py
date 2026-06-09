# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int, method: str):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens
        self.method = method

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""

        # Draft speculative decoding has both wide verification forwards and
        # single-token iterative forwards. Keep their graph buffers shaped like
        # the runtime query layout even when target_hidden_size is identical.
        if num_tokens == origin_batch_size:
            return batch_size

        return batch_size * (self.num_spec_tokens + 1)
