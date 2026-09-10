# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int, draft_arch: str | None = None):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens
        self.uniform_query_len = draft_arch == 'MiMoV2FlashMTPModel'

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        if num_tokens == origin_batch_size:
            return batch_size

        if self.uniform_query_len:
            if num_tokens % origin_batch_size != 0:
                raise ValueError('Speculative CUDA Graph requires a uniform query length per batch.')
            query_len = num_tokens // origin_batch_size
            return batch_size * query_len

        return batch_size * (self.num_spec_tokens + 1)
