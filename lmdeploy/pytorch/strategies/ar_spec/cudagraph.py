# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int, method: str):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens
        self.method = method

    def get_max_tokens(self, batch_size: int, origin_batch_size: int, num_tokens: int) -> int:
        """Get max tokens."""
        if origin_batch_size <= 0 or num_tokens % origin_batch_size != 0:
            raise ValueError('AR speculative CUDA graphs require a rectangular '
                             f'query block, got num_tokens={num_tokens}, '
                             f'origin_batch_size={origin_batch_size}.')
        # Derive the query width from the actual call rather than assuming the
        # target verifier's N+1 layout. DFlash-family draft graphs can use a
        # distinct width (DSpark sample-from-anchor uses N), and both graph
        # buffers must retain their own fixed token capacity.
        query_len = num_tokens // origin_batch_size
        return batch_size * query_len
