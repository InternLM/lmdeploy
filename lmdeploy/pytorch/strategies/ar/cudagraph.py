# Copyright (c) OpenMMLab. All rights reserved.
from ..base.cudagraph import CudagraphStrategy


class ARCudagraphStrategy(CudagraphStrategy):

    def get_max_tokens(self, batch_size: int) -> int:
        """Get max tokens."""
        return batch_size
