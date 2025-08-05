# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..base.cudagraph import CudagraphStrategy


class ARCudagraphStrategy(CudagraphStrategy):

    def get_max_tokens(self, batch_size: int, input_ids: torch.Tensor, q_seqlens: torch.Tensor) -> int:
        """Get max tokens."""
        return batch_size
