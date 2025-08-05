# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..base.cudagraph import CudagraphStrategy


class ARSpecCudagraphStrategy(CudagraphStrategy):

    def __init__(self, num_spec_tokens: int):
        super().__init__()
        self.num_spec_tokens = num_spec_tokens

    def get_max_tokens(self, batch_size: int, input_ids: torch.Tensor, q_seqlens: torch.Tensor) -> int:
        """Get max tokens."""
        num_tokens = input_ids.size(1)
        orig_batch = q_seqlens.size(0)
        if num_tokens == orig_batch:
            return batch_size

        assert num_tokens % (self.num_spec_tokens + 1) == 0, 'The input_ids length must be divisible by batch_size.'
        return batch_size * (self.num_spec_tokens + 1)
