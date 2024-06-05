# Copyright (c) OpenMMLab. All rights reserved.
import torch

from .dispatcher import FunctionDispatcher


def _rearange_all_gather_api(x: torch.Tensor,
                             b_start_loc: torch.Tensor,
                             b_seq_lens: torch.Tensor,
                             adapter_ids: torch.LongTensor,
                             ranks: torch.Tensor,
                             world_size: int,
                             max_seq_len: int,
                             output: torch.Tensor = None):
    """rearange all gather."""
    ...


rearange_all_gather = FunctionDispatcher('rearange_all_gather').make_caller(
    _rearange_all_gather_api)
