# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _mbgmm_a_api(x: Tensor,
                 lora_a: Tensor,
                 q_start_loc: Tensor,
                 q_seqlens: Tensor,
                 adapter_ids: Tensor,
                 rank_offset: Tensor,
                 ranks: Tensor,
                 max_seq_len: int,
                 max_rank: int,
                 rank_step: int = 1):
    """mbgmm_a."""
    ...


def _mbgmm_b_api(xa: Tensor,
                 lora_b: Tensor,
                 q_start_loc: Tensor,
                 q_seqlens: Tensor,
                 adapter_ids: Tensor,
                 scaling: Tensor,
                 rank_offset: Tensor,
                 ranks: Tensor,
                 max_seq_len: int,
                 max_rank: int,
                 out_size: int = None):
    """mbgmm_b."""
    ...


mbgmm_a = FunctionDispatcher('mbgmm_a').make_caller(_mbgmm_a_api)

mbgmm_b = FunctionDispatcher('mbgmm_b').make_caller(_mbgmm_b_api)
