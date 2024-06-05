# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from .dispatcher import FunctionDispatcher


def _mbgmv_a_api(x: Tensor,
                 lora_a: Tensor,
                 adapter_ids: Tensor,
                 rank_offset: Tensor,
                 ranks: Tensor,
                 max_rank: int,
                 rank_step: int = 1):
    """mbgmv_a."""
    ...


def _mbgmv_b_api(xa: Tensor,
                 lora_b: Tensor,
                 adapter_ids: Tensor,
                 scaling: Tensor,
                 rank_offset: Tensor,
                 ranks: Tensor,
                 max_rank: int,
                 out_size: int = None):
    """mbgmv_b."""
    ...


mbgmv_a = FunctionDispatcher('mbgmv_a').make_caller(_mbgmv_a_api)

mbgmv_b = FunctionDispatcher('mbgmv_b').make_caller(_mbgmv_b_api)
