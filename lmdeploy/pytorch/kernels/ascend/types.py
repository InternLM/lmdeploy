# Copyright (c) 2024, DeepLink. All rights reserved.
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
from torch import Tensor


class MoECommType(Enum):
    NAIVE = auto()
    ALLGATHER = auto()
    ALLTOALL = auto()
    MC2 = auto()
    TP = auto()
    UNDEFINED = auto()


@dataclass
class MoeMetadata:
    max_tokens_across_dp: int = 1
    pad_size: int = 0
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0
    ep_rank: int = 0
    tp_group: torch.distributed.ProcessGroup = None
    ep_group: torch.distributed.ProcessGroup = None
    moe_comm_type: MoECommType = MoECommType.UNDEFINED
    x_active_mask: torch.Tensor = None
    moe_group_name: str = None
    expert_ids_per_ep_rank: torch.Tensor = None
    router_n_groups: int = -1


linear_w8a8_scale_type = torch.Tensor
dynamic_quant_scale_type = torch.Tensor

__all__ = [
    'Tensor',
    'Optional',
    'Sequence',
    'MoECommType',
    'MoeMetadata',
    'linear_w8a8_scale_type',
    'dynamic_quant_scale_type',
]
