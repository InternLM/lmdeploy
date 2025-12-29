# Copyright (c) OpenMMLab. All rights reserved.
import torch
from dataclasses import dataclass


@dataclass
class DlinferDistContext:
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1

    dp_rank: int = 0
    tp_rank: int = 0
    ep_rank: int = 0

    max_tokens_accros_dp: int = 1

    tp_group: torch.distributed.ProcessGroup = None
    ep_group: torch.distributed.ProcessGroup = None
