# Copyright (c) OpenMMLab. All rights reserved.
import torch
import dlinfer.ops as ext_ops
from typing import List
from torch import Tensor


def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    ep_size: int,
    ep_group: torch.distributed.ProcessGroup = None,
):
    """Dlinfer fused moe."""
    return ext_ops.fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk, renormalize, ep_size, ep_group)
