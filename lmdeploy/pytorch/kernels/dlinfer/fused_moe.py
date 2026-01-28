# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
import torch.distributed as dist
from dlinfer.utils.type_annotation import MoeType
from torch import Tensor


def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    pad_size: int,
    tp_size: int,
    ep_size: int,
    tp_rank: int,
    ep_rank: int,
    tp_group: dist.ProcessGroup,
    ep_group: dist.ProcessGroup,
    moe_type: MoeType,
    x_active_mask: Tensor,
    moe_group_name: str,
    expert_ids_per_ep_rank: Tensor,
):
    """Dlinfer fused moe."""
    return ext_ops.fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk, renormalize,
                             pad_size, tp_size, ep_size, tp_rank, ep_rank, tp_group, ep_group, moe_type, x_active_mask,
                             moe_group_name, expert_ids_per_ep_rank)
