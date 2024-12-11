# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def fused_moe(
    hidden_states: Tensor,
    top_k: int,
    topk_ids: Tensor,
    topk_weights: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
):
    """ascend fused moe."""
    return ext_ops.fused_moe(hidden_states, top_k, topk_ids, topk_weights,
                             gate_up_weights, down_weights)
