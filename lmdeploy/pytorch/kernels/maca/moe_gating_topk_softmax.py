# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def moe_gating_topk_softmax(router_logits: Tensor, topk: int):
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(
        router_logits, topk)
    return routing_weights, selected_experts
