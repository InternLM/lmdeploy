# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor
from . import DlinferDistContext


def moe_gating_topk_softmax(router_logits: Tensor, topk: int, dist_ctx: DlinferDistContext):
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(router_logits, topk, dist_ctx)
    return routing_weights, selected_experts
