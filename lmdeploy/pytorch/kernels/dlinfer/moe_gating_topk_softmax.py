# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import dlinfer.ops as ext_ops
from torch import Tensor


def moe_gating_topk_softmax(router_logits: Tensor, topk: int, moe_metadata: Any) -> tuple[Tensor, Tensor]:
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(router_logits, topk, moe_metadata)
    return routing_weights, selected_experts
