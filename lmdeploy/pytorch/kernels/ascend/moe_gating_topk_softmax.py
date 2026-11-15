# Copyright (c) OpenMMLab. All rights reserved.

from torch import Tensor

from . import ops as ext_ops
from .types import MoeMetadata as AscendMoeMetadata


def moe_gating_topk_softmax(router_logits: Tensor, topk: int, moe_metadata: AscendMoeMetadata) -> tuple[Tensor, Tensor]:
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(router_logits, topk, moe_metadata)
    return routing_weights, selected_experts
