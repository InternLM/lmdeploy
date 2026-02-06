# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from dlinfer.utils.type_annotation import MoeMetadata as DlinferMoeMetadata
from torch import Tensor


def moe_gating_topk_softmax(router_logits: Tensor, topk: int,
                            moe_metadata: DlinferMoeMetadata) -> tuple[Tensor, Tensor]:
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(router_logits, topk, moe_metadata)
    return routing_weights, selected_experts
