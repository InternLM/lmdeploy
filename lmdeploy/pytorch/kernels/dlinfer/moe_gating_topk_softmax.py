# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def moe_gating_topk_softmax(router_logits: Tensor, topk: int, max_tokens_across_dp: int, pad_size: int, tp_size: int,
                            ep_size: int, tp_rank: int) -> tuple[Tensor, Tensor]:
    routing_weights, selected_experts = ext_ops.moe_gating_topk_softmax(router_logits, topk, max_tokens_across_dp,
                                                                        pad_size, tp_size, ep_size, tp_rank)
    return routing_weights, selected_experts
