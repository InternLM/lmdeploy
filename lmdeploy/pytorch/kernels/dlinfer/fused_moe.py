# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from dlinfer.utils.type_annotation import MoeMetadata as DlinferMoeMetada
from dlinfer.utils.type_annotation import MoeType as DlinferMoeType  # noqa: F401
from torch import Tensor


def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    moe_metadata: DlinferMoeMetada,
):
    """Dlinfer fused moe."""
    return ext_ops.fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk, renormalize,
                             moe_metadata)
