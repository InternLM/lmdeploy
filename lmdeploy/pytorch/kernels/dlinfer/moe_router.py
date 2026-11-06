# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from torch import Tensor


def moe_gating_top_k(
    router_logits: Tensor,
    bias: Tensor,
    *,
    top_k: int,
    k_group: int,
    group_count: int,
    renorm: int,
    norm_type: int,
    routed_scaling_factor: float,
    eps: float,
) -> tuple[Tensor, Tensor]:
    """Run the vendor fused grouped MoE router."""
    return ext_ops.moe_gating_top_k(
        router_logits,
        bias,
        top_k,
        k_group,
        group_count,
        renorm,
        norm_type,
        routed_scaling_factor,
        eps,
    )
