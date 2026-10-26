# Copyright (c) OpenMMLab. All rights reserved.
import dlinfer.ops as ext_ops
from dlinfer.utils.type_annotation import MoECommType as DlinferMoECommType  # noqa: F401
from dlinfer.utils.type_annotation import MoeMetadata as DlinferMoeMetadata
from torch import Tensor


def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    moe_metadata: DlinferMoeMetadata,
    chunked_moe_layout=None,
):
    """Dlinfer fused moe."""
    return ext_ops.fused_moe(hidden_states, gate_up_weights, down_weights, topk_weights, topk_ids, topk, renormalize,
                             moe_metadata, chunked_moe_layout)


def fused_moe_w8a8(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    gate_up_scales: Tensor,
    down_weights: Tensor,
    down_scales: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    moe_metadata: DlinferMoeMetadata,
    use_fused_op: bool = False,
):
    """Dlinfer dynamic W8A8 MoE using the Ascend public-op fallback."""
    return ext_ops.fused_moe_w8a8(hidden_states, gate_up_weights, gate_up_scales, down_weights, down_scales,
                                  topk_weights, topk_ids, topk, renormalize, moe_metadata, use_fused_op)


def pack_w8a8_scales(gate_up_scales: Tensor, down_scales: Tensor):
    """Pack scales for the Ascend dispatch_ffn_combine path."""
    from dlinfer.vendor.ascend.moe import pack_w8a8_scales as _pack
    return _pack(gate_up_scales, down_scales)


def weights_in_fused_layout(weights: Tensor) -> bool:
    """True when the expert weights were converted to the fused NZ layout."""
    if weights.device.type != 'npu':
        return False
    import torch_npu
    return torch_npu.get_npu_format(weights) == 29
