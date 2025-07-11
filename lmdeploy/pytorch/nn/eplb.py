# Copyright (c) OpenMMLab. All rights reserved.
import torch


class EPLBDispatchInfo:

    def __init__(self, info) -> None:
        self.info = info


class EPLBManager:
    eplb = None

    @classmethod
    def init_global_eplb_metadata(cls, ep_size: int, num_routed_experts: int, num_hidden_layers: int):
        assert ep_size > 1, 'eplb requires ep_size > 1'
        from dlblas.layers.moe import eplb
        EPLBManager.eplb = eplb
        eplb.init_global_eplb_metadata(ep_size=ep_size,
                                       num_routed_experts=num_routed_experts,
                                       num_hidden_layers=num_hidden_layers)

    @classmethod
    def num_physical_experts(cls) -> int:
        return EPLBManager.eplb.get_global_eplb_metadata().num_physical_experts()

    @classmethod
    def topk_ids_logical_to_physical(cls, topk_ids: torch.Tensor, eplb_dispatch_info: EPLBDispatchInfo):
        return EPLBManager.eplb.topk_ids_logical_to_physical(topk_ids=topk_ids, info=eplb_dispatch_info.info)

    @classmethod
    def get_dispatch_info(cls, ep_rank, layer_idx) -> EPLBDispatchInfo:
        info = EPLBManager.eplb.EPLBDispatchInfo.init_new(ep_rank=ep_rank, layer_idx=layer_idx)
        return EPLBDispatchInfo(info)
