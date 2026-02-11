# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing

from ..default.moe_router import DefaultRouterNoauxTCImpl
from ..moe_router import RouterNoauxTCBuilder, RouterNoauxTCImpl


class TritonRouterNoauxTCImpl(DefaultRouterNoauxTCImpl):

    def __init__(
        self,
        scoring_func: str,
        top_k: int,
        n_group: int,
        topk_group: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        renormalize: bool = True,
        router_n_groups: int = -1,
    ):
        super().__init__(
            scoring_func=scoring_func,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            router_n_groups=router_n_groups,
        )

    def forward(self, logits: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Router forward."""
        if self.router_n_groups <= 0 and self.scoring_func == 'sigmoid' and self.n_routed_experts % 32 == 0:
            return fused_noaux_tc_routing(
                logits,
                bias,
                num_experts=self.n_routed_experts,
                n_group=self.n_group,
                topk_group=self.topk_group,
                top_k=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            return super().forward(logits, bias)


class TritonRouterNoauxTCBuilder(RouterNoauxTCBuilder):

    @staticmethod
    def build(
        scoring_func: str,
        top_k: int,
        n_group: int,
        topk_group: int,
        n_routed_experts: int,
        routed_scaling_factor: float,
        renormalize: bool = True,
        router_n_groups: int = -1,
    ) -> RouterNoauxTCImpl:
        return TritonRouterNoauxTCImpl(
            scoring_func=scoring_func,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            router_n_groups=router_n_groups,
        )
