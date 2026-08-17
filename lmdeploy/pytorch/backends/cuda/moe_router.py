# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.kernels.cuda.fused_noaux_tc import fused_noaux_tc_routing
from lmdeploy.pytorch.kernels.cuda.fused_single_group_router import (
    fused_single_group_topk_router,
)

from ..default.moe_router import DefaultRouterNoauxTCImpl
from ..moe_router import RouterNoauxTCBuilder, RouterNoauxTCImpl


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


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

        self.enable_custom_kernel = self.should_enable_custom_kernel()
        single_group_eligible = (
            not self.enable_custom_kernel
            and self.router_n_groups <= 0
            and self.scoring_func == 'sigmoid'
            and self.n_group == 1
            and self.topk_group == 1
        )
        self.enable_single_group_fused = (
            _envs.router_single_group_fused
            and single_group_eligible
            and self.n_routed_experts == 192
            and self.top_k == 8
            and self.renormalize
        )

    def should_enable_custom_kernel(self) -> bool:
        if self.router_n_groups > 0:
            return False

        if self.scoring_func != 'sigmoid':
            return False

        if self.n_routed_experts % 32 != 0:
            return False

        if self.n_routed_experts % self.n_group != 0:
            return False

        if not is_power_of_two(self.n_group):
            return False

        if is_power_of_two(self.n_routed_experts):
            return True

        return (
            self.n_routed_experts == 384
            and self.n_group == 1
            and self.topk_group == 1
            and self.top_k == 8
            and self.renormalize
            and self.routed_scaling_factor == 2.827
        )

    def _forward_single_group_fused(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse top-k routing while preserving PyTorch sigmoid."""
        scores = logits.sigmoid()
        return fused_single_group_topk_router(
            scores,
            bias,
            num_experts=self.n_routed_experts,
            top_k=self.top_k,
            routed_scaling_factor=(
                self.routed_scaling_factor
            ),
        )

    def forward(self, logits: torch.Tensor, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Router forward."""
        if self.enable_single_group_fused:
            return self._forward_single_group_fused(
                logits,
                bias,
            )
        elif self.enable_custom_kernel:
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
