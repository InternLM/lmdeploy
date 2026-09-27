# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.moe_router import RouterGemmBuildSpec, RouterNoauxTCBuildSpec
from lmdeploy.pytorch.models.patch import get_build_model_context


class RouterGemm(torch.nn.Module):
    """Backend-dispatched router GEMM."""

    def __init__(self, out_dtype: torch.dtype | None = None):
        super().__init__()
        self.impl = get_backend().build_op(
            RouterGemmBuildSpec(output_dtype=out_dtype),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self, hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Compute router logits."""
        return self.impl.forward(hidden_states, weight)


class NoauxTCRouter(torch.nn.Module):

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
        super().__init__()

        self.impl = get_backend().build_op(
            RouterNoauxTCBuildSpec(
                scoring_func=scoring_func,
                top_k=top_k,
                n_group=n_group,
                top_k_group=topk_group,
                n_routed_experts=n_routed_experts,
                routed_scaling_factor=routed_scaling_factor,
                renormalize=renormalize,
                router_n_groups=router_n_groups,
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

    def forward(self, router_logits: torch.Tensor,
                e_score_correction_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Router forward."""
        return self.impl.forward(router_logits, e_score_correction_bias)
