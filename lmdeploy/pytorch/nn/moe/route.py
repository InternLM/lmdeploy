# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from lmdeploy.pytorch.backends import OpType, get_backend


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

        impl_builder = get_backend().get_layer_impl_builder(OpType.RouterNoauxTC)
        self.impl = impl_builder.build(
            scoring_func=scoring_func,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            router_n_groups=router_n_groups,
        )

    def forward(self, router_logits: torch.Tensor,
                e_score_correction_bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Router forward."""
        return self.impl.forward(router_logits, e_score_correction_bias)
