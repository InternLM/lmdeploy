# Copyright (c) OpenMMLab. All rights reserved.
import torch

from lmdeploy.pytorch.kernels.dlinfer import moe_gating_top_k

from ..default.moe_router import DefaultRouterNoauxTCImpl


class DlinferRouterNoauxTCImpl(DefaultRouterNoauxTCImpl):
    """Ascend no-aux router backed by ``npu_moe_gating_top_k``.

    The fused kernel implements the grouped top-2 score selection used by
    DeepSeek/GLM no-aux routing. Unsupported configurations retain the
    default PyTorch implementation so this backend remains compatible with
    other model variants.
    """

    def forward(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens with the fused Ascend top-k kernel."""
        expert_count = logits.shape[-1]
        if (
            self.router_n_groups > 0
            or self.scoring_func not in ('softmax', 'sigmoid')
            or self.n_group <= 0
            or self.topk_group <= 0
            or self.topk_group > self.n_group
            or self.top_k < 1
            or expert_count % self.n_group != 0
            or expert_count // self.n_group <= 2
            or self.top_k * self.n_group * self.topk_group > expert_count
            or self.topk_group * (expert_count // self.n_group) < self.top_k
            or (self.scoring_func == 'sigmoid' and not self.renormalize)
        ):
            return super().forward(logits, bias)

        return moe_gating_top_k(
            logits,
            bias,
            top_k=self.top_k,
            k_group=self.topk_group,
            group_count=self.n_group,
            renorm=int(self.renormalize),
            norm_type=0 if self.scoring_func == 'softmax' else 1,
            routed_scaling_factor=self.routed_scaling_factor,
            eps=1e-20,
        )
