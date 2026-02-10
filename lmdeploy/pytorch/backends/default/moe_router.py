# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Tuple

import torch

from ..moe_router import RouterNoauxTCBuilder, RouterNoauxTCImpl


def _compute_scores(scoring_func: str, logits: torch.Tensor):
    """Compute scores."""
    if scoring_func == 'softmax':
        scores = logits.softmax(dim=-1, dtype=torch.float32)
    elif scoring_func == 'sigmoid':
        scores = logits.sigmoid()
    else:
        raise NotImplementedError('insupportable scoring function '
                                  f'for MoE gating: {scoring_func}')
    return scores


@functools.lru_cache
def get_group_offsets(n_groups: int, group_size: int, device: str):
    group_offsets = (torch.arange(n_groups, device=device) * group_size).view(1, -1, 1)  # [1, n_groups, 1]
    return group_offsets


class DefaultRouterNoauxTCImpl(RouterNoauxTCImpl):

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

        self.scoring_func = scoring_func
        self.top_k = top_k
        self.n_group = n_group
        self.topk_group = topk_group
        self.n_routed_experts = n_routed_experts

        # renorm
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor

        # n_group
        self.router_n_groups = router_n_groups

    def _forward_router_n_groups(self, scores_for_choice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert scores_for_choice.shape[-1] % self.router_n_groups == 0, \
            f'{scores_for_choice.shape[-1]} cannot be divided by {self.router_n_groups}'
        per_group_top_k = self.top_k // self.router_n_groups
        group_size = scores_for_choice.shape[-1] // self.router_n_groups
        group_offsets = get_group_offsets(self.router_n_groups, group_size, device=scores_for_choice.device)
        scores_for_choice = scores_for_choice.unflatten(-1, (self.router_n_groups, group_size))
        topk_weight, topk_idx = torch.topk(scores_for_choice, per_group_top_k, dim=-1)
        topk_idx = (topk_idx + group_offsets).flatten(-2, -1)
        topk_weight = topk_weight.flatten(-2, -1)
        return topk_weight, topk_idx

    def _forward_default(self, scores: torch.Tensor, scores_for_choice: torch.Tensor,
                         sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        group_scores = (scores_for_choice.view(sequence_length, self.n_group,
                                               -1).topk(2, dim=-1)[0].sum(dim=-1))  # [n, n_group]
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, n_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
        score_mask = (group_mask.unsqueeze(-1).expand(sequence_length, self.n_group,
                                                      self.n_routed_experts // self.n_group).reshape(
                                                          sequence_length, -1))  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        return topk_weight, topk_idx

    def renorm(self, topk_weight: torch.Tensor) -> torch.Tensor:
        if self.renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
            if not topk_weight.is_contiguous():
                topk_weight = topk_weight.contiguous()

        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_weight

    def forward(self, logits: torch.Tensor, bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Router forward."""
        sequence_length = logits.shape[0]

        scores = _compute_scores(self.scoring_func, logits)
        scores_for_choice = scores.view(sequence_length, -1) + bias[None]
        if self.router_n_groups > 0:
            topk_weight, topk_idx = self._forward_router_n_groups(scores_for_choice)
        else:
            topk_weight, topk_idx = self._forward_default(scores, scores_for_choice, sequence_length)

        topk_weight = self.renorm(topk_weight)
        return topk_weight, topk_idx


class DefaultRouterNoauxTCBuilder(RouterNoauxTCBuilder):

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
    ):
        return DefaultRouterNoauxTCImpl(
            scoring_func=scoring_func,
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            n_routed_experts=n_routed_experts,
            routed_scaling_factor=routed_scaling_factor,
            renormalize=renormalize,
            router_n_groups=router_n_groups,
        )
