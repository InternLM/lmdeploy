# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Tuple

import torch


class TokenDispatcherImpl(ABC):
    """Token dispatcher implementation api."""

    def permute(
        self,
        tokens,
        routing_map,
    ):
        """Copy from Megatron-Core moe for token permutation."""
        num_tokens, _ = tokens.shape
        num_experts = routing_map.shape[1]
        routing_map = routing_map.bool().T.contiguous()
        token_indices = (torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1))
        sorted_indices = token_indices.masked_select(routing_map)
        permuted_input = tokens.index_select(0, sorted_indices)
        return permuted_input, sorted_indices

    def unpermute(
        self,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
        probs: torch.Tensor = None,
        routing_map: torch.Tensor = None,
    ):
        """Copy from Megatron-Core moe for token unpermutation."""
        _, hidden = restore_shape
        if probs is not None:
            assert routing_map is not None, 'Mask must be provided to permute the probs.'
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
            permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
        output_tokens = torch.zeros(restore_shape, device=permuted_tokens.device, dtype=permuted_tokens.dtype)
        output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
        return output_tokens

    def indices_to_multihot(self, topk_ids, topk_weight, num_experts):
        tokens = topk_ids.shape[0]
        multihot_routing_map = torch.zeros(
            (tokens, num_experts),
            dtype=torch.bool,
            device=topk_ids.device,
        )

        multihot_probs = torch.zeros(
            (tokens, num_experts),
            dtype=topk_weight.dtype,
            device=topk_weight.device,
        )

        mask = topk_ids != -1
        valid_indices = topk_ids[mask]
        row_indices = torch.arange(tokens, device=topk_ids.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = True
        multihot_probs[row_indices, valid_indices] = topk_weight[mask]
        return multihot_routing_map, multihot_probs

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor, topk_ids: torch.Tensor,
                 local_expert_indices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """dispatch."""
        raise NotImplementedError

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """combine."""
        raise NotImplementedError
