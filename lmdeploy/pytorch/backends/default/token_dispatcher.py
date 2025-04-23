# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch

from ..token_dispatcher import TokenDispatcherImpl


class AlltoAllTokenDispatcher(TokenDispatcherImpl):

    def __init__(
        self,
        ep_group,
        num_experts,
        num_local_experts: int,
    ) -> None:
        self.num_local_experts = num_local_experts
        assert num_experts is not None
        self.num_experts = num_experts
        assert self.num_local_experts > 0, 'Expected at least one expert'
        self.ep_size = num_experts // num_local_experts
        self.ep_group = ep_group
        self.tp_size = 1
        self.input_splits = None
        self.output_splits = None
        input_chunk_idxs = torch.arange(self.num_experts, device=torch.device('cpu'))
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(-1, self.num_local_experts).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(self.num_local_experts, -1).T.ravel()

    def sort_chunks_by_idxs(self, input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
        """Split and sort the input tensor based on the split_sizes and sorted
        indices."""
        input = torch.split(input, split_sizes.tolist(), dim=0)
        output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
        return output

    def all_to_all(self, group: torch.distributed.group, input_: torch.Tensor, output_split: torch.Tensor,
                   input_split: torch.Tensor):
        output_split_sizes_ = output_split.tolist()
        input_split_sizes = input_split.tolist()
        output = input_.new_empty(
            size=[sum(output_split_sizes_)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            device=torch.cuda.current_device(),
        )
        torch.distributed.all_to_all_single(
            output,
            input_,
            output_split_sizes=output_split_sizes_,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    def preprocess(self, routing_map: torch.Tensor, local_expert_indices) -> torch.Tensor:
        assert (len(local_expert_indices) == self.num_local_experts), 'Invalid local expert indices'
        for i in range(len(local_expert_indices) - 1):
            assert (local_expert_indices[i] == local_expert_indices[i + 1] -
                    1), 'local_expert_indices must be continous'

        num_local_tokens_per_expert = routing_map.sum(dim=0).long()
        self.input_splits = (num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts).sum(axis=1).to(
            torch.device('cpu'), non_blocking=True))
        dim_size = list(num_local_tokens_per_expert.size())
        dim_size[0] = dim_size[0] * torch.distributed.get_world_size(self.ep_group)
        output = num_local_tokens_per_expert.new_empty(dim_size)
        torch.distributed.all_gather_into_tensor(output, num_local_tokens_per_expert.contiguous(), group=self.ep_group)
        num_global_tokens_per_expert = (output.reshape(self.ep_size, self.tp_size, self.num_experts).transpose(0, 1))
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, :, local_expert_indices[0]:
                                                                          local_expert_indices[-1] + 1].contiguous()
        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=2)
        self.output_splits = (num_global_tokens_per_rank[0].to(torch.device('cpu'), non_blocking=True))
        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=(0, 1))
        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts)

            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.to(torch.device('cpu'),
                                                                                            non_blocking=True)
        return num_tokens_per_local_expert

    def dispatch(self, hidden_states: torch.Tensor, topk_ids: torch.Tensor, probs: torch.Tensor,
                 local_expert_indices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        self.topk_ids = topk_ids
        self.routing_map, self.topk_weights = super().indices_to_multihot(topk_ids, probs, self.num_experts)
        assert probs.dim() == 2, 'Expected 2D tensor for probs'
        assert self.routing_map.dim() == 2, 'Expected 2D tensor for token2expert mask'
        assert self.routing_map.dtype == torch.bool, 'Expected bool tensor for mask'
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(self.routing_map, local_expert_indices)
        self.hidden_shape_before_permute = hidden_states.shape

        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = super().permute(
            hidden_states,
            self.routing_map,
        )
        global_input_tokens = self.all_to_all(self.ep_group, permutated_local_input_tokens, self.output_splits,
                                              self.input_splits)
        if self.num_local_experts > 1:
            global_input_tokens = self.sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )
        return global_input_tokens, None, None, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_local_experts > 1:
            hidden_states = self.sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.mT.ravel(),
                self.restore_output_by_local_experts,
            )
        permutated_local_input_tokens = self.all_to_all(self.ep_group, hidden_states, self.input_splits,
                                                        self.output_splits)
        output = super().unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            probs=self.topk_weights,
            routing_map=self.routing_map,
        )
        output = output.view(self.hidden_shape)
        return output
