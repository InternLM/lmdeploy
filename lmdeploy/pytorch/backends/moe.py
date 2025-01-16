# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import List

import torch


class SoftmaxTopKImpl(ABC):
    """Softmax topk implementation api."""

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """forward."""
        raise NotImplementedError


class SoftmaxTopKBuilder(ABC):
    """Softmax topk implementation builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, dim: int = -1):
        """build."""
        raise NotImplementedError


class FusedMoEImpl(ABC):
    """fused moe implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor,
                       down_weights: torch.Tensor):
        """update weights."""
        return gate_up_weights, down_weights

    def support_ep(self):
        """support expert parallelism."""
        return False

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        raise NotImplementedError


class FusedMoEBuilder(ABC):
    """fused moe builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, num_experts: int, renormalize: bool = False):
        """build from mlp."""
        raise NotImplementedError


class FusedMoEW8A8Impl(ABC):
    """fused moe w8a8 implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor,
                       down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def support_ep(self):
        """support expert parallelism."""
        return False

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                input_scale: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        raise NotImplementedError


class FusedMoEW8A8Builder(ABC):
    """fused moe w8a8 builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              out_dtype: torch.dtype = torch.float16,
              quant_dtype: torch.dtype = torch.int8):
        """build from mlp."""
        raise NotImplementedError


class FusedMoEBlockedF8Impl(ABC):
    """fused moe blocked f8 implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor,
                       down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def support_ep(self):
        """support expert parallelism."""
        return False

    def ep_expert_list(self, world_size: int, rank: int):
        """experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                input_scale: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                gate_up_scale: torch.Tensor,
                down_weights: torch.Tensor,
                down_scale: torch.Tensor,
                expert_list: List[int] = None):
        """forward."""
        raise NotImplementedError


class FusedMoEBlockedF8Builder(ABC):
    """fused moe blocked f8 builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              out_dtype: torch.dtype = torch.float16):
        """build from mlp."""
        raise NotImplementedError
