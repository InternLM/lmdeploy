# Copyright (c) OpenMMLab. All rights reserved.
import functools
from abc import ABC, abstractmethod
from typing import Callable, List, Optional

import torch
import torch.distributed as dist


class SoftmaxTopKImpl(ABC):
    """Softmax topk implementation api."""

    @staticmethod
    @functools.lru_cache
    def get_group_offsets(n_groups: int, group_size: int, device: str):
        group_offsets = (torch.arange(n_groups, device=device) * group_size).view(1, -1, 1)  # [1, n_groups, 1]
        return group_offsets

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """forward."""
        raise NotImplementedError


class SoftmaxTopKBuilder(ABC):
    """Softmax topk implementation builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int, dim: int = -1, n_groups: int = -1):
        """build."""
        raise NotImplementedError


class FusedMoEImpl(ABC):
    """Fused moe implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    @abstractmethod
    def forward(self,
                hidden_states: torch.Tensor,
                topk_weights: torch.Tensor,
                topk_ids: torch.LongTensor,
                gate_up_weights: torch.Tensor,
                down_weights: torch.Tensor,
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
        """forward."""
        raise NotImplementedError


class FusedMoEBuilder(ABC):
    """Fused moe builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              hidden_dim: int = 1,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              layer_idx: int = 0,
              out_dtype: torch.dtype = torch.bfloat16):
        """Build from mlp."""
        raise NotImplementedError


class FusedMoEW8A8Impl(ABC):
    """Fused moe w8a8 implementation."""

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
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
    """Fused moe w8a8 builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              renormalize: bool = False,
              out_dtype: torch.dtype = torch.float16,
              quant_dtype: torch.dtype = torch.int8):
        """Build from mlp."""
        raise NotImplementedError


class FusedMoEBlockedF8Impl(ABC):
    """Fused moe blocked f8 implementation."""

    def __init__(self):
        self.scale_fmt: Optional[str] = None

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    def set_scale_fmt(self, scale_fmt: Optional[str]):
        """Set scale fmt."""
        self.scale_fmt = scale_fmt

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
                gate_up_bias: torch.Tensor = None,
                down_bias: torch.Tensor = None,
                expert_list: List[int] = None,
                act_func: Callable = None):
        """forward."""
        raise NotImplementedError


class FusedMoEBlockedF8Builder(ABC):
    """Fused moe blocked f8 builder."""

    @staticmethod
    @abstractmethod
    def build(top_k: int,
              num_experts: int,
              hidden_dim: int = 1,
              renormalize: bool = False,
              block_size: int = 128,
              ep_size: int = 1,
              ep_group: dist.ProcessGroup = None,
              out_dtype: torch.dtype = torch.float16,
              layer_idx: int = 0,
              custom_gateup_act: bool = False):
        """Build from mlp."""
        raise NotImplementedError
