# Copyright (c) OpenMMLab. All rights reserved.
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .base import BuildSpec


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
                expert_list: list[int] = None,
                act_func: Callable = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class FusedMoEBuildSpec(BuildSpec[FusedMoEImpl]):
    """Immutable requirements for constructing a fused MoE operator."""

    top_k: int
    num_experts: int
    renormalize: bool
    hidden_dim: int
    ep_size: int
    ep_group: dist.ProcessGroup | None
    layer_idx: int
    out_dtype: torch.dtype
    num_max_dispatch_tokens_per_rank: int


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
                expert_list: list[int] = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class FusedMoEW8A8BuildSpec(BuildSpec[FusedMoEW8A8Impl]):
    """Immutable requirements for constructing a W8A8 fused MoE operator."""

    top_k: int
    num_experts: int
    renormalize: bool
    out_dtype: torch.dtype
    quant_dtype: torch.dtype | None


class FusedMoEStaticF8Impl(ABC):
    """Fused MoE static FP8 implementation."""

    def update_weights(
        self,
        gate_up_weights: torch.Tensor,
        gate_up_weight_scale: torch.Tensor,
        gate_up_input_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_weight_scale: torch.Tensor,
        down_input_scale: torch.Tensor,
    ):
        """Update weights and scales."""
        return (
            gate_up_weights,
            gate_up_weight_scale,
            gate_up_input_scale,
            down_weights,
            down_weight_scale,
            down_input_scale,
        )

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        gate_up_weights: torch.Tensor,
        gate_up_weight_scale: torch.Tensor,
        gate_up_input_scale: torch.Tensor,
        down_weights: torch.Tensor,
        down_weight_scale: torch.Tensor,
        down_input_scale: torch.Tensor,
        expert_list: list[int] = None,
    ):
        """Forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class FusedMoEStaticF8BuildSpec(BuildSpec[FusedMoEStaticF8Impl]):
    """Immutable requirements for constructing a static-FP8 fused MoE
    operator."""

    top_k: int
    num_experts: int
    renormalize: bool
    out_dtype: torch.dtype
    quant_dtype: torch.dtype


class FusedMoEBlockedF8Impl(ABC):
    """Fused moe blocked f8 implementation."""

    def __init__(self):
        self.scale_fmt: str | None = None

    def update_weights(self, gate_up_weights: torch.Tensor, down_weights: torch.Tensor, gate_up_scale: torch.Tensor,
                       down_scale: torch.Tensor):
        """Update weights."""
        return gate_up_weights, down_weights, gate_up_scale, down_scale

    def ep_expert_list(self, world_size: int, rank: int):
        """Experts list of current rank."""
        raise NotImplementedError('Not Implemented.')

    def set_scale_fmt(self, scale_fmt: str | None):
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
                expert_list: list[int] = None,
                act_func: Callable = None):
        """forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class FusedMoEBlockedF8BuildSpec(BuildSpec[FusedMoEBlockedF8Impl]):
    """Immutable requirements for constructing a blocked-FP8 fused MoE
    operator."""

    top_k: int
    num_experts: int
    hidden_dim: int
    renormalize: bool
    block_size: int
    ep_size: int
    ep_group: dist.ProcessGroup | None
    out_dtype: torch.dtype
    fp8_dtype: torch.dtype
    num_max_dispatch_tokens_per_rank: int
    layer_idx: int
    custom_gateup_act: bool
    scale_fmt: str | None


class FusedMoEV4FP4Impl(ABC):
    """DeepSeek-V4 FP4 fused MoE implementation API."""

    @abstractmethod
    def update_weights(
        self,
        gate_up_weight: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_weight: torch.Tensor,
        down_scale: torch.Tensor,
    ):
        """Update weights and scales."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
        gate_up_weight: torch.Tensor,
        gate_up_scale: torch.Tensor,
        down_weight: torch.Tensor,
        down_scale: torch.Tensor,
    ):
        """Forward."""
        raise NotImplementedError


@dataclass(frozen=True)
class FusedMoEV4FP4BuildSpec(BuildSpec[FusedMoEV4FP4Impl]):
    """Immutable requirements for constructing a V4 FP4 fused MoE operator."""

    top_k: int
    num_experts: int
    hidden_dim: int
    ffn_dim: int
    expert_offset: int
    swiglu_limit: float
    scale_fmt: str | None
    ep_size: int
    ep_group: dist.ProcessGroup | None
    layer_idx: int
    num_max_dispatch_tokens_per_rank: int
