# Copyright (c) OpenMMLab. All rights reserved.
# modify from:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/abstract.py
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig


class OpType(Enum):
    """Layer type enumerate."""
    PagedAttention = auto()
    FlashAttention = auto()
    Linear = auto()
    RotaryEmbedding = auto()
    ApplyRotaryEmb = auto()
    SiluAndMul = auto()
    GeluAndMul = auto()
    RMSNorm = auto()
    LayerNorm = auto()
    LoRA = auto()
    LinearW8A8 = auto()
    RMSNormW8A8 = auto()
    MultinomialSampling = auto()
    LinearW4A16 = auto()
    SoftmaxTopK = auto()
    FusedMoE = auto()
    FusedMoEW8A8 = auto()
    LinearBlockedF8 = auto()
    FusedMoEBlockedF8 = auto()
    Embedding = auto()


class OpsBackend(ABC):
    """Layer backend abstract."""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get backend name."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_layer_impl_builder(cls, layer_type: OpType):
        """Get builder of given layer type."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_attention_metadata_cls():
        """Get attention metadata class."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """Get block shape of k."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        """Get block shape of v."""
        raise NotImplementedError

    @classmethod
    def update_step_context(cls, step_context):
        """Update StepContext for inference.

        attention meta should be built here.
        """
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        from .graph_runner import GraphRunner
        return GraphRunner(model, model_config, cache_config, backend_config, device)

    @staticmethod
    def device_count():
        """Get num available devices."""
        return None

    @staticmethod
    def support_ray():
        """Support ray."""
        return False
