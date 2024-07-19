# Copyright (c) OpenMMLab. All rights reserved.
# modify from:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/backends/abstract.py
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple

import torch


class LayerType(Enum):
    Attention = auto()
    Linear = auto()
    RotaryEmbedding = auto()
    ApplyRotaryEmb = auto()
    SiluAndMul = auto()
    RMSNorm = auto()


class LayersBackend(ABC):

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def get_layer_impl_builder(cls, layer_type: LayerType):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_attention_metadata_cls():
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        raise NotImplementedError

    @classmethod
    def update_step_context(cls, step_context):
        return step_context
