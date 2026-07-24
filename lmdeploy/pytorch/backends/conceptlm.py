# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod

from torch import Tensor


class ConceptLMRuntimeOpsImpl(ABC):
    """ConceptLM runtime operation implementation.

    Model-specific runtime/cache operations live behind this single backend
    interface. That keeps model code free from direct kernel calls while
    avoiding one OpType/nn module per small ConceptLM state operation.
    """

    @abstractmethod
    def decode_chunk_state_update(
        self,
        chunk_source_state_cache: Tensor,
        current_source_states: Tensor,
        state_ids: Tensor,
        position_ids: Tensor,
        chunk_size: int,
        merge_method: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update state cache and return concept inputs, next rows, and mask."""
        raise NotImplementedError('Not implemented.')


class ConceptLMRuntimeOpsBuilder(ABC):
    """ConceptLM runtime operation builder."""

    @staticmethod
    @abstractmethod
    def build() -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        raise NotImplementedError('Not implemented.')
