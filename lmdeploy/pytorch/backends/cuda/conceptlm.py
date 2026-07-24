# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor

from lmdeploy.pytorch.kernels.cuda.conceptlm import decode_chunk_state_update

from ..conceptlm import ConceptLMRuntimeOpsBuilder, ConceptLMRuntimeOpsImpl


class TritonConceptLMRuntimeOpsImpl(ConceptLMRuntimeOpsImpl):
    """Triton implementation of ConceptLM runtime operations."""

    def decode_chunk_state_update(
        self,
        chunk_source_state_cache: Tensor,
        current_source_states: Tensor,
        state_ids: Tensor,
        position_ids: Tensor,
        chunk_size: int,
        merge_method: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update state cache and return fixed-shape decode rows."""
        return decode_chunk_state_update(
            chunk_source_state_cache,
            current_source_states,
            state_ids,
            position_ids,
            chunk_size,
            merge_method,
        )


class TritonConceptLMRuntimeOpsBuilder(ConceptLMRuntimeOpsBuilder):
    """Triton ConceptLM runtime operation builder."""

    @staticmethod
    def build() -> ConceptLMRuntimeOpsImpl:
        """Build layer implementation."""
        return TritonConceptLMRuntimeOpsImpl()
