# Copyright (c) OpenMMLab. All rights reserved.
from torch import Tensor, nn

from lmdeploy.pytorch.backends import OpType, get_backend


class ConceptLMRuntimeOps(nn.Module):
    """ConceptLM model-specific runtime operation wrapper.

    The model calls this nn module only. Backend implementations own dispatch,
    and CUDA implementations own direct Triton kernel launchers.
    """

    def __init__(self):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ConceptLMRuntimeOps)
        self.impl = builder.build()

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
        return self.impl.decode_chunk_state_update(
            chunk_source_state_cache,
            current_source_states,
            state_ids,
            position_ids,
            chunk_size,
            merge_method,
        )

    def forward(
        self,
        chunk_source_state_cache: Tensor,
        current_source_states: Tensor,
        state_ids: Tensor,
        position_ids: Tensor,
        chunk_size: int,
        merge_method: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Alias the current runtime op for module-call compatibility."""
        return self.decode_chunk_state_update(
            chunk_source_state_cache,
            current_source_states,
            state_ids,
            position_ids,
            chunk_size,
            merge_method,
        )
