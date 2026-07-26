# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
from torch import Tensor, nn

from lmdeploy.pytorch.backends import OpType, get_backend
from lmdeploy.pytorch.backends.conceptlm import (
    ConceptChunkInput,
    ConceptDecoderInput,
    ConceptForwardContext,
    ConceptRuntimeCaches,
)


class ConceptLMRuntimeOps(nn.Module):
    """ConceptLM model-specific runtime operation wrapper.

    The model calls this nn module only. Backend implementations own dispatch, and CUDA implementations own direct
    Triton kernel launchers.
    """

    def __init__(self, config):
        super().__init__()
        backend = get_backend()
        builder = backend.get_layer_impl_builder(OpType.ConceptLMRuntimeOps)
        self.impl = builder.build(config)

    def flatten_decode_position_ids(self, position_ids: Tensor, batch_size: int, device: torch.device) -> Tensor:
        """Normalize decode position ids to one absolute position per batch
        row."""
        return self.impl.flatten_decode_position_ids(position_ids, batch_size, device)

    def build_concept_chunk_input(
        self,
        source_states: Tensor,
        token_attn_metadata: Any,
        position_ids: Tensor,
        state_ids: Tensor | None = None,
        chunk_source_state_cache: Tensor | None = None,
    ) -> ConceptChunkInput:
        """Build concept-predictor source rows for prefill or decode."""
        return self.impl.build_concept_chunk_input(
            source_states,
            token_attn_metadata,
            position_ids,
            state_ids=state_ids,
            chunk_source_state_cache=chunk_source_state_cache,
        )

    def begin_concept_forward(self, chunk_input: ConceptChunkInput,
                              runtime_caches: ConceptRuntimeCaches) -> ConceptForwardContext:
        """Prepare transient state before the concept predictor forward."""
        return self.impl.begin_concept_forward(chunk_input, runtime_caches)

    def end_concept_forward(
        self,
        chunk_input: ConceptChunkInput,
        runtime_caches: ConceptRuntimeCaches,
        forward_context: ConceptForwardContext,
        source_states: Tensor,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> None:
        """Commit concept-predictor side effects for prefill or decode."""
        return self.impl.end_concept_forward(
            chunk_input,
            runtime_caches,
            forward_context,
            source_states,
            predicted_vectors,
            concept_raw_states,
        )

    def build_decoder_concept_input(
        self,
        chunk_input: ConceptChunkInput,
        runtime_caches: ConceptRuntimeCaches,
        forward_context: ConceptForwardContext,
        predicted_vectors: Tensor,
        concept_raw_states: list[Tensor],
    ) -> ConceptDecoderInput:
        """Build token-decoder concept inputs for prefill or decode."""
        return self.impl.build_decoder_concept_input(
            chunk_input,
            runtime_caches,
            forward_context,
            predicted_vectors,
            concept_raw_states,
        )
