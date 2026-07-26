# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.backends.conceptlm import (
    ConceptChunkStateUpdateResult,
    ConceptDecodeMetadata,
    ConceptPrefillMetadata,
)
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ConceptLMRuntimeOps
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from ..patch import add_prefix
from ..utils.cudagraph import CudaGraphMixin
from ..utils.model import DeployModelMixinV1
from .metadata import (
    ConceptCaches,
    ConceptMetadata,
)
from .modules import (
    ConceptPredictor,
    Embedding,
    OlmoBlock,
    Quantizer,
    ResidualRoute,
    SelfDD,
    TwoRouteAdd,
)
from .weight import _load_stacked_codebook_weight


@dataclass
class _ConceptPredictorOutput:
    """Concept predictor output shared by prefill and decode paths."""

    predicted_vectors: torch.Tensor
    raw_states: list[torch.Tensor]


@dataclass
class _ConceptPredictorRequest:
    """Concept-stream inputs consumed by the concept predictor."""

    hidden_states: torch.Tensor
    encoder_states: torch.Tensor
    position_ids: torch.Tensor
    attn_metadata: Any


@dataclass
class _DecoderConceptInput:
    """Concept inputs consumed by the decoder stack."""

    final_state: torch.Tensor
    raw_states: list[torch.Tensor] | None = None
    prefill_metadata: ConceptPrefillMetadata | None = None
    decode_states: torch.Tensor | None = None

    @classmethod
    def for_prefill(cls,
                    final_state: torch.Tensor,
                    raw_states: list[torch.Tensor],
                    prefill_metadata: ConceptPrefillMetadata):
        """Build decoder concept inputs from compact prefill predictor
        states."""
        return cls(
            final_state=final_state,
            raw_states=raw_states,
            prefill_metadata=prefill_metadata,
        )

    @classmethod
    def for_decode(cls, final_state: torch.Tensor, decode_states: torch.Tensor):
        """Build decoder concept inputs from already gathered decode states."""
        return cls(
            final_state=final_state,
            decode_states=decode_states,
        )


@dataclass
class _EncoderOutput:
    """Encoder result plus its reusable layer-major SelfDD history buffer."""

    hidden_states: torch.Tensor
    raw_states: list[torch.Tensor]
    history_buffer: torch.Tensor


class ConceptLMV22VQForCausalLM(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """Rewrote model of ConceptLMV22VQForCausalLM."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        # token embedding — mirrors ``self.embedding`` in the reference.
        self.embedding = Embedding(config, dtype=dtype, device=device)
        self.encoder = OlmoBlock(config,
                                 config.concept_encoder_layers,
                                 post_layer_norm=False,
                                 dtype=dtype,
                                 device=device,
                                 prefix=add_prefix('encoder', prefix))
        self.decoder = OlmoBlock(config,
                                 config.concept_decoder_layers,
                                 post_layer_norm=True,
                                 dtype=dtype,
                                 device=device,
                                 prefix=add_prefix('decoder', prefix))
        self.concept_vq_input_norm = nn.LayerNorm(config.hidden_size,
                                                  eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                                  dtype=dtype,
                                                  device=device)
        self.concept_quantizer = Quantizer(config, dtype=dtype, device=device)
        self.concept_predictor = ConceptPredictor(config,
                                                  dtype=dtype,
                                                  device=device,
                                                  prefix=add_prefix('concept_predictor', prefix))
        self.fusion_tok_norm = nn.LayerNorm(config.hidden_size,
                                            eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                            dtype=dtype,
                                            device=device)
        self.fusion_hl_norm = nn.LayerNorm(config.hidden_size,
                                           eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                           dtype=dtype,
                                           device=device)
        self.fusion_norm_alpha = nn.Parameter(
            torch.tensor(getattr(config, 'concept_fusion_norm_alpha_init', 0.1), dtype=dtype, device=device),
            requires_grad=False)
        self.dd_encoder_self_dd = SelfDD(config,
                                         config.concept_encoder_layers,
                                         use_softmax=False,
                                         dtype=dtype,
                                         device=device)
        self.decoder_read_encoder_routes = nn.ModuleList([
            ResidualRoute(config,
                          config.concept_encoder_layers,
                          use_softmax=True,
                          dtype=dtype,
                          device=device)
            for _ in range(config.concept_decoder_layers)
        ])
        self.decoder_read_encoder_shared_source_norm = nn.LayerNorm(config.hidden_size,
                                                                    eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                                                    dtype=dtype,
                                                                    device=device)
        self.decoder_read_concept_routes = nn.ModuleList([
            ResidualRoute(config,
                          config.concept_special_layers,
                          use_softmax=True,
                          dtype=dtype,
                          device=device)
            for _ in range(config.concept_decoder_layers)
        ])
        self.decoder_read_concept_shared_source_norm = nn.LayerNorm(config.hidden_size,
                                                                    eps=getattr(config, 'layernorm_epsilon', 1e-6),
                                                                    dtype=dtype,
                                                                    device=device)
        self.final_read_concept_gate_logits = nn.Parameter(
            torch.zeros(config.concept_decoder_layers, 2, dtype=dtype, device=device),
            requires_grad=False)
        self.dd_two_route_add = TwoRouteAdd(config, dtype=dtype, device=device)
        self.concept_ops = ConceptLMRuntimeOps(config)
        # output projection — mirrors ``output_layer`` in the reference. Built
        # via build_lm_head and named ``lm_head`` so DeployModelMixinV1.
        # get_logits picks it up directly; load_weights maps the checkpoint's
        # ``output_layer`` onto it.
        self.lm_head = self.build_lm_head(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device)

    def forward(self,
                input_ids: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]],
                attn_metadata=None,
                inputs_embeds: torch.Tensor = None,
                state_ids: torch.Tensor | None = None,
                state_caches: list[torch.Tensor] | None = None,
                named_state_caches: Mapping[str, torch.Tensor] | None = None,
                **kwargs):
        """Model forward, return hidden_states (logits computed by runtime)."""
        concept_metadata = self._build_concept_metadata(position_ids, attn_metadata, state_ids)
        concept_caches = self._build_concept_caches(past_key_values, state_caches, named_state_caches)
        if inputs_embeds is None:
            hidden_states = self.embedding(input_ids)
        else:
            hidden_states = inputs_embeds

        if concept_metadata.is_decoding:
            return self._forward_decode(
                hidden_states,
                position_ids,
                concept_metadata,
                concept_caches,
            )

        hidden_states, prefill_position_ids = self._normalize_prefill_inputs(
            hidden_states,
            position_ids,
            attn_metadata,
        )
        return self._forward_prefill_packed(
            hidden_states,
            prefill_position_ids,
            concept_metadata,
            concept_caches,
        )

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embedding.word_embeddings

    def get_output_embeddings(self):
        """Get output embeddings."""
        return self.lm_head

    def _build_concept_metadata(self,
                                position_ids: torch.Tensor | None,
                                attn_metadata: Any = None,
                                state_ids: torch.Tensor | None = None) -> ConceptMetadata:
        """Build top-level ConceptLM metadata for future serving paths."""
        return ConceptMetadata.build(
            self.config,
            position_ids=position_ids,
            attn_metadata=attn_metadata,
            state_ids=state_ids,
        )

    def _build_concept_caches(self,
                              past_key_values: list[list[torch.Tensor]] | None,
                              state_caches: list[torch.Tensor] | None = None,
                              named_state_caches: Mapping[str, torch.Tensor] | None = None) -> ConceptCaches:
        """Build top-level ConceptLM cache views for future serving paths."""
        return ConceptCaches.build(
            self.config,
            past_key_values=past_key_values,
            state_caches=state_caches,
            named_state_caches=named_state_caches,
        )

    def _decode_chunk_state_update(self,
                                   current_source_states: torch.Tensor,
                                   concept_metadata: ConceptMetadata,
                                   concept_caches: ConceptCaches) -> ConceptChunkStateUpdateResult:
        """Update decode chunk-source state and return fixed-shape rows.

        CUDA uses the Triton writer. CPU uses the reference writer for tests. The returned rows deliberately avoid
        dynamic concept-row compaction, matching the CUDA graph route in the design doc.
        """
        chunk_source_state = concept_caches.chunk_source_state
        concept_input_states, update_mask = self.concept_ops.decode_chunk_state_update(
            chunk_source_state,
            current_source_states,
            concept_metadata.state_ids,
            concept_metadata.position_ids,
            concept_metadata.chunk_size,
            concept_metadata.merge_method,
        )
        return ConceptChunkStateUpdateResult(
            concept_input_states=concept_input_states,
            concept_update_mask=update_mask,
        )

    def _route_gate(self, layer_idx: int) -> torch.Tensor:
        """Return decoder route gate ``[decoder_dd_scale,
        concept_route_scale]``."""
        return self.final_read_concept_gate_logits[int(layer_idx)].float().softmax(dim=-1)

    def _normalize_prefill_inputs(self,
                                  hidden_states: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  attn_metadata: Any = None):
        """Normalize LMDeploy prefill inputs to packed token layout.

        Engine layout stays fixed as ``[1, total_tokens, hidden]`` with
        ``position_ids=[1, total_tokens]``. Per-request boundaries come from
        ``attn_metadata.q_seqlens``/``q_start_loc`` and are used only to build
        the concept stream.
        """
        if attn_metadata is None or getattr(attn_metadata, 'q_seqlens', None) is None:
            raise RuntimeError('ConceptLM prefill requires q_seqlens attention metadata.')
        if getattr(attn_metadata, 'is_decoding', False):
            raise RuntimeError('ConceptLM prefill input normalization received decode metadata.')

        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            raise NotImplementedError(
                f'ConceptLM prefill expects fixed engine layout [1, total_tokens, hidden], '
                f'got {tuple(hidden_states.shape)}.')

        total_tokens = hidden_states.size(1)
        if position_ids.dim() != 2 or position_ids.size(0) != 1:
            raise NotImplementedError(
                f'ConceptLM prefill expects fixed position layout [1, total_tokens], '
                f'got {tuple(position_ids.shape)}.')
        if position_ids.size(1) != total_tokens:
            raise ValueError(f'position_ids length {position_ids.size(1)} does not match token length {total_tokens}.')

        q_seqlens = attn_metadata.q_seqlens
        if q_seqlens.dim() != 1:
            raise ValueError(f'ConceptLM prefill expects 1-D q_seqlens, got {tuple(q_seqlens.shape)}.')

        hidden_states = hidden_states[0].contiguous()
        return hidden_states, position_ids[0].to(device=hidden_states.device)

    def _normalize_decode_inputs(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        """Normalize LMDeploy decode inputs to one flat row per active
        request."""
        if hidden_states.dim() != 3 or hidden_states.size(0) != 1:
            raise NotImplementedError(
                f'ConceptLM decode expects fixed engine layout [1, batch, hidden], '
                f'got {tuple(hidden_states.shape)}.')
        batch_size = hidden_states.size(1)
        position_ids = self.concept_ops.flatten_decode_position_ids(position_ids, batch_size, hidden_states.device)
        if position_ids.numel() != batch_size:
            raise ValueError(f'Expected {batch_size} decode position ids, got {position_ids.numel()}.')
        return hidden_states[0].contiguous(), position_ids

    def _select_decode_last_state_rows(self, concept_caches: ConceptCaches,
                                       decode_metadata: ConceptDecodeMetadata) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather latest concept state rows through backend-owned layout."""
        return self.concept_ops.select_decode_last_state_rows(
            concept_caches.last_state,
            concept_caches.last_final_state,
            concept_caches.last_raw_states,
            decode_metadata,
        )

    def _select_decode_decoder_concepts(self,
                                        concept_caches: ConceptCaches,
                                        decode_metadata: ConceptDecodeMetadata,
                                        previous_final_state: torch.Tensor | None,
                                        previous_raw_states: torch.Tensor | None) -> _DecoderConceptInput:
        """Select the final/raw concept state visible to this decode token."""
        if bool(getattr(self.config, 'concept_shift_feature', True)):
            final_state, raw_states = self._select_decode_last_state_rows(concept_caches, decode_metadata)
        else:
            final_state = previous_final_state
            raw_states = previous_raw_states

        concept_read_mask = self.concept_ops.decode_concept_read_mask(decode_metadata)
        final_state = torch.where(concept_read_mask.view(-1, 1), final_state, torch.zeros_like(final_state))
        raw_states = torch.where(concept_read_mask.view(-1, 1, 1), raw_states, torch.zeros_like(raw_states))
        return _DecoderConceptInput.for_decode(final_state, raw_states)

    @staticmethod
    def _build_decode_chunk_source_states(encoder_output: _EncoderOutput) -> torch.Tensor:
        """Return current per-row states accumulated until the next concept
        boundary as a view over encoder history.

        Row 0 is the final encoder hidden, used as concept-predictor input when
        a boundary is reached. Remaining rows mirror the prefill
        ``encoder_raw_states[:-1]`` route sources.
        """
        num_sources = max(len(encoder_output.raw_states), 1)
        return encoder_output.history_buffer[:num_sources].movedim(0, -2)

    @staticmethod
    def _decode_concept_position_ids(position_ids: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Return reference RoPE positions for concept rows emitted at decode
        boundaries."""
        return (position_ids - int(chunk_size) + 1).clamp(min=0)

    def _snapshot_decode_concept_kv(self,
                                    concept_caches: ConceptCaches,
                                    concept_attn_metadata: Any):
        """Snapshot concept KV slots that dummy non-boundary rows may
        overwrite."""
        return self.concept_ops.snapshot_decode_concept_kv(
            concept_caches.concept_past_key_values,
            concept_attn_metadata,
        )

    def _restore_decode_concept_kv_(
        self,
        concept_caches: ConceptCaches,
        concept_attn_metadata: Any,
        saved_kv,
        restore_mask: torch.Tensor,
    ):
        """Restore concept KV slots for non-boundary and padded rows."""
        self.concept_ops.restore_decode_concept_kv(
            concept_caches.concept_past_key_values,
            concept_attn_metadata,
            saved_kv,
            restore_mask,
        )

    def _write_decode_concept_states_static_(self,
                                             concept_caches: ConceptCaches,
                                             decode_metadata: ConceptDecodeMetadata,
                                             update_mask: torch.Tensor,
                                             predicted_vectors: torch.Tensor,
                                             concept_raw_states: list[torch.Tensor]):
        """Write newly emitted concept states to persistent decode caches."""
        self.concept_ops.write_decode_concept_states(
            concept_caches.last_raw_states,
            concept_caches.last_final_state,
            predicted_vectors,
            concept_raw_states,
            decode_metadata.state_ids,
            update_mask,
        )

    def _build_decode_concept_request(self,
                                      chunk_update: ConceptChunkStateUpdateResult,
                                      decode_metadata: ConceptDecodeMetadata,
                                      concept_metadata: ConceptMetadata) -> _ConceptPredictorRequest:
        """Build fixed-shape concept predictor inputs for decode."""
        concept_hidden = self.concept_vq_input_norm(chunk_update.concept_input_states[:, 0])
        encoder_concept_states = self.concept_predictor.normalize_encoder_concept_states(
            chunk_update.concept_input_states[:, 1:])
        concept_position_ids = self._decode_concept_position_ids(
            decode_metadata.position_ids,
            concept_metadata.chunk_size,
        )
        concept_attn_metadata = self.concept_ops.build_concept_decode_metadata_static(
            concept_metadata.attn_metadata,
            decode_metadata,
        )
        return _ConceptPredictorRequest(
            hidden_states=concept_hidden,
            encoder_states=encoder_concept_states,
            position_ids=concept_position_ids,
            attn_metadata=concept_attn_metadata,
        )

    def _update_decode_concept_states_static_(self,
                                              chunk_update: ConceptChunkStateUpdateResult,
                                              decode_metadata: ConceptDecodeMetadata,
                                              concept_metadata: ConceptMetadata,
                                              concept_caches: ConceptCaches):
        """Emit/cache concept states with fixed batch shape.

        The predictor runs for every decode row so CUDA graph capture sees a stable launch sequence. Non-boundary rows
        are dummy work: their concept KV writes are restored and their final/raw state writes are masked.
        """
        request = self._build_decode_concept_request(chunk_update, decode_metadata, concept_metadata)
        saved_kv = self._snapshot_decode_concept_kv(concept_caches, request.attn_metadata)
        concept_output = self._run_concept_predictor(request, concept_caches)
        self._restore_decode_concept_kv_(
            concept_caches,
            request.attn_metadata,
            saved_kv,
            ~chunk_update.concept_update_mask,
        )
        self._write_decode_concept_states_static_(
            concept_caches,
            decode_metadata,
            chunk_update.concept_update_mask,
            concept_output.predicted_vectors,
            concept_output.raw_states,
        )

    def _build_encoder_concept_states_packed(self,
                                             encoder_raw_states: list[torch.Tensor],
                                             prefill_metadata: ConceptPrefillMetadata) -> torch.Tensor:
        """Build packed chunk-level encoder states used by the concept
        predictor."""
        chunks = [self.concept_ops.merge_chunks_packed(state, prefill_metadata) for state in encoder_raw_states[:-1]]
        states = torch.stack(chunks, dim=-2)
        return self.concept_predictor.normalize_encoder_concept_states(states)

    def _run_concept_predictor(self,
                               request: _ConceptPredictorRequest,
                               concept_caches: ConceptCaches) -> _ConceptPredictorOutput:
        """Run concept predictor and quantizer for any concept stream
        layout."""
        concept_logits, concept_raw_states = self.concept_predictor(
            request.hidden_states,
            request.encoder_states,
            request.position_ids,
            past_key_values=concept_caches.concept_past_key_values,
            attn_metadata=request.attn_metadata,
        )
        return _ConceptPredictorOutput(
            predicted_vectors=self.concept_quantizer(concept_logits),
            raw_states=concept_raw_states,
        )

    def _build_prefill_concept_request(self,
                                       hidden_states: torch.Tensor,
                                       encoder_raw_states: list[torch.Tensor],
                                       prefill_metadata: ConceptPrefillMetadata,
                                       concept_metadata: ConceptMetadata) -> _ConceptPredictorRequest:
        """Build concept predictor inputs for compact packed prefill."""
        concept_hidden = self.concept_ops.merge_chunks_packed(hidden_states, prefill_metadata)
        concept_hidden = self.concept_vq_input_norm(concept_hidden)
        encoder_concept_states = self._build_encoder_concept_states_packed(encoder_raw_states, prefill_metadata)
        concept_attn_metadata = self.concept_ops.build_concept_prefill_metadata(
            concept_metadata.attn_metadata,
            prefill_metadata,
        )
        return _ConceptPredictorRequest(
            hidden_states=concept_hidden,
            encoder_states=encoder_concept_states,
            position_ids=prefill_metadata.concept_position_ids,
            attn_metadata=concept_attn_metadata,
        )

    def _fuse_token_concept_states(self,
                                   hidden_states: torch.Tensor,
                                   final_concept_state: torch.Tensor) -> torch.Tensor:
        """Fuse token and final-concept states before the decoder stack."""
        return self.fusion_tok_norm(hidden_states) + self.fusion_norm_alpha.to(
            hidden_states.dtype) * self.fusion_hl_norm(final_concept_state.to(hidden_states.dtype))

    def _run_decoder_from_concepts(self,
                                   hidden_states: torch.Tensor,
                                   encoder_raw_states: list[torch.Tensor],
                                   decoder_concepts: _DecoderConceptInput,
                                   position_ids: torch.Tensor,
                                   concept_caches: ConceptCaches,
                                   attn_metadata: Any) -> torch.Tensor:
        """Fuse visible concept state and run the decoder stack."""
        decoder_input = self._fuse_token_concept_states(hidden_states, decoder_concepts.final_state)
        return self._decode(
            decoder_input,
            encoder_raw_states,
            decoder_concepts,
            position_ids,
            concept_caches,
            attn_metadata,
        )

    def _prepare_decoder_route_sources(self,
                                       encoder_raw_states: list[torch.Tensor],
                                       decoder_concepts: _DecoderConceptInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare encoder/concept route sources in packed ``[..., L, H]``
        layout."""
        decoder_encoder_states = torch.stack(tuple(encoder_raw_states), dim=-2)
        decoder_encoder_states = self.decoder_read_encoder_shared_source_norm(decoder_encoder_states)

        if decoder_concepts.decode_states is not None:
            concept_states = self.decoder_read_concept_shared_source_norm(decoder_concepts.decode_states)
            return decoder_encoder_states, concept_states

        concept_states = torch.stack(tuple(decoder_concepts.raw_states), dim=-2)
        zero_chunk = torch.zeros_like(concept_states[:1])
        concept_states = torch.cat((zero_chunk, concept_states), dim=0)
        concept_states = self.decoder_read_concept_shared_source_norm(concept_states)
        concept_states = self.concept_ops.repeat_shift_source_states_packed(concept_states,
                                                                            decoder_concepts.prefill_metadata)
        return decoder_encoder_states, concept_states

    def _forward_prefill_packed(self,
                                hidden_states: torch.Tensor,
                                position_ids: torch.Tensor,
                                concept_metadata: ConceptMetadata,
                                concept_caches: ConceptCaches):
        """Packed non-decode ConceptLM forward."""
        if (concept_caches.encoder_past_key_values is None or concept_caches.concept_past_key_values is None
                or concept_caches.decoder_past_key_values is None):
            raise RuntimeError('ConceptLM prefill requires encoder, concept, and decoder KV caches.')
        prefill_metadata = self.concept_ops.build_prefill_metadata(concept_metadata.attn_metadata, position_ids)
        encoder_output = self._encode(
            hidden_states,
            position_ids,
            past_key_values=concept_caches.encoder_past_key_values,
            attn_metadata=concept_metadata.attn_metadata,
        )
        hidden_states = encoder_output.hidden_states
        encoder_raw_states = encoder_output.raw_states
        concept_request = self._build_prefill_concept_request(
            hidden_states,
            encoder_raw_states,
            prefill_metadata,
            concept_metadata,
        )
        concept_output = self._run_concept_predictor(concept_request, concept_caches)
        self.concept_ops.write_prefill_state_caches_eager(
            concept_caches.chunk_source_state,
            concept_caches.last_raw_states,
            concept_caches.last_final_state,
            concept_metadata.state_ids,
            prefill_metadata,
            self._build_decode_chunk_source_states(encoder_output),
            concept_output.predicted_vectors,
            concept_output.raw_states,
        )
        repeated_concepts = self.concept_ops.repeat_shift_packed(concept_output.predicted_vectors, prefill_metadata)
        decoder_concepts = _DecoderConceptInput.for_prefill(
            repeated_concepts,
            concept_output.raw_states,
            prefill_metadata,
        )
        final_hidden = self._run_decoder_from_concepts(
            hidden_states,
            encoder_raw_states,
            decoder_concepts,
            position_ids,
            concept_caches,
            concept_metadata.attn_metadata,
        )
        return final_hidden.unsqueeze(0).contiguous()

    def _forward_decode(self,
                        hidden_states: torch.Tensor,
                        position_ids: torch.Tensor,
                        concept_metadata: ConceptMetadata,
                        concept_caches: ConceptCaches):
        """ConceptLM decode path.

        This path is semantically structured for serving. Boundary concept
        updates run with fixed batch shape so it is eligible for CUDA graph
        replay through the base ``CudaGraphMixin`` decode-only policy.
        """
        if (concept_caches.encoder_past_key_values is None or concept_caches.concept_past_key_values is None
                or concept_caches.decoder_past_key_values is None):
            raise RuntimeError('ConceptLM decode requires encoder, concept, and decoder KV caches.')
        if concept_caches.chunk_source_state is None:
            raise RuntimeError('ConceptLM decode requires chunk source state cache.')
        if concept_caches.last_final_state is None or concept_caches.last_raw_states is None:
            raise RuntimeError('ConceptLM decode requires cached last concept states.')

        hidden_states, decode_position_ids = self._normalize_decode_inputs(hidden_states, position_ids)
        decode_metadata = self.concept_ops.build_decode_metadata(
            decode_position_ids,
            concept_metadata.state_ids,
            hidden_states.size(0),
            hidden_states.device,
        )
        decode_concept_metadata = replace(
            concept_metadata,
            position_ids=decode_position_ids,
            state_ids=decode_metadata.state_ids,
        )

        encoder_output = self._encode(
            hidden_states,
            decode_position_ids,
            past_key_values=concept_caches.encoder_past_key_values,
            attn_metadata=concept_metadata.attn_metadata,
        )
        hidden_states = encoder_output.hidden_states
        encoder_raw_states = encoder_output.raw_states
        previous_final_concept_state = None
        previous_concept_raw_state_rows = None
        if not bool(getattr(self.config, 'concept_shift_feature', True)):
            previous_final_concept_state, previous_concept_raw_state_rows = self._select_decode_last_state_rows(
                concept_caches, decode_metadata)
        current_source_states = self._build_decode_chunk_source_states(encoder_output)
        chunk_update = self._decode_chunk_state_update(
            current_source_states,
            decode_concept_metadata,
            concept_caches,
        )
        self._update_decode_concept_states_static_(
            chunk_update,
            decode_metadata,
            decode_concept_metadata,
            concept_caches,
        )

        decoder_concepts = self._select_decode_decoder_concepts(
            concept_caches,
            decode_metadata,
            previous_final_concept_state,
            previous_concept_raw_state_rows,
        )
        final_hidden = self._run_decoder_from_concepts(
            hidden_states,
            encoder_raw_states,
            decoder_concepts,
            decode_position_ids,
            concept_caches,
            concept_metadata.attn_metadata,
        )
        return final_hidden.unsqueeze(0).contiguous()

    def _encode(self,
                hidden_states: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: list[list[torch.Tensor]] | None = None,
                attn_metadata: Any = None):
        """Encoder stack plus encoder self-DD.

        This helper mirrors the reference flow but consumes LMDeploy attention inputs. It is wired for the future full
        forward path; continuous batching still needs caller-side concept metadata before the full model can use it
        safely.
        """
        raw_states = []
        history_buffer = self.dd_encoder_self_dd.make_history_buffer(hidden_states)
        self.dd_encoder_self_dd.write_history(history_buffer, 0, hidden_states)
        rotary_pos_emb = self.encoder._make_rotary_pos_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.encoder.layers):
            pkv = past_key_values[layer_idx] if past_key_values is not None else None
            raw = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=pkv,
                attn_metadata=attn_metadata,
            )
            raw_states.append(raw)
            self.dd_encoder_self_dd.write_history(history_buffer, layer_idx + 1, raw)
            hidden_states = self.dd_encoder_self_dd.forward_from_buffer(layer_idx, raw, history_buffer)

        # The encoder SelfDD path no longer needs the original input stored in
        # slot 0 after the loop. Reuse that slot for the final encoder hidden
        # so decode/prefill state seeding can view ``[final, raw[:-1]]``
        # without stacking and copying every raw layer output again.
        self.dd_encoder_self_dd.write_history(history_buffer, 0, hidden_states)
        return _EncoderOutput(
            hidden_states=hidden_states,
            raw_states=raw_states,
            history_buffer=history_buffer,
        )

    def _decode(self,
                decoder_input: torch.Tensor,
                encoder_raw_states: list[torch.Tensor],
                decoder_concepts: _DecoderConceptInput,
                position_ids: torch.Tensor,
                concept_caches: ConceptCaches,
                attn_metadata: Any = None):
        """Decoder stack plus decoder DD and residual routes."""
        past_key_values = concept_caches.decoder_past_key_values
        decoder_encoder_states, concept_states = self._prepare_decoder_route_sources(
            encoder_raw_states,
            decoder_concepts,
        )
        final_concept_state = decoder_concepts.final_state
        hidden_states = decoder_input
        history_buffer = self.dd_two_route_add.make_history_buffer(hidden_states)
        self.dd_two_route_add.write_history(history_buffer, 0, hidden_states)
        rotary_pos_emb = self.decoder._make_rotary_pos_emb(hidden_states, position_ids)

        for layer_idx, layer in enumerate(self.decoder.layers):
            pkv = past_key_values[layer_idx] if past_key_values is not None else None
            raw = layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=pkv,
                attn_metadata=attn_metadata,
            )
            self.dd_two_route_add.write_history(history_buffer, layer_idx + 1, raw)
            gate = self._route_gate(layer_idx)
            hidden_states = self.dd_two_route_add.forward_from_buffer(
                layer_idx,
                raw,
                history_buffer,
                final_concept_state,
                gate[0],
            )
            hidden_states = self.decoder_read_encoder_routes[layer_idx](
                hidden_states,
                decoder_encoder_states,
                source_dim=-2,
            )
            hidden_states = self.decoder_read_concept_routes[layer_idx](
                hidden_states,
                concept_states,
                residual_scale=gate[1],
                source_dim=-2,
            )

        if self.decoder.final_layernorm is not None:
            hidden_states = self.decoder.final_layernorm(hidden_states)
        return hidden_states

    def prepare_inputs_for_generation(self,
                                      past_key_values: list[list[torch.Tensor]],
                                      inputs_embeds: torch.Tensor | None = None,
                                      context: StepContext = None):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            state_caches=context.state_caches,
            named_state_caches=context.named_state_caches,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load native ConceptLM checkpoint weights into implemented
        modules."""
        # (checkpoint_name, target_name)
        weight_map = {
            'embedding.word_embeddings.weight': 'embedding.word_embeddings.weight',
            'output_layer.weight': 'lm_head.weight',
        }
        codebook_prefix = 'concept_quantizer.codebook.'
        prediction_head_prefix = 'concept_predictor.prediction_heads.'
        block_prefixes = (
            ('encoder.', self.encoder),
            ('decoder.', self.decoder),
            ('concept_predictor.hlm_block.', self.concept_predictor.hlm_block),
        )
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            loaded_by_block = False
            for block_prefix, block in block_prefixes:
                if name.startswith(block_prefix):
                    block.load_weights([(name, loaded_weight)], prefix=block_prefix[:-1])
                    loaded_by_block = True
                    break
            if loaded_by_block:
                continue
            if name.startswith(prediction_head_prefix):
                suffix = name[len(prediction_head_prefix):]
                parts = suffix.split('.')
                if len(parts) == 2 and parts[0].isdigit() and parts[1] in ('weight', 'bias'):
                    target_name = f'{prediction_head_prefix}proj.{parts[1]}'
                    param = params_dict.get(target_name)
                    if param is not None:
                        load_weight(param, loaded_weight, shard_id=int(parts[0]))
                    continue
                if suffix in ('proj.weight', 'proj.bias'):
                    param = params_dict.get(name)
                    if param is not None:
                        for shard_id, shard_weight in enumerate(param.weight_spliter(loaded_weight)):
                            load_weight(param, shard_weight, shard_id=shard_id)
                    continue
            if name.startswith(codebook_prefix):
                codebook_idx = int(name[len(codebook_prefix):])
                param = params_dict['concept_quantizer.codebook']
                _load_stacked_codebook_weight(param, loaded_weight, codebook_idx)
                continue
            target = weight_map.get(name)
            if target is None:
                target = name
            if target not in params_dict:
                # skip checkpoint metadata or tensors owned by future runtime-only paths
                continue
            param = params_dict[target]
            load_weight(param, loaded_weight)
