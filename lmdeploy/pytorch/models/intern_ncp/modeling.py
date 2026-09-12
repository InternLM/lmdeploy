# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.backends.conceptlm import (
    ConceptChunkInput,
    ConceptDecoderInput,
    ConceptRuntimeCaches,
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

        hidden_states, position_ids = self._normalize_forward_inputs(
            hidden_states,
            position_ids,
            concept_metadata,
        )
        self._validate_concept_caches(concept_metadata, concept_caches)
        return self._forward_token_stream(
            hidden_states,
            position_ids,
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

    @staticmethod
    def _build_runtime_caches(concept_caches: ConceptCaches) -> ConceptRuntimeCaches:
        """Expose only backend-owned cache views to ConceptLM runtime ops."""
        return ConceptRuntimeCaches(
            chunk_source_state=concept_caches.chunk_source_state,
            last_state=concept_caches.last_state,
            last_raw_states=concept_caches.last_raw_states,
            last_final_state=concept_caches.last_final_state,
            concept_past_key_values=concept_caches.concept_past_key_values,
        )

    def _route_gate(self, layer_idx: int) -> torch.Tensor:
        """Return decoder route gate ``[decoder_dd_scale,
        concept_route_scale]``."""
        return self.final_read_concept_gate_logits[int(layer_idx)].float().softmax(dim=-1)

    def _normalize_forward_inputs(self,
                                  hidden_states: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  concept_metadata: ConceptMetadata):
        """Normalize token-stream inputs to flat ``[tokens_or_batch, hidden]``.

        Prefill and decode use different engine layouts, but the model body below consumes one flat token stream for
        both phases.
        """
        if concept_metadata.is_decoding:
            return self._normalize_decode_inputs(hidden_states, position_ids)
        return self._normalize_prefill_inputs(
            hidden_states,
            position_ids,
            concept_metadata.attn_metadata,
        )

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

    @staticmethod
    def _validate_concept_caches(concept_metadata: ConceptMetadata, concept_caches: ConceptCaches) -> None:
        """Validate cache streams needed by the shared forward body."""
        phase = 'decode' if concept_metadata.is_decoding else 'prefill'
        if (concept_caches.encoder_past_key_values is None or concept_caches.concept_past_key_values is None
                or concept_caches.decoder_past_key_values is None):
            raise RuntimeError(f'ConceptLM {phase} requires encoder, concept, and decoder KV caches.')
        if not concept_metadata.is_decoding:
            return
        if concept_caches.chunk_source_state is None:
            raise RuntimeError('ConceptLM decode requires chunk source state cache.')
        if concept_caches.last_final_state is None or concept_caches.last_raw_states is None:
            raise RuntimeError('ConceptLM decode requires cached last concept states.')

    @staticmethod
    def _build_chunk_source_states(encoder_output: _EncoderOutput) -> torch.Tensor:
        """Return source states consumed by concept chunk preparation.

        Row 0 is the final encoder hidden, used as concept-predictor input when
        a chunk concept is produced. Remaining rows mirror
        ``encoder_raw_states[:-1]`` route sources. The returned layout is shared
        by prefill and decode: ``[token_or_batch, num_sources, hidden]``.
        """
        num_sources = max(len(encoder_output.raw_states), 1)
        return encoder_output.history_buffer[:num_sources].movedim(0, -2)

    def _build_concept_request(self, chunk_input: ConceptChunkInput) -> _ConceptPredictorRequest:
        """Build concept predictor inputs from backend-prepared chunk rows."""
        concept_hidden = self.concept_vq_input_norm(chunk_input.source_states[:, 0])
        encoder_sources = chunk_input.source_states[:, 1:]
        encoder_concept_states = self.concept_predictor.normalize_encoder_concept_states(encoder_sources)
        return _ConceptPredictorRequest(
            hidden_states=concept_hidden,
            encoder_states=encoder_concept_states,
            position_ids=chunk_input.position_ids,
            attn_metadata=chunk_input.attn_metadata,
        )

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

    def _fuse_token_concept_states(self,
                                   hidden_states: torch.Tensor,
                                   final_concept_state: torch.Tensor) -> torch.Tensor:
        """Fuse token and final-concept states before the decoder stack."""
        return self.fusion_tok_norm(hidden_states) + self.fusion_norm_alpha.to(
            hidden_states.dtype) * self.fusion_hl_norm(final_concept_state.to(hidden_states.dtype))

    def _run_decoder_from_concepts(self,
                                   hidden_states: torch.Tensor,
                                   encoder_raw_states: list[torch.Tensor],
                                   decoder_concepts: ConceptDecoderInput,
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
                                       decoder_concepts: ConceptDecoderInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare encoder/concept route sources in packed ``[..., L, H]``
        layout."""
        decoder_encoder_states = torch.stack(tuple(encoder_raw_states), dim=-2)
        decoder_encoder_states = self.decoder_read_encoder_shared_source_norm(decoder_encoder_states)

        concept_states = self.decoder_read_concept_shared_source_norm(decoder_concepts.route_states)
        return decoder_encoder_states, concept_states

    def _forward_token_stream(self,
                              hidden_states: torch.Tensor,
                              position_ids: torch.Tensor,
                              concept_metadata: ConceptMetadata,
                              concept_caches: ConceptCaches):
        """Shared ConceptLM forward for normalized prefill and decode rows."""
        encoder_output = self._encode(
            hidden_states,
            position_ids,
            past_key_values=concept_caches.encoder_past_key_values,
            attn_metadata=concept_metadata.attn_metadata,
        )
        hidden_states = encoder_output.hidden_states
        encoder_raw_states = encoder_output.raw_states
        source_states = self._build_chunk_source_states(encoder_output)
        runtime_caches = self._build_runtime_caches(concept_caches)
        chunk_input = self.concept_ops.build_concept_chunk_input(
            source_states,
            concept_metadata.attn_metadata,
            position_ids,
            state_ids=concept_metadata.state_ids,
            chunk_source_state_cache=runtime_caches.chunk_source_state,
        )
        forward_context = self.concept_ops.begin_concept_forward(chunk_input, runtime_caches)
        concept_request = self._build_concept_request(chunk_input)
        concept_output = self._run_concept_predictor(concept_request, concept_caches)
        self.concept_ops.end_concept_forward(
            chunk_input,
            runtime_caches,
            forward_context,
            source_states,
            concept_output.predicted_vectors,
            concept_output.raw_states,
        )
        decoder_concepts = self.concept_ops.build_decoder_concept_input(
            chunk_input,
            runtime_caches,
            forward_context,
            concept_output.predicted_vectors,
            concept_output.raw_states,
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
                decoder_concepts: ConceptDecoderInput,
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
