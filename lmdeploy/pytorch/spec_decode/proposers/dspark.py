# Copyright (c) OpenMMLab. All rights reserved.
"""Fixed-window DSpark block proposer."""

import torch

from lmdeploy.utils import get_logger

from ...config import CacheConfig, ModelConfig
from ...engine.cache_engine import CacheEngine
from ...model_inputs import ModelInputs
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..block_parallel import prepare_query
from .base import (
    SPEC_PROPOSERS,
    BaseSpecProposer,
    ProposalWarmupCase,
    ProposalWarmupPlan,
)
from .dflash import DFlash

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='dspark')
class DSpark(DFlash):
    """DFlash-family proposal with sequential DSpark logit correction."""

    # Inherit DFlash.propose so deterministic DSpark tokens use the common
    # one-hot rejection path for both greedy and non-greedy target sampling.

    # Opt in only after warmup has checked the allocated draft cache geometry.
    _full_context_materialization = False

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None,
                    build_model_ctx=None):
        self._full_context_materialization = False
        if target_model is None:
            raise RuntimeError('DSpark requires the target model for shared modules.')
        BaseSpecProposer.build_model(self,
                                     empty_init,
                                     target_model=target_model,
                                     build_model_ctx=build_model_ctx)
        if not hasattr(self.model, 'set_input_embeddings'):
            raise RuntimeError('DSpark draft model must implement set_input_embeddings().')
        self.model.set_input_embeddings(target_model.get_input_embeddings())
        output_head = getattr(target_model, 'head', None)
        if output_head is None:
            output_head = getattr(target_model, 'lm_head', None)
        if output_head is None or not hasattr(self.model, 'set_output_head'):
            raise RuntimeError('DSpark draft/target models do not expose a shareable output head.')
        self.model.set_output_head(output_head)
        logger.info('Using target embeddings and output head for DSpark draft.')

    @property
    def draft_query_len(self) -> int:
        query_len = self.specdecode_config.dspark_draft_query_len
        if query_len is None:
            raise RuntimeError('DSpark draft query length was not resolved during config parsing.')
        return int(query_len)

    def get_warmup_plan(self,
                        max_batches: int,
                        target_model_config: ModelConfig,
                        capture_batch_sizes: list[int],
                        cache_config: CacheConfig) -> ProposalWarmupPlan:
        if cache_config is None:
            raise RuntimeError('DSpark warmup requires a draft cache configuration.')
        target_hidden_size = self.get_target_hidden_size(target_model_config)
        cache_block_size = max(1, int(cache_config.block_size))
        prefill_budget = max(1, int(cache_config.max_prefill_token_num))
        prefill_q_len = min(cache_block_size, prefill_budget)
        prefill_batch = min(max_batches, max(1, prefill_budget // prefill_q_len))
        cases = [ProposalWarmupCase(prefill_batch,
                                    is_decoding=False,
                                    max_q_seqlen=prefill_q_len,
                                    target_hidden_size=target_hidden_size)]
        cases.extend(ProposalWarmupCase(batch_size,
                                        is_decoding=True,
                                        max_q_seqlen=self.draft_query_len,
                                        target_hidden_size=target_hidden_size)
                     for batch_size in capture_batch_sizes)
        return ProposalWarmupPlan(tuple(cases))

    def _configure_context_materialization(self, cache_engine: CacheEngine):
        """Select the fixed-layout path using host metadata, never GPU
        values."""
        model = self._draft_model()
        enabled = bool(getattr(model, 'supports_full_context_materialization', False))
        state_engine = getattr(cache_engine, 'state_cache_engine', None)
        named_caches = {} if state_engine is None else state_engine.named_state_caches
        ring_capacity = getattr(getattr(model, 'args', None), 'ring_storage_capacity', None)
        if ring_capacity is not None:
            ring = named_caches.get('v4_window_kv_fp8')
            if ring is not None and ring.size(2) != ring_capacity:
                raise RuntimeError('DSpark draft ring capacity does not match the allocated cache: '
                                   f'model={ring_capacity}, cache={ring.size(2)}.')
            enabled = enabled and set(named_caches) == {'v4_window_kv_fp8'}
        elif named_caches:
            # A new stateful draft needs its own full-block safety proof.
            enabled = False
        dist_config = getattr(self.specdecode_config, 'dist_config', None)
        if dist_config is not None and (dist_config.dp > 1 or dist_config.ep > 1) and not enabled:
            # Reject before the FIRST warmup materialization, not only when
            # decode later attempts the accepted-prefix compaction fallback.
            raise ValueError('DSpark DP/EP requires a validated full-context cache geometry.')
        self._full_context_materialization = enabled

    def prepare_warmup_forward(self, inputs: ModelInputs, cache_engine: CacheEngine) -> ModelInputs | None:
        self._configure_context_materialization(cache_engine)
        return super().prepare_warmup_forward(inputs, cache_engine)

    def _build_query_inputs(self,
                            model_inputs: ModelInputs,
                            context_lengths: torch.Tensor,
                            next_token_ids: torch.Tensor,
                            *,
                            query_start_positions: torch.Tensor | None = None):
        batch_size = model_inputs.seq_length.numel()
        query_len = self.draft_query_len
        query_ids = model_inputs.input_ids.new_full(
            (batch_size, query_len), int(self.specdecode_config.mask_token_id))
        query_ids[:, 0] = next_token_ids
        query_history = model_inputs.history_lengths + context_lengths
        if query_start_positions is None:
            query_start_positions = query_history
        query_positions = query_start_positions[:, None] + torch.arange(
            query_len, device=query_history.device)[None, :]
        return model_inputs.clone(
            input_ids=query_ids.reshape(1, -1),
            seq_length=model_inputs.seq_length.new_full((batch_size,), query_len),
            history_lengths=query_history,
            max_q_seqlen=query_len,
            max_kv_seqlen=model_inputs.max_kv_seqlen + query_len,
            sum_kv_seqlen=model_inputs.sum_kv_seqlen + batch_size * query_len,
            is_decoding=True,
            target_hidden_states=None,
            target_position_ids=query_positions.reshape(1, -1),
            target_inputs_embeds=None,
        )

    def _prepare_context_materialization(
            self, model_inputs: ModelInputs,
            extra_inputs: ARSpecExtraInputs):
        """Keep physical context layout separate from committed logical
        lengths.

        Pageable KV and validated W+N SWA rings may store the full verifier block. The next query starts at the
        committed prefix, hiding rejected tails without boolean indexing/nonzero and its host synchronization.
        Unvalidated/stateful layouts retain accepted-prefix compaction.
        """
        if not model_inputs.is_decoding or self._full_context_materialization:
            return super()._prepare_context_materialization(
                model_inputs, extra_inputs)

        if model_inputs.dp_meta is not None:
            raise ValueError('DSpark DP requires a validated full-context cache geometry.')
        target_hidden = self._flatten_target_hidden(extra_inputs)
        context_lengths = self._context_lengths(model_inputs, extra_inputs)
        query_start_positions = self._query_start_positions(
            model_inputs, context_lengths)
        batch_size = context_lengths.numel()
        query_width = model_inputs.max_q_seqlen
        keep = (torch.arange(query_width,
                             device=context_lengths.device).unsqueeze(0)
                < context_lengths.unsqueeze(1))
        input_ids = model_inputs.input_ids.reshape(
            batch_size, query_width)[keep].unsqueeze(0)
        hidden_width = target_hidden.size(-1)
        target_hidden = target_hidden.reshape(
            batch_size, query_width, hidden_width)[keep]

        context_inputs = model_inputs.clone(
            input_ids=input_ids,
            seq_length=context_lengths,
            target_hidden_states=None,
            target_position_ids=None,
            target_inputs_embeds=None,
            is_decoding=False,
        )
        return (context_inputs, target_hidden, context_lengths,
                query_start_positions)

    def _forward_query(self, query_inputs: ModelInputs,
                       cache_engine: CacheEngine):
        """Run a DSpark query block without committing mask-query V4 state.

        Query rows are proposal workspace, not committed context.  They share the bundled V4 draft's circular window
        cache, so leaving them there can evict live history once the 128-token ring wraps.  Materialized target rows
        remain committed; only the following query forward is rolled back.
        """
        state_engine = getattr(cache_engine, 'state_cache_engine', None)
        if (state_engine is None
                or 'v4_window_kv_fp8' not in state_engine.named_state_caches):
            return self._forward(query_inputs, cache_engine=cache_engine)
        transaction = state_engine.begin_v4_speculative_transaction(
            query_inputs.state_offsets,
            query_inputs.history_lengths,
            query_inputs.seq_length,
            query_inputs.max_q_seqlen,
        )
        try:
            return self._forward(query_inputs, cache_engine=cache_engine)
        finally:
            # None of the anchor/mask query KVs is the target-feature KV that
            # represents committed context.  Restore every touched row.
            state_engine.finish_v4_speculative_transaction(
                transaction, query_inputs.seq_length)

    async def propose_block(self,
                            model_inputs: ModelInputs,
                            extra_inputs: ARSpecExtraInputs,
                            cache_engine: CacheEngine,
                            guided_processors: dict | None = None):
        if guided_processors:
            raise NotImplementedError('DSpark guided decoding is not implemented in V1.')
        if cache_engine is None:
            raise RuntimeError('DSpark requires a draft cache engine.')
        if extra_inputs.next_token_ids is None:
            raise RuntimeError('DSpark requires target next_token_ids as anchors.')

        context_inputs, target_hidden, context_lengths, query_starts = \
            self._prepare_context_materialization(model_inputs, extra_inputs)
        query_inputs = self._build_query_inputs(
            model_inputs, context_lengths, extra_inputs.next_token_ids,
            query_start_positions=query_starts)
        self._materialize_context(context_inputs, target_hidden, cache_engine)
        local_batch = model_inputs.seq_length.numel()
        query_inputs = prepare_query(self, query_inputs, cache_engine)
        outputs = self._forward_query(query_inputs, cache_engine)
        batch_size = query_inputs.seq_length.numel()
        sampled = outputs.get('draft_token_ids')
        if sampled is None:
            raise RuntimeError(
                'DSpark draft model must return graph-capturable '
                'draft_token_ids during decode.')
        expected_shape = (batch_size, self.num_speculative_tokens)
        if tuple(sampled.shape) != expected_shape:
            raise RuntimeError(
                'DSpark draft_token_ids shape mismatch: '
                f'{tuple(sampled.shape)} vs {expected_shape}.')
        # CUDA graph runners reuse one memory pool across capture buckets.
        # Draft ids survive beyond this forward (for example while new
        # prefills merge into an existing decode batch), so retaining the
        # graph-owned view lets another replay overwrite them.  Own the ids
        # before returning them to ARSpecExtraInputs.
        return sampled[:local_batch].clone()
