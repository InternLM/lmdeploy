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

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None,
                    build_model_ctx=None):
        """Build the draft and attach target embeddings and a shareable output
        head."""
        if target_model is None:
            raise RuntimeError('DSpark requires the target model for shared modules.')
        BaseSpecProposer.build_model(self,
                                     empty_init,
                                     target_model=target_model,
                                     build_model_ctx=build_model_ctx)
        self._validate_context_materialization()
        self.model.set_input_embeddings(target_model.get_input_embeddings())
        output_head = getattr(target_model, 'head', None)
        if output_head is None:
            output_head = getattr(target_model, 'lm_head', None)
        self.model.set_output_head(output_head)
        logger.info('Using target embeddings and output head for DSpark draft.')

    @property
    def draft_query_len(self) -> int:
        """Return the fixed query width resolved from the checkpoint layout."""
        return self.specdecode_config.dspark.draft_query_len

    def _validate_context_materialization(self):
        if not getattr(self.model, 'supports_full_context_materialization', False):
            raise ValueError('DSpark requires a validated full-context cache geometry.')

    def get_warmup_plan(self,
                        max_batches: int,
                        target_model_config: ModelConfig,
                        capture_batch_sizes: list[int],
                        cache_config: CacheConfig) -> ProposalWarmupPlan:
        """Plan context prefill and fixed-width query warmup for each graph
        bucket."""
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

    async def propose_block(self,
                            model_inputs: ModelInputs,
                            extra_inputs: ARSpecExtraInputs,
                            cache_engine: CacheEngine):
        """Materialize context and return owned proposal IDs with DP padding
        removed."""
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
        outputs = self._forward(query_inputs, cache_engine=cache_engine)
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
