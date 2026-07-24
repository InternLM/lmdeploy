# Copyright (c) OpenMMLab. All rights reserved.

import torch

from lmdeploy.utils import get_logger

from ...config import CacheConfig, ModelConfig
from ...engine.cache_engine import CacheEngine
from ...model_inputs import ModelInputs, step_ctx_manager
from ...strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..dflash_debug import debug_tensor, write_dflash_debug
from .base import (
    SPEC_PROPOSERS,
    BaseSpecProposer,
    ProposalContext,
    ProposalMethod,
    ProposalWarmupCase,
    ProposalWarmupPlan,
)

logger = get_logger('lmdeploy')


@SPEC_PROPOSERS.register_module(name='dflash')
class DFlash(BaseSpecProposer):
    """DFlash proposer with one-shot block drafting."""

    proposal_method = ProposalMethod.DIFFUSION
    requires_target_inputs_embeds = False

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
        if target_model is None:
            raise RuntimeError('DFlash requires the target model for shared input embeddings.')
        super().build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)
        if not hasattr(self.model, 'set_input_embeddings'):
            raise RuntimeError('DFlash draft model must implement set_input_embeddings().')
        logger.info('Using embed_tokens from target model for DFlash draft.')
        self.model.set_input_embeddings(target_model.get_input_embeddings())
        if self.model.get_input_embeddings() is None:
            raise RuntimeError('DFlash target input embeddings are not available.')

    def get_target_hidden_size(self, model_config):
        """Get concatenated DFlash target hidden size."""
        return model_config.hidden_size * len(self.specdecode_config.target_layer_ids)

    def get_warmup_plan(self,
                        max_batches: int,
                        target_model_config: ModelConfig,
                        capture_batch_sizes: list[int],
                        cache_config: CacheConfig) -> ProposalWarmupPlan:
        """Declare materialization shapes and fixed block-query graph cases."""
        if cache_config is None:
            raise RuntimeError('DFlash warmup requires a draft cache configuration.')
        target_hidden_size = self.get_target_hidden_size(target_model_config)
        cache_block_size = max(1, int(cache_config.block_size))
        prefill_token_budget = max(1, int(cache_config.max_prefill_token_num))
        prefill_q_len = min(cache_block_size, prefill_token_budget)
        prefill_batch_size = min(max_batches, max(1, prefill_token_budget // prefill_q_len))
        decode_q_len = self.num_speculative_tokens + 1
        cases = [
            ProposalWarmupCase(prefill_batch_size,
                               is_decoding=False,
                               max_q_seqlen=prefill_q_len,
                               target_hidden_size=target_hidden_size)
        ]
        cases.extend(
            ProposalWarmupCase(batch_size,
                               is_decoding=True,
                               max_q_seqlen=decode_q_len,
                               target_hidden_size=target_hidden_size) for batch_size in capture_batch_sizes)
        return ProposalWarmupPlan(cases=tuple(cases))

    def prepare_warmup_forward(self, inputs: ModelInputs, cache_engine: CacheEngine) -> ModelInputs | None:
        """Materialize target features and return only production query
        inputs."""
        if cache_engine is None:
            raise RuntimeError('DFlash warmup requires a draft cache engine.')
        cache_block_size = max(1, int(cache_engine.cache_config.block_size))
        required_blocks = (inputs.max_q_seqlen + cache_block_size - 1) // cache_block_size
        if inputs.block_offsets.size(1) < required_blocks:
            inputs.block_offsets = torch.nn.functional.pad(inputs.block_offsets,
                                                           (0, required_blocks - inputs.block_offsets.size(1)),
                                                           value=0)

        batch_size = int(inputs.seq_length.numel())
        extra_inputs = ARSpecExtraInputs(
            next_token_ids=inputs.input_ids.new_zeros(batch_size),
            target_hidden_states=inputs.target_hidden_states,
        )
        context_inputs, target_hidden, context_lengths, context_position_ids = \
            self._prepare_context_materialization(inputs, extra_inputs)
        self._materialize_context(context_inputs, target_hidden, cache_engine)
        if not inputs.is_decoding:
            return None
        return self._build_query_inputs(inputs, context_lengths, extra_inputs.next_token_ids,
                                        context_position_ids)

    def _draft_model(self):
        """Return the underlying draft nn.Module when graph runner wraps it."""
        if hasattr(self.model, 'get_model'):
            return self.model.get_model()
        return self.model

    def _flatten_target_hidden(self, extra_inputs: ARSpecExtraInputs) -> torch.Tensor:
        """Return flattened target aux hidden states."""
        target_hidden = extra_inputs.target_hidden_states
        if target_hidden is None:
            raise RuntimeError('DFlash requires target aux hidden states from the main model.')
        if target_hidden.dim() == 3:
            target_hidden = target_hidden[0]
        if target_hidden.dim() != 2:
            raise RuntimeError(f'DFlash expected target hidden states with shape [N, H], got {target_hidden.shape}.')
        return target_hidden

    def _context_lengths(self, model_inputs: ModelInputs, extra_inputs: ARSpecExtraInputs):
        """Resolve committed context lengths visible to DFlash."""
        context_lengths = model_inputs.seq_length
        if extra_inputs.num_rejected_tokens is not None:
            context_lengths = context_lengths - extra_inputs.num_rejected_tokens.to(context_lengths)
        return context_lengths

    @staticmethod
    def _slice_by_lengths(tensor: torch.Tensor,
                          seq_lengths: torch.Tensor,
                          keep_lengths: torch.Tensor,
                          max_seq_length: int,
                          preserve_features: bool = False):
        """Slice a flattened per-token tensor by per-request valid lengths."""
        if preserve_features:
            if tensor.dim() == 3 and tensor.size(0) == 1:
                flat_tensor = tensor[0]
            elif tensor.dim() >= 2:
                flat_tensor = tensor.flatten(0, -2)
            else:
                flat_tensor = tensor.reshape(-1)
        elif tensor.dim() == 2 and tensor.size(0) == 1:
            flat_tensor = tensor[0]
        elif tensor.dim() >= 2:
            flat_tensor = tensor.flatten(0, -2)
        else:
            flat_tensor = tensor.reshape(-1)
        starts = seq_lengths.cumsum(0) - seq_lengths
        offsets = torch.arange(max_seq_length, device=seq_lengths.device)
        valid = offsets[None, :] < keep_lengths[:, None]
        indices = starts[:, None] + offsets[None, :]
        indices = indices[valid]
        return flat_tensor.index_select(0, indices)

    def _slice_target_position_ids(self, model_inputs: ModelInputs, context_lengths: torch.Tensor):
        """Slice explicit target positions when the target path provides
        them."""
        target_position_ids = model_inputs.target_position_ids
        if target_position_ids is None:
            return None
        if target_position_ids.dim() == 2 and target_position_ids.size(0) == 1:
            target_position_ids = target_position_ids[0]
        if target_position_ids.dim() != 1:
            raise RuntimeError('DFlash supports only 1D target_position_ids for draft context materialization, '
                               f'got shape={tuple(target_position_ids.shape)}.')
        return self._slice_by_lengths(target_position_ids,
                                      model_inputs.seq_length,
                                      context_lengths,
                                      max_seq_length=model_inputs.max_q_seqlen)

    def _build_context_inputs(self,
                              model_inputs: ModelInputs,
                              context_lengths: torch.Tensor,
                              context_position_ids: torch.Tensor | None = None):
        """Build draft-cache materialization inputs for committed context
        tokens."""
        context_ids = self._slice_by_lengths(model_inputs.input_ids,
                                             model_inputs.seq_length,
                                             context_lengths,
                                             max_seq_length=model_inputs.max_q_seqlen)
        target_position_ids = None if context_position_ids is None else context_position_ids.unsqueeze(0)
        return model_inputs.clone(
            input_ids=context_ids.unsqueeze(0),
            seq_length=context_lengths,
            max_q_seqlen=model_inputs.max_q_seqlen,
            max_kv_seqlen=model_inputs.max_kv_seqlen,
            sum_kv_seqlen=model_inputs.sum_kv_seqlen,
            is_decoding=False,
            target_hidden_states=None,
            target_position_ids=target_position_ids,
            target_inputs_embeds=None,
        )

    def _build_query_inputs(self, model_inputs: ModelInputs, context_lengths: torch.Tensor,
                            next_token_ids: torch.Tensor, context_position_ids: torch.Tensor | None = None):
        """Build one DFlash query block per request: [next, mask, ...]."""
        batch_size = int(model_inputs.seq_length.numel())
        query_len = self.num_speculative_tokens + 1
        query_ids = model_inputs.input_ids.new_full((batch_size, query_len), int(self.specdecode_config.mask_token_id))
        query_ids[:, 0] = next_token_ids
        query_history = model_inputs.history_lengths + context_lengths
        if context_position_ids is None:
            query_start_positions = query_history
        else:
            starts = context_lengths.cumsum(0) - context_lengths
            query_start_positions = context_position_ids[starts + context_lengths - 1] + 1
        query_positions = query_start_positions[:, None] + torch.arange(query_len, device=query_history.device)[None, :]
        return model_inputs.clone(
            input_ids=query_ids.reshape(1, -1),
            seq_length=model_inputs.seq_length.new_full((batch_size, ), query_len),
            history_lengths=query_history,
            max_q_seqlen=query_len,
            max_kv_seqlen=model_inputs.max_kv_seqlen + query_len,
            sum_kv_seqlen=model_inputs.sum_kv_seqlen + batch_size * query_len,
            # DFlash queries are fixed-size speculative blocks over paged
            # context.  Mark them as decoding so FA3 reads the block table
            # directly instead of flattening the full KV history as prefill.
            is_decoding=True,
            target_hidden_states=None,
            target_position_ids=query_positions.reshape(1, -1),
            target_inputs_embeds=None,
        )

    def _materialize_context(
        self,
        context_inputs: ModelInputs,
        target_hidden: torch.Tensor,
        cache_engine: CacheEngine,
    ):
        """Project target aux hidden states into the draft KV cache."""
        if target_hidden.numel() == 0:
            return
        kv_caches = cache_engine.gpu_cache
        ctx_mgr = self.model.ctx_mgr
        with step_ctx_manager(ctx_mgr):
            context = ctx_mgr.build_context(
                inputs=context_inputs,
                model_config=self.specdecode_config.model_config,
                cache_config=cache_engine.cache_config,
                kv_caches=kv_caches,
            )
            with ctx_mgr.context(context):
                self._draft_model().precompute_and_store_context_kv(
                    target_hidden=target_hidden,
                    position_ids=context.position_ids.flatten(),
                    past_key_values=kv_caches,
                    attn_metadata=context.attn_metadata,
                    max_q_seqlen=context_inputs.max_q_seqlen,
                )

    def _prepare_context_materialization(self, model_inputs: ModelInputs, extra_inputs: ARSpecExtraInputs):
        """Build inputs and target states for DFlash context K/V
        materialization."""
        target_hidden = self._flatten_target_hidden(extra_inputs)
        context_lengths = self._context_lengths(model_inputs, extra_inputs)
        context_position_ids = self._slice_target_position_ids(model_inputs, context_lengths)
        context_inputs = self._build_context_inputs(model_inputs, context_lengths, context_position_ids)
        target_hidden = self._slice_by_lengths(target_hidden,
                                               model_inputs.seq_length,
                                               context_lengths,
                                               max_seq_length=model_inputs.max_q_seqlen,
                                               preserve_features=True)
        return context_inputs, target_hidden, context_lengths, context_position_ids

    def materialize_context(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ARSpecExtraInputs,
        cache_engine: CacheEngine,
    ):
        """Materialize DFlash context K/V without running the mask-query
        draft."""
        if cache_engine is None:
            raise RuntimeError('DFlash requires a draft cache engine for context K/V materialization.')

        context_inputs, target_hidden, _, _ = self._prepare_context_materialization(model_inputs, extra_inputs)
        self._materialize_context(context_inputs, target_hidden, cache_engine)

    async def propose_block(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ARSpecExtraInputs,
        cache_engine: CacheEngine,
        guided_processors: dict | None = None,
    ):
        """Run a one-shot DFlash draft proposal."""
        if guided_processors:
            raise NotImplementedError('DFlash guided decoding is not implemented yet.')
        if cache_engine is None:
            raise RuntimeError('DFlash requires a draft cache engine for context K/V materialization.')
        if extra_inputs.next_token_ids is None:
            raise RuntimeError('DFlash requires sampled next_token_ids from the target model.')

        context_inputs, target_hidden, context_lengths, context_position_ids = self._prepare_context_materialization(
            model_inputs, extra_inputs)
        query_inputs = self._build_query_inputs(model_inputs, context_lengths, extra_inputs.next_token_ids,
                                                context_position_ids)

        self._materialize_context(context_inputs, target_hidden, cache_engine)
        outputs = self._forward(query_inputs, cache_engine=cache_engine)
        hidden_states = outputs['hidden_states']
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[0]

        batch_size = int(query_inputs.seq_length.numel())
        query_len = self.num_speculative_tokens + 1
        mask_indices = torch.arange(batch_size * query_len, device=hidden_states.device).view(batch_size, query_len)
        mask_indices = mask_indices[:, 1:].reshape(-1)
        logits = self.get_logits(hidden_states[mask_indices][None])[0]
        draft_token_ids = logits.argmax(dim=-1).view(batch_size, self.num_speculative_tokens)
        return draft_token_ids

    async def propose(self,
                      model_inputs: ModelInputs,
                      extra_inputs: ARSpecExtraInputs,
                      sampling_inputs,
                      proposal_ctx: ProposalContext | None = None):
        """Produce all DFlash draft tokens in one block proposal."""
        if proposal_ctx is None:
            raise RuntimeError('DFlash requires ProposalContext with a draft cache engine.')
        orig_processors = self.guided_helper.get_processors(
            sampling_inputs.session_ctx if sampling_inputs else None,
            sampling_inputs.response_formats if sampling_inputs else None)
        if orig_processors:
            raise NotImplementedError('DFlash guided decoding is not implemented yet.')

        if model_inputs.is_chunk and not model_inputs.is_last_chunk:
            self.materialize_context(model_inputs, extra_inputs, proposal_ctx.cache_engine)
            output_draft_ids = model_inputs.input_ids.new_zeros(model_inputs.seq_length.size(0),
                                                                self.num_speculative_tokens)
        else:
            output_draft_ids = await self.propose_block(model_inputs, extra_inputs, proposal_ctx.cache_engine)

        write_dflash_debug(proposal_ctx.rank, 'proposal', lambda: {
            'step': proposal_ctx.debug_step,
            'is_decoding': bool(model_inputs.is_decoding),
            'seq_length': debug_tensor(model_inputs.seq_length),
            'history_lengths': debug_tensor(model_inputs.history_lengths),
            'input_ids': debug_tensor(model_inputs.input_ids),
            'next_token_ids': debug_tensor(extra_inputs.next_token_ids),
            'prev_num_rejected_tokens': debug_tensor(extra_inputs.num_rejected_tokens),
            'draft_token_ids': debug_tensor(output_draft_ids),
        })

        return ARSpecExtraInputs(
            output_draft_token_ids=output_draft_ids,
            next_token_ids=extra_inputs.next_token_ids,
            num_rejected_tokens=extra_inputs.num_rejected_tokens,
            output_token_ids=extra_inputs.output_token_ids,
            logprobs=extra_inputs.logprobs,
        )

    async def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ARSpecExtraInputs = None,
                    guided_processors: dict | None = None):
        """Get DFlash draft outputs."""
        raise NotImplementedError('DFlash requires the block-proposal hook; the autoregressive draft loop is invalid '
                                  'because context K/V must be materialized before a one-shot mask query forward.')
