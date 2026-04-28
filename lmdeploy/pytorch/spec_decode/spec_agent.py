# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.profiler import record_function

from lmdeploy.utils import get_logger

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, MiscConfig, ModelConfig, SpecDecodeConfig
from ..engine.cache_engine import CacheEngine
from ..engine.logits_process import FusedLogitsProcessor, SamplingInputs, _torch_topk
from ..engine.model_agent.agent import BatchedLogProbs
from ..model_inputs import ModelInputs
from ..strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..strategies.base.model_agent import ExtraInputs
from .base import BaseSpecModelAgent
from .proposers.base import build_specdecode_proposer
from .reject_sampler import RejectionSampler

if TYPE_CHECKING:
    import xgrammar as xgr

    from ..engine.guided_process import GuidedDecodingManager

logger = get_logger('lmdeploy')


def _expand_sampling_inputs(sampling_inputs: SamplingInputs, num_tokens: int) -> SamplingInputs:
    """Expand per-batch SamplingInputs to per-token by repeating each batch
    element num_tokens times via repeat_interleave.

    Args:
        sampling_inputs: SamplingInputs with batch_size elements.
        num_tokens: Number of tokens per batch element.

    Returns:
        New SamplingInputs with batch_size * num_tokens elements.
    """
    if num_tokens == 1:
        return sampling_inputs

    from dataclasses import fields
    out_dict = {}
    _SCALAR_FIELDS = {
        'max_top_k', 'min_top_p', 'max_num_logprobs',
        'max_repetition_ngram_size',
    }
    for f in fields(sampling_inputs):
        k = f.name
        v = getattr(sampling_inputs, k)
        if isinstance(v, torch.Tensor):
            v = v.repeat_interleave(num_tokens, dim=0)
            if k == 'random_offsets':
                # Each token position needs a different offset for
                # reproducible but distinct random sampling
                arange = torch.arange(num_tokens, device=v.device)
                v = v + arange.repeat(sampling_inputs.batch_size)
        elif k in _SCALAR_FIELDS:
            pass
        elif k == 'batch_size':
            v = sampling_inputs.batch_size * num_tokens
        elif isinstance(v, (list, tuple)) and len(v) == sampling_inputs.batch_size:
            v = type(v)(_item for elem in v for _item in [elem] * num_tokens)
        out_dict[k] = v

    out_dict['batch_size'] = sampling_inputs.batch_size * num_tokens
    return SamplingInputs(**out_dict)


def _slice_sampling_inputs(sampling_inputs: SamplingInputs, num_tokens: int, is_last: bool = True) -> SamplingInputs:
    """Slice expanded SamplingInputs.

    After _expand_sampling_inputs repeats each batch element num_tokens
    times, this function extracts a subset per batch element.

    Args:
        sampling_inputs: Expanded SamplingInputs with
            batch_size * num_tokens elements.
        num_tokens: Number of tokens per batch element.
        is_last: If True (default), take the last token per batch element
            (for bonus token sampling), returning batch_size elements.
            If False, take the first num_tokens-1 tokens per batch element
            (all except the last), returning
            batch_size * (num_tokens - 1) elements.

    Returns:
        Sliced SamplingInputs.
    """
    if num_tokens == 1:
        return sampling_inputs

    from dataclasses import fields

    _SCALAR_FIELDS = {
        'max_top_k', 'min_top_p', 'max_num_logprobs',
        'max_repetition_ngram_size',
    }

    batch_size = sampling_inputs.batch_size // num_tokens
    out_dict = {}
    for f in fields(sampling_inputs):
        k = f.name
        v = getattr(sampling_inputs, k)
        if isinstance(v, torch.Tensor):
            if is_last:
                v = v[num_tokens - 1::num_tokens]
            else:
                shape = v.shape
                v = v.view(batch_size, num_tokens, *shape[1:])
                v = v[:, :-1].reshape(batch_size * (num_tokens - 1), *shape[1:])
        elif k in _SCALAR_FIELDS:
            pass
        elif isinstance(v, (list, tuple)) and v is not None:
            # Skip if length doesn't match the expanded batch size (e.g.
            # empty defaults or fields that were not per-batch).
            if len(v) == sampling_inputs.batch_size:
                if is_last:
                    indices = list(range(num_tokens - 1, len(v), num_tokens))
                    v = type(v)(v[i] for i in indices)
                else:
                    indices = []
                    for b in range(batch_size):
                        start = b * num_tokens
                        indices.extend(range(start, start + num_tokens - 1))
                    v = type(v)(v[i] for i in indices)
        out_dict[k] = v

    if is_last:
        out_dict['batch_size'] = batch_size
    else:
        out_dict['batch_size'] = batch_size * (num_tokens - 1)
    return SamplingInputs(**out_dict)


class SpecModelAgent(BaseSpecModelAgent):
    """Speculative model agent."""

    def __init__(
        self,
        specdecode_config: SpecDecodeConfig,
        backend_config: BackendConfig,
        inputs_strategy,
        agent_strategy,
        misc_config: MiscConfig,
        device: str = 'cuda',
    ):
        super().__init__(specdecode_config, enable=True)

        self.backend_config = backend_config
        self.device = device
        self.cache_engine = None
        self.inputs_strategy = inputs_strategy
        self.agent_strategy = agent_strategy
        self.misc_config = misc_config
        self.rejection_sampler = RejectionSampler()
        self.proposer = build_specdecode_proposer(specdecode_config, device=device)
        self.method = specdecode_config.method
        self.model_config = specdecode_config.model_config
        self.cache_config = specdecode_config.cache_config

        # Guided decoding — set by ModelAgent after construction
        self.guided_decoding_manager = None

        # make dummy meta
        self.make_dummy_meta = self.inputs_strategy.create_make_dummy_meta(self.model_config)
        # for long context carry-over in chunked decoding
        self._prev_chunk_last = {}

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        self.cache_config = cache_config

    def set_model_config(self, model_config: ModelConfig):
        """Set model config."""
        self.model_config = model_config
        if model_config is not None:
            # make dummy meta
            self.make_dummy_meta = self.inputs_strategy.create_make_dummy_meta(self.model_config)

    def build_model(self, empty_init: bool, target_model=None, build_model_ctx=None):
        """Build draft model."""
        self.proposer.build_model(empty_init, target_model=target_model, build_model_ctx=build_model_ctx)

    def build_graph_runner(self):
        """Build graph runner."""
        backend = get_backend()
        self.proposer.model = backend.build_graph_runner(self.proposer.model,
                                                         model_config=self.model_config,
                                                         cache_config=self.cache_config,
                                                         backend_config=self.backend_config,
                                                         device=self.device)

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        if self.cache_config is not None:
            self.cache_engine = CacheEngine(self.cache_config,
                                            self.model_config,
                                            rank=0,
                                            tp_rank=0,
                                            world_size=1,
                                            cache_stream=cache_stream)

    def _prepare_inputs_from_main(self, model_inputs: ModelInputs, extra_inputs: ExtraInputs):
        """Update inputs from main model inputs."""
        next_token_ids = extra_inputs.next_token_ids
        last_token_indices = extra_inputs.last_token_indices
        # create new inputs for draft model (offset by 1 from main model)
        target_hidden_states = extra_inputs.target_hidden_states
        target_position_ids = extra_inputs.target_position_ids
        target_inputs_embeds = extra_inputs.target_inputs_embeds
        mrope_pos_ids = model_inputs.mrope_pos_ids
        seq_length = model_inputs.seq_length
        max_q_seqlen = model_inputs.max_q_seqlen
        max_kv_seqlen = model_inputs.max_kv_seqlen
        sum_kv_seqlen = model_inputs.sum_kv_seqlen
        history_lengths = model_inputs.history_lengths.clone()

        if not model_inputs.is_chunk:
            # Case A: non-chunked — shift left by 1, place next_token at end
            input_ids = model_inputs.input_ids.clone()
            input_ids[:, :-1] = model_inputs.input_ids[:, 1:]
            input_ids[:, last_token_indices] = next_token_ids

            if target_inputs_embeds is not None:
                input_embeds = target_inputs_embeds.clone()
                input_embeds[:, :-1, :] = target_inputs_embeds[:, 1:, :]
                next_token_embeds = self.proposer.embed_input_ids(next_token_ids)
                input_embeds[:, last_token_indices, :] = next_token_embeds
                target_inputs_embeds = input_embeds

        else:
            if model_inputs.is_first_chunk:
                # Case B: first chunk — skip first token, save last for next chunk
                input_ids = model_inputs.input_ids[:, 1:]
                seq_length = model_inputs.seq_length - 1
                max_q_seqlen = model_inputs.max_q_seqlen - 1
                max_kv_seqlen = model_inputs.max_kv_seqlen - 1
                sum_kv_seqlen = model_inputs.sum_kv_seqlen - 1

                target_hidden_states = self._prepare_long_context_chunk_save_last('hidden_states', target_hidden_states)
                if target_position_ids is not None:
                    target_position_ids = self._prepare_long_context_chunk_save_last(
                        'position_ids', target_position_ids)
                if target_inputs_embeds is not None:
                    target_inputs_embeds = target_inputs_embeds[:, 1:]
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_save_last('mrope_pos_ids', mrope_pos_ids)

            elif model_inputs.is_last_chunk:
                # Case C: last chunk — prepend saved last, append next_token
                seq_length = model_inputs.seq_length + 1
                max_q_seqlen = model_inputs.max_q_seqlen + 1
                last_token_indices = last_token_indices + 1
                max_kv_seqlen = model_inputs.max_kv_seqlen - 1
                sum_kv_seqlen = model_inputs.sum_kv_seqlen - 1
                history_lengths = model_inputs.history_lengths - 1
                input_ids = torch.cat([model_inputs.input_ids, next_token_ids.unsqueeze(0)], dim=-1)

                target_hidden_states = self._prepare_long_context_chunk_prepend_saved('hidden_states',
                                                                                      target_hidden_states,
                                                                                      save_last=False)
                if target_position_ids is not None:
                    target_position_ids = self._prepare_long_context_chunk_prepend_saved('position_ids',
                                                                                         target_position_ids,
                                                                                         save_last=False)
                if target_inputs_embeds is not None:
                    next_token_embeds = self.proposer.embed_input_ids(next_token_ids)[None]
                    target_inputs_embeds = torch.cat(
                        [target_inputs_embeds, next_token_embeds], dim=1)
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_prepend_saved('mrope_pos_ids',
                                                                                   mrope_pos_ids,
                                                                                   save_last=False)

                # clear cross-chunk state
                self._prev_chunk_last.clear()
            else:
                # Case D: middle chunk — prepend saved last, save current last
                input_ids = model_inputs.input_ids
                max_kv_seqlen = model_inputs.max_kv_seqlen - 1
                sum_kv_seqlen = model_inputs.sum_kv_seqlen - 1
                history_lengths = model_inputs.history_lengths - 1

                target_hidden_states = self._prepare_long_context_chunk_prepend_saved(
                    'hidden_states', target_hidden_states)
                if target_position_ids is not None:
                    target_position_ids = self._prepare_long_context_chunk_prepend_saved(
                        'position_ids', target_position_ids)
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_prepend_saved('mrope_pos_ids', mrope_pos_ids)

        new_model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            max_kv_seqlen=max_kv_seqlen,
            max_q_seqlen=max_q_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            history_lengths=history_lengths,
            block_offsets=model_inputs.block_offsets,
            num_ignored_history=model_inputs.num_ignored_history,
            is_decoding=model_inputs.is_decoding,
            target_hidden_states=target_hidden_states,
            target_position_ids=target_position_ids,
            target_inputs_embeds=target_inputs_embeds,
            mrope_pos_ids=mrope_pos_ids,
            is_chunk=model_inputs.is_chunk,
            is_first_chunk=model_inputs.is_first_chunk,
            is_last_chunk=model_inputs.is_last_chunk,
        )

        new_extra_inputs = extra_inputs.clone(
            target_hidden_states=None,
            target_inputs_embeds=None,
            target_position_ids=None,
            last_token_indices=last_token_indices,
        )
        return new_model_inputs, new_extra_inputs

    def _prepare_long_context_chunk_save_last(self, key, tensor):
        """Save the last entry of a tensor for cross-chunk carry-over."""
        self._prev_chunk_last[key] = tensor[:, -1:]
        return tensor[:, :-1]

    def _prepare_long_context_chunk_prepend_saved(self, key, tensor, save_last=True):
        """Prepend saved last entry from previous chunk."""
        saved = self._prev_chunk_last[key]
        if save_last:
            self._prev_chunk_last[key] = tensor[:, -1:]
            tensor = tensor[:, :-1]
        else:
            self._prev_chunk_last.pop(key, None)
        return torch.cat([saved, tensor], dim=1)

    async def _rejection_sampling(self, model_inputs: ModelInputs, extra_inputs: ARSpecExtraInputs,
                                  sampling_inputs: SamplingInputs):
        """Do rejection sampling."""

        @torch.inference_mode()
        def __compute_logprobs(raw_logprobs: torch.Tensor, token_ids: torch.LongTensor,
                               max_num_logprobs: int):
            """Compute logprobs."""
            if raw_logprobs is None or max_num_logprobs <= 0:
                return None

            indices = token_ids.flatten().unsqueeze(-1)
            clamped_indices = indices.clamp_min(0)
            logprobs = raw_logprobs.gather(-1, clamped_indices)
            topk_logprobs, topk_indices = _torch_topk(raw_logprobs, max_num_logprobs, dim=-1)
            logprobs = torch.cat([logprobs, topk_logprobs], dim=-1)
            indices = torch.cat([indices, topk_indices], dim=-1).to(torch.int32)
            output_logprobs = BatchedLogProbs(
                vals=logprobs,
                indices=indices,
            )
            return output_logprobs

        target_logits = extra_inputs.target_logits
        batch_size = model_inputs.seq_length.size(0)

        num_expand_sampling = 1 if not model_inputs.is_decoding else self.num_spec_tokens + 1
        expanded_sampling_inputs = _expand_sampling_inputs(sampling_inputs, num_expand_sampling)
        num_rejected_tokens = torch.zeros_like(model_inputs.seq_length)
        last_token_indices = model_inputs.seq_length.cumsum(0) - 1

        guided_processors = {}
        guided_manager = self.guided_decoding_manager
        if guided_manager:
            session_to_cleanup = sampling_inputs.session_to_cleanup
            if session_to_cleanup is not None:
                for session_id in session_to_cleanup:
                    guided_manager.remove_processor(session_id)

            if sampling_inputs.session_ctx is not None:
                guided_processors = guided_manager.get_processors(
                    sampling_inputs.session_ctx, sampling_inputs.response_formats)

        if model_inputs.is_decoding:
            if guided_processors:
                # Position-serial grammar mask via forked matchers;
                # original matchers are NOT modified.
                processed_logits, raw_logprobs = await self._guided_spec_logits_process(
                    target_logits, expanded_sampling_inputs, guided_manager,
                    guided_processors, batch_size, num_expand_sampling)
            else:
                logits_processor = FusedLogitsProcessor(
                    expanded_sampling_inputs,
                    logprobs_mode=self.misc_config.logprobs_mode,
                )
                processed_logits, raw_logprobs = await logits_processor(target_logits)

            # Bonus logits already have grammar mask applied in guided path
            bonus_logits = processed_logits[num_expand_sampling - 1::num_expand_sampling]  # [batch_size, vocab]

            bonus_sampling_inputs = _slice_sampling_inputs(expanded_sampling_inputs, num_expand_sampling)

            logits_processor = FusedLogitsProcessor(
                bonus_sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
            )
            logits_processor.sampling_inputs = bonus_sampling_inputs

            next_token_ids = logits_processor.sampling(bonus_logits)  # [batch_size]

            processed_logits = processed_logits.view(batch_size, num_expand_sampling, -1)
            # Rejection sampling on processed logits (exclude bonus position)
            target_draft_logits = processed_logits[:, :-1].contiguous()  # [batch, num_spec, vocab]
            draft_sampling_inputs = _slice_sampling_inputs(expanded_sampling_inputs, num_expand_sampling, is_last=False)
            output_token_ids, num_rejected_tokens, next_token_ids = self.rejection_sampler(
                target_draft_logits,
                extra_inputs.output_draft_token_ids,
                next_token_ids,
                sampling_inputs=draft_sampling_inputs,
            )
            last_token_indices = last_token_indices - num_rejected_tokens

            # Guided: accept final tokens on original matchers.
            # Forked matchers were used during processing, so originals are still
            # at pre-step state.  Accept rejection-sampled output + bonus token
            # to bring originals to the correct state for the next step.
            if guided_processors:
                for idx, processor in guided_processors.items():
                    n_rejected = num_rejected_tokens[idx].item()
                    n_valid_draft = self.num_spec_tokens - n_rejected
                    for pos in range(n_valid_draft):
                        tid = output_token_ids[idx, pos].item()
                        if tid >= 0:
                            guided_manager.accept_token(processor, tid)
                    guided_manager.accept_token(processor, next_token_ids[idx].item())
        else:
            # Prefill path — standard FusedLogitsProcessor handles guided decoding
            logits_processor = FusedLogitsProcessor(
                expanded_sampling_inputs,
                logprobs_mode=self.misc_config.logprobs_mode,
                guided_decoding_manager=guided_manager if guided_processors else None,
            )
            if model_inputs.is_chunk and not model_inputs.is_last_chunk:
                # dummy output, no need to sampling or compute logprobs for non-last chunk
                next_token_ids = num_rejected_tokens
                output_token_ids = num_rejected_tokens.unsqueeze(-1)
                raw_logprobs = None
            else:
                bonus_logits, raw_logprobs = await logits_processor(target_logits)
                next_token_ids = logits_processor.sampling(bonus_logits)  # [batch_size]
                output_token_ids = next_token_ids.unsqueeze(-1)

        logprobs = __compute_logprobs(raw_logprobs, output_token_ids, sampling_inputs.max_num_logprobs)

        new_extra_inputs = extra_inputs.clone(
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
            num_rejected_tokens=num_rejected_tokens,
            output_token_ids=output_token_ids,
            target_logits=None,  # clear for next step
            logprobs=logprobs,
        )
        return new_extra_inputs

    async def _guided_spec_logits_process(
        self,
        target_logits: torch.Tensor,
        expanded_sampling_inputs: SamplingInputs,
        guided_manager: GuidedDecodingManager,
        guided_processors: dict[int, xgr.GrammarMatcher],
        batch_size: int,
        num_expand: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply position-serial grammar mask to target logits for spec decode.

        Uses forked GrammarMatchers so that the original matchers are NOT
        modified.  The caller is responsible for accepting the final tokens
        on the original matchers after rejection sampling.

        All ``num_expand`` positions (including the bonus position) are masked.

        Args:
            target_logits: [batch_size * num_expand, vocab_size]
            expanded_sampling_inputs: Expanded sampling inputs.
            guided_manager: The GuidedDecodingManager instance.
            guided_processors: {orig_batch_idx: GrammarMatcher} for guided seqs.
            batch_size: Number of sequences.
            num_expand: num_spec_tokens + 1.

        Returns:
            (processed_logits, raw_logprobs) — same shapes as FusedLogitsProcessor.
        """
        # Step 1: Non-grammar logits processing (temperature, penalties, etc.)
        # FusedLogitsProcessor is created WITHOUT guided_decoding_manager so it
        # skips grammar mask — we apply it ourselves below.
        logits_processor = FusedLogitsProcessor(
            expanded_sampling_inputs,
            logprobs_mode=self.misc_config.logprobs_mode,
        )
        scores, raw_logprobs = await logits_processor(target_logits)

        if not guided_processors:
            return scores, raw_logprobs

        # Step 2: Position-serial grammar mask via forked matchers.
        scores_3d = scores.view(batch_size, num_expand, -1)

        # One fork per guided sequence; each fork is advanced in-place so
        # subsequent positions get the correct mask.
        forked = {idx: proc.fork() for idx, proc in guided_processors.items()}

        guided_bitmask = guided_manager.allocate_batched_bitmap(batch_size)
        for pos in range(num_expand):

            for idx, fork_proc in forked.items():
                guided_manager.fill_bitmap(fork_proc, guided_bitmask, idx)

            pos_logits = scores_3d[:, pos, :]
            guided_manager.apply_batched_bitmap(pos_logits, guided_bitmask)
            scores_3d[:, pos, :] = pos_logits

            # Argmax as a greedy approximation to advance the forked matcher.
            # The actual sampled token may differ, but rejection sampling
            # ensures only accepted tokens are fed to original matchers.
            pos_token_ids = pos_logits.argmax(dim=-1)

            for idx, fork_proc in forked.items():
                guided_manager.accept_token(fork_proc, pos_token_ids[idx].item())

        # Forked matchers go out of scope — originals untouched.
        scores = scores_3d.view(batch_size * num_expand, -1)
        return scores, raw_logprobs

    def _forward_impl(self, inputs: ModelInputs):
        """Forward impl."""
        output = self.proposer._forward(inputs, cache_engine=self.cache_engine)
        return output

    async def _async_model_forward(self, inputs: ModelInputs, extra_inputs: ARSpecExtraInputs,
                                   sampling_inputs: SamplingInputs):
        """Model forward.

        Args:
            inputs (dict): The input data comes from _make_inputs.
        """
        outputs = self._forward_impl(inputs)
        if inputs.is_chunk and not inputs.is_last_chunk:
            # create dummy draft tokens
            output_draft_ids = inputs.input_ids.new_zeros(1, self.num_spec_tokens)
        else:
            # Fork guided processors for draft model.
            draft_guided_processors = None
            guided_manager = self.guided_decoding_manager
            if guided_manager and sampling_inputs.session_ctx is not None:
                orig_processors = guided_manager.get_processors(
                    sampling_inputs.session_ctx, sampling_inputs.response_formats)
                draft_guided_processors = {idx: proc.fork()
                                           for idx, proc in orig_processors.items()}

            loop_count = self.num_spec_tokens - 1
            draft_token_ids, model_metas, target_hidden_states = self.proposer.get_outputs(
                outputs, inputs, extra_inputs,
                guided_processors=draft_guided_processors)
            draft_tokens_li = [draft_token_ids]
            if loop_count > 0:
                inputs = self.proposer.update_inputs_decoding(inputs, extra_inputs, draft_token_ids.transpose(0, 1),
                                                              target_hidden_states, model_metas)
                # set last_token_indices to None for decoding
                extra_inputs.last_token_indices = None

                for loop_idx in range(loop_count):
                    outputs = self._forward_impl(inputs)
                    draft_token_ids, model_metas, target_hidden_states = self.proposer.get_outputs(
                        outputs, inputs,
                        guided_processors=draft_guided_processors)
                    draft_tokens_li.append(draft_token_ids)
                    if loop_idx < loop_count - 1:
                        step_seqlens = inputs.seq_length.new_ones(inputs.seq_length.size(0))
                        inputs = inputs.step(draft_token_ids.transpose(0, 1), step_seqlens)
                        inputs.model_metas = model_metas
                        inputs.target_hidden_states = target_hidden_states
                        if inputs.target_position_ids is not None:
                            inputs.target_position_ids += 1

            output_draft_ids = torch.cat(draft_tokens_li, dim=-1)

        # create new extra inputs
        extra_inputs = ARSpecExtraInputs(
            output_draft_token_ids=output_draft_ids,
            next_token_ids=extra_inputs.next_token_ids,
            num_rejected_tokens=extra_inputs.num_rejected_tokens,
            output_token_ids=extra_inputs.output_token_ids,
            logprobs=extra_inputs.logprobs,
        )
        return extra_inputs

    async def async_model_forward(
        self,
        model_inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        sampling_inputs: SamplingInputs,
    ):
        """Draft model forward."""
        with record_function('spec_rejection_sampling'):
            draft_extra_inputs = await self._rejection_sampling(model_inputs, extra_inputs, sampling_inputs)
        draft_model_inputs, draft_extra_inputs = self._prepare_inputs_from_main(model_inputs, draft_extra_inputs)
        return await self._async_model_forward(draft_model_inputs, draft_extra_inputs, sampling_inputs)

    def warmup(self, max_batches: int, target_model_config: ModelConfig):
        """warmup."""
        target_hidden_size = self.proposer.get_target_hidden_size(target_model_config)

        # warmup prefill
        inputs = self.inputs_strategy.make_dummy(max_batches,
                                                 is_decoding=False,
                                                 device='cuda',
                                                 vocab_size=self.model_config.vocab_size,
                                                 target_hidden_size=target_hidden_size,
                                                 target_dtype=self.model_config.dtype,
                                                 meta=self.make_dummy_meta)

        self._forward_impl(inputs)

        capture_batch_sizes = self.proposer.model.get_capture_batch_sizes()
        capture_batch_sizes = sorted(capture_batch_sizes, reverse=True)

        for batch_size in capture_batch_sizes:
            # decode with num_spec_tokens + 1 per seq
            inputs = self.inputs_strategy.make_dummy(batch_size,
                                                     is_decoding=True,
                                                     device='cuda',
                                                     vocab_size=self.model_config.vocab_size,
                                                     max_q_seqlen=self.num_spec_tokens + 1,
                                                     target_hidden_size=target_hidden_size,
                                                     target_dtype=self.model_config.dtype,
                                                     meta=self.make_dummy_meta)
            self._forward_impl(inputs)
            # decode 1 tokens per sequence
            inputs = self.inputs_strategy.make_dummy(batch_size,
                                                     is_decoding=True,
                                                     device='cuda',
                                                     vocab_size=self.model_config.vocab_size,
                                                     max_q_seqlen=1,
                                                     target_hidden_size=self.model_config.hidden_size,
                                                     target_dtype=self.model_config.dtype,
                                                     meta=self.make_dummy_meta)
            self._forward_impl(inputs)

    def reset_graph_runner(self):
        """Reset graph runner."""
        if self.proposer.model is not None and hasattr(self.proposer.model, 'reset'):
            self.proposer.model.reset()

    def get_model(self):
        """Get model."""
        return self.proposer.model.get_model()
