# Copyright (c) OpenMMLab. All rights reserved.

import asyncio

import torch

from lmdeploy.utils import get_logger

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, ModelConfig, SpecDecodeConfig
from ..engine.cache_engine import CacheEngine
from ..engine.logits_process import SamplingInputs
from ..model_inputs import ModelInputs
from ..strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..strategies.base.model_agent import ExtraInputs
from .base import BaseSpecModelAgent
from .proposers.base import build_specdecode_proposer
from .reject_sampler import RejectionSampler

logger = get_logger('lmdeploy')


class SpecModelAgent(BaseSpecModelAgent):
    """Speculative model agent."""

    def __init__(
        self,
        specdecode_config: SpecDecodeConfig,
        backend_config: BackendConfig,
        inputs_strategy,
        agent_strategy,
        device: str = 'cuda',
    ):
        super().__init__(specdecode_config, enable=True)

        self.backend_config = backend_config
        self.device = device
        self.cache_engine = None
        self.inputs_strategy = inputs_strategy
        self.agent_strategy = agent_strategy
        self.rejection_sampler = RejectionSampler()
        self.proposer = build_specdecode_proposer(specdecode_config, device=device)
        self.method = specdecode_config.method
        self.model_config = specdecode_config.model_config
        self.cache_config = specdecode_config.cache_config

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
                    target_inputs_embeds = self._prepare_long_context_chunk_save_last(
                        'input_embeds', target_inputs_embeds)
                if mrope_pos_ids is not None:
                    mrope_pos_ids = self._prepare_long_context_chunk_save_last('mrope_pos_ids', mrope_pos_ids)

            elif model_inputs.is_last_chunk:
                # Case C: last chunk — prepend saved last, append next_token
                seq_length = model_inputs.seq_length + 1
                max_q_seqlen = model_inputs.max_q_seqlen + 1
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
                    saved = self._prev_chunk_last['input_embeds']
                    next_token_embeds = self.proposer.embed_input_ids(next_token_ids)
                    target_inputs_embeds = torch.cat(
                        [saved, target_inputs_embeds, next_token_embeds.unsqueeze(1)], dim=1)
                    self._prev_chunk_last.pop('input_embeds', None)
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
                if target_inputs_embeds is not None:
                    target_inputs_embeds = self._prepare_long_context_chunk_prepend_saved(
                        'input_embeds', target_inputs_embeds)
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

    def _rejection_sampling(self, next_token_ids, model_inputs: 'ModelInputs', extra_inputs: ARSpecExtraInputs):
        """Do rejection sampling."""
        num_rejected_tokens = torch.zeros_like(model_inputs.seq_length)
        bonus_token_ids = output_token_ids = next_token_ids.unsqueeze(-1)
        last_token_indices = model_inputs.seq_length.cumsum(0) - 1
        if model_inputs.is_decoding:
            # only do rejection sample for decoding with draft tokens
            input_draft_token_ids = model_inputs.input_ids.squeeze(0).unflatten(0, (-1, self.num_spec_tokens + 1))[:,
                                                                                                                   1:]
            output_token_ids, num_rejected_tokens, next_token_ids = self.rejection_sampler(
                extra_inputs.target_logits,
                input_draft_token_ids,
                bonus_token_ids,
            )
            # update last token indices
            last_token_indices = last_token_indices - num_rejected_tokens
        new_extra_inputs = extra_inputs.clone(
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
            num_rejected_tokens=num_rejected_tokens,
            output_token_ids=output_token_ids,
        )
        return new_extra_inputs

    def _forward_impl(self, inputs: ModelInputs):
        """Forward impl."""
        output = self.proposer._forward(inputs, cache_engine=self.cache_engine)
        return output

    async def _async_model_forward(self, inputs: ModelInputs, extra_inputs: ARSpecExtraInputs,
                                   sampling_inputs: SamplingInputs):
        """Model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
        """
        outputs = self._forward_impl(inputs)
        if inputs.is_chunk and not inputs.is_last_chunk:
            await asyncio.sleep(0)
            return torch.zeros_like(inputs.input_ids)

        loop_count = self.num_spec_tokens - 1
        draft_token_ids, model_metas, target_hidden_states = self.proposer.get_outputs(outputs, inputs, extra_inputs)
        draft_tokens_li = [draft_token_ids]
        if loop_count > 0:
            # set last_token_indices to None for decoding
            extra_inputs.last_token_indices = None
            inputs = self.proposer.update_inputs_decoding(inputs, extra_inputs, draft_token_ids.transpose(0, 1),
                                                          target_hidden_states, model_metas)
            for loop_idx in range(loop_count):
                outputs = self._forward_impl(inputs)
                draft_token_ids, model_metas, target_hidden_states = self.proposer.get_outputs(outputs, inputs)
                draft_tokens_li.append(draft_token_ids)
                if loop_idx < loop_count - 1:
                    step_seqlens = inputs.seq_length.new_ones(inputs.seq_length.size(0))
                    inputs = inputs.step(draft_token_ids.transpose(0, 1), step_seqlens)
                    inputs.model_metas = model_metas
                    inputs.target_hidden_states = target_hidden_states
                    if inputs.target_position_ids is not None:
                        inputs.target_position_ids += 1

        output_draft_ids = torch.cat(draft_tokens_li, dim=-1)
        await asyncio.sleep(0)
        return output_draft_ids

    async def async_model_forward(
        self,
        next_token_ids: torch.Tensor,
        model_inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        sampling_inputs: SamplingInputs,
    ):
        """Draft model forward."""
        draft_extra_inputs = self._rejection_sampling(next_token_ids, model_inputs, extra_inputs)
        draft_model_inputs, draft_extra_inputs = self._prepare_inputs_from_main(model_inputs, draft_extra_inputs)
        next_draft_ids = await self._async_model_forward(draft_model_inputs, draft_extra_inputs, sampling_inputs)
        draft_extra_inputs.output_draft_token_ids = next_draft_ids
        return draft_extra_inputs

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
        'reset graph runner'
        if self.proposer.model is not None and hasattr(self.proposer.model, 'reset'):
            self.proposer.model.reset()
