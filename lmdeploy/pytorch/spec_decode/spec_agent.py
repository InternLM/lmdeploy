# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from typing import Dict

import torch

from lmdeploy.utils import get_logger

from ..backends import get_backend
from ..config import BackendConfig, CacheConfig, ModelConfig, SpecDecodeConfig
from ..distributed import DistContext
from ..engine.cache_engine import CacheEngine
from ..engine.logits_process import SamplingInputs
from ..model_inputs import ModelInputs
from ..strategies.ar_spec.model_agent import ARSpecExtraInputs
from ..strategies.base.model_agent import ExtraInputs
from .proposers.base import build_specdecode_proposer
from .reject_sampler import RejectionSampler

logger = get_logger('lmdeploy')


class SpecModelAgent:
    """Speculative model agent."""

    def __init__(
        self,
        specdecode_config: SpecDecodeConfig,
        backend_config: BackendConfig,
        dist_ctx: DistContext,
        inputs_strategy,
        agent_strategy,
        device: str = 'cuda',
    ):
        self.method = specdecode_config.method
        self.model_config = specdecode_config.model_config
        self.cache_config = specdecode_config.cache_config
        self.num_spec_tokens = specdecode_config.num_speculative_tokens
        self.backend_config = backend_config
        self.device = device
        self.dist_ctx = dist_ctx

        self.proposer = build_specdecode_proposer(specdecode_config, device=device)
        self.cache_engine = None
        self.inputs_strategy = inputs_strategy
        self.agent_strategy = agent_strategy
        self.to_runable = dist_ctx.dp > 1 or dist_ctx.rank % dist_ctx.tp == 0
        self.rejection_sampler = RejectionSampler()

    def set_cache_config(self, cache_config: CacheConfig):
        """Set all cache config."""
        self.cache_config = cache_config

    def set_model_config(self, model_config: ModelConfig):
        """Set model config."""
        self.model_config = model_config

    def build_model(self, empty_init: bool, target_model=None, model_format=None, build_model_ctx=None):
        """Build draft model."""
        if not self.to_runable:
            return
        self.proposer.build_model(empty_init,
                                  target_model=target_model,
                                  model_format=model_format,
                                  build_model_ctx=build_model_ctx)

    def build_graph_runner(self):
        """Build graph runner."""
        if not self.to_runable:
            return
        backend = get_backend()
        self.proposer.model = backend.build_graph_runner(self.proposer.model,
                                                         model_config=self.model_config,
                                                         cache_config=self.cache_config,
                                                         backend_config=self.backend_config,
                                                         device=self.device)

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        if not self.to_runable:
            return
        if self.cache_config is not None:
            self.cache_engine = CacheEngine(self.cache_config,
                                            self.model_config,
                                            rank=0,
                                            tp_rank=0,
                                            world_size=1,
                                            cache_stream=cache_stream)

    def rejection_sampling(self, next_token_ids, model_inputs: 'ModelInputs', extra_inputs: ExtraInputs):
        """Do rejection sampling."""
        num_rejected_tokens = None
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

        # create new inputs
        input_ids = model_inputs.input_ids.clone()
        seq_length = model_inputs.seq_length
        # # offset by 1 token
        input_ids[:, :-1] = model_inputs.input_ids[:, 1:]
        # # update next tokens
        input_ids[:, last_token_indices] = next_token_ids
        # use new inputs
        new_model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            max_kv_seqlen=model_inputs.max_kv_seqlen,
            max_q_seqlen=model_inputs.max_q_seqlen,
            sum_kv_seqlen=model_inputs.sum_kv_seqlen,
            history_lengths=model_inputs.history_lengths.clone(),
            block_offsets=model_inputs.block_offsets,
            num_ignored_history=model_inputs.num_ignored_history,
            is_decoding=model_inputs.is_decoding,
            target_hidden_states=extra_inputs.target_hidden_states,
            target_position_ids=extra_inputs.target_position_ids,
        )
        new_extra_inputs = ARSpecExtraInputs(
            next_token_ids=next_token_ids,
            last_token_indices=last_token_indices,
            num_rejected_tokens=num_rejected_tokens,
            output_token_ids=output_token_ids,
            loop_last_step=extra_inputs.loop_last_step,
        )
        return new_model_inputs, new_extra_inputs

    def _forward_impl(self, inputs: ModelInputs):
        """Forward impl."""
        output = self.proposer._forward(inputs, cache_engine=self.cache_engine)
        return output

    async def async_forward(self, inputs: ModelInputs):
        """Model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
        """
        output = self._forward_impl(inputs)
        await asyncio.sleep(0)
        return output

    async def _async_model_forward(self, inputs: ModelInputs, extra_inputs: ExtraInputs,
                                   sampling_inputs: SamplingInputs):
        """Model forward.

        Args:
            inputs (Dict): The input data comes from _make_inputs.
        """
        max_prefill_token_num = self.cache_config.max_prefill_token_num

        async def __long_context_single_forward(new_inputs):
            """One large sequence."""
            model_metas = new_inputs[0].model_metas
            for inp in new_inputs:
                inp.model_metas = model_metas
                output = await self.async_forward(inp)
                model_metas = output.get('model_metas')
            return output

        # make long context inputs
        is_long_context = inputs.input_ids.numel() > max_prefill_token_num and not inputs.is_decoding

        if is_long_context:
            seq_len = inputs.seq_length
            batch_size = seq_len.size(0)
            assert batch_size == 1, 'Do not support batched long context.'
            inputs_li = inputs.split(max_prefill_token_num)
            outputs = await __long_context_single_forward(inputs_li)
        else:
            outputs = await self.async_forward(inputs)

        loop_count = self.num_spec_tokens - 1
        draft_token_ids, model_metas, target_hidden_states = self.proposer.get_outputs(outputs, inputs, extra_inputs)
        draft_tokens_li = [draft_token_ids]
        if loop_count > 0:
            # set last_token_indices to None for decoding
            extra_inputs.last_token_indices = None
            inputs = self.proposer.update_inputs_decoding(inputs, extra_inputs, draft_token_ids.transpose(0, 1),
                                                          target_hidden_states, model_metas)
            for loop_idx in range(loop_count):
                outputs = await self.async_forward(inputs)
                draft_token_ids, model_metas, target_hidden_states = self.proposer.get_outputs(outputs, inputs)
                draft_tokens_li.append(draft_token_ids)
                if loop_idx < loop_count - 1:
                    step_seqlens = inputs.seq_length.new_ones(inputs.seq_length.size(0))
                    inputs.step(draft_token_ids.transpose(0, 1), step_seqlens)
                    inputs.model_metas = model_metas
                    inputs.target_hidden_states = target_hidden_states
                    if inputs.target_position_ids is not None:
                        inputs.target_position_ids += 1

        output_draft_ids = torch.cat(draft_tokens_li, dim=-1)
        return output_draft_ids

    async def async_model_forward(
        self,
        next_token_ids: torch.Tensor,
        model_inputs: ModelInputs,
        extra_inputs: ExtraInputs,
        sampling_inputs: SamplingInputs,
    ):
        """Draft model forward."""
        draft_model_inputs, draft_extra_inputs = self.rejection_sampling(next_token_ids, model_inputs, extra_inputs)
        next_draft_ids = await self._async_model_forward(draft_model_inputs, draft_extra_inputs, sampling_inputs)
        draft_extra_inputs.output_draft_token_ids = next_draft_ids
        return draft_extra_inputs

    def warmup(self, max_batches: int, target_model_config: ModelConfig):
        """warmup."""
        if not self.to_runable:
            return

        target_hidden_size = self.proposer.get_target_hidden_size(target_model_config)

        # warmup prefill
        inputs = self.inputs_strategy.make_dummy(max_batches,
                                                 is_decoding=False,
                                                 device='cuda',
                                                 vocab_size=self.model_config.vocab_size,
                                                 target_hidden_size=target_hidden_size,
                                                 target_dtype=self.model_config.dtype)

        self._forward_impl(inputs)

        capture_batch_sizes = self.proposer.model.get_capture_batch_sizes()
        capture_batch_sizes = sorted(capture_batch_sizes, reverse=True)

        for batch_size in capture_batch_sizes:
            # decode with num_spec_tokens + 1 per seq
            inputs = self.inputs_strategy.make_dummy(
                batch_size,
                is_decoding=True,
                device='cuda',
                vocab_size=self.model_config.vocab_size,
                max_q_seqlen=self.num_spec_tokens + 1,
                target_hidden_size=target_hidden_size,
                target_dtype=self.model_config.dtype,
            )
            self._forward_impl(inputs)
            # decode 1 tokens per sequence
            inputs = self.inputs_strategy.make_dummy(
                batch_size,
                is_decoding=True,
                device='cuda',
                vocab_size=self.model_config.vocab_size,
                max_q_seqlen=1,
                target_hidden_size=self.model_config.hidden_size,
                target_dtype=self.model_config.dtype,
            )
            self._forward_impl(inputs)

    def update_main_model_outputs(self, output: Dict[str, torch.Tensor], model_inputs: ModelInputs):
        """Update outputs of main model."""
        hidden_states = output['hidden_states']
        if not model_inputs.is_decoding:
            logits_indices = model_inputs.seq_length.cumsum(0) - 1
            hidden_states = hidden_states[:, logits_indices]
        if 'aux_hidden_states' in output:
            # replace with aux
            output['hidden_states'] = output.pop('aux_hidden_states')
        return hidden_states, output
