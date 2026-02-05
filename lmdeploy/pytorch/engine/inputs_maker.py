# Copyright (c) OpenMMLab. All rights reserved.
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch
from torch.profiler import record_function

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.messages import MessageStatus
from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta, VisionModelInputs
from lmdeploy.utils import get_logger

if TYPE_CHECKING:
    from lmdeploy.pytorch.adapter.adapter import AdapterManager
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.multimodal.data_type import MultiModalInputs
    from lmdeploy.pytorch.paging import Scheduler
    from lmdeploy.pytorch.strategies.base.engine import EngineStrategy
    from lmdeploy.pytorch.strategies.base.model_agent import ModelAgentStrategy
    from lmdeploy.pytorch.strategies.base.sampling import SamplingStrategy

    from .engine import Engine, SeqList
    from .executor import ExecutorBase

logger = get_logger('lmdeploy')


def _tensorlize_block_offsets(block_offsets, dtype=torch.int32):
    """Tensorlize block_offsets."""
    # copy on numpy is faster than torch.nn.utils.rnn.pad_sequence
    batch_size = len(block_offsets)
    max_len = max([len(off) for off in block_offsets])
    out = np.zeros((batch_size, max_len), dtype=block_offsets[0].dtype)

    for idx, off in enumerate(block_offsets):
        off_len = len(off)
        out[idx, :off_len] = off
    return torch.as_tensor(out, dtype=dtype)


@dataclass
class InputsMakerConfig:
    """Input maker config.

    This config is added for Dependency Injection
    """
    max_batches: int
    max_prefill_token_num: int
    role: EngineRole
    is_ssm: bool = False
    dp: int = 1
    spec_decoding: bool = False
    enable_chunked_prefill: bool = False

    @staticmethod
    def from_engine(engine: 'Engine'):
        cache_config = engine.cache_config
        return InputsMakerConfig(
            spec_decoding=engine.specdecode_config is not None,
            max_batches=cache_config.max_batches,
            max_prefill_token_num=cache_config.max_prefill_token_num,
            role=cache_config.role,
            is_ssm=len(cache_config.states_shapes) > 0,
            dp=engine.dist_config.dp,
            enable_chunked_prefill=engine.misc_config.enable_chunked_prefill,
        )


class LongContextChunker:
    """Long context chunker."""

    def __init__(self, max_prefill_token_num: int):
        self.max_prefill_token_num = max_prefill_token_num

        # long prefill seq
        self.clear()

    def enabled(self):
        """Is enabled."""
        return self.seq is not None

    def is_long_context(self, seq: 'SchedulerSequence'):
        """Is long context."""
        return seq.num_token_ids > self.max_prefill_token_num

    def set_seq(self, seq: 'SchedulerSequence'):
        """Set seq."""
        self.seq = seq
        self.next_step = seq.num_history_ids

        # fill multimodals
        # if image size exceeds max_prefill_token_num, enlarge it
        max_prefill_num = self.max_prefill_token_num
        mm = seq.get_input_multimodals()
        self.multimodals = defaultdict(list)
        for key, value in mm.items():
            # sorted by start
            value = sorted(value, key=lambda x: x.start)
            self.multimodals[key] = value
            max_mm_size = max([v.end - v.start for v in value], default=0)
            max_prefill_num = max(max_prefill_num, max_mm_size)

        self.max_prefill_num = max_prefill_num

    def multimodal_iter(self):
        """Multimodal iterator."""
        multimodal_data = []
        for modal_type, modal_datas in self.multimodals.items():
            if len(modal_datas) == 0:
                continue
            multimodal_data += [(modal_type, data) for data in modal_datas]

        multimodal_data = sorted(multimodal_data, key=lambda x: x[1].start)
        for modal_type, data in multimodal_data:
            yield modal_type, data

    def next_chunk_size(self):
        """Get chunk size."""
        seq = self.seq
        if seq is None:
            return 0, None

        llm_chunk_size = min(seq.num_token_ids, self.max_prefill_num)

        if len(self.multimodals) == 0:
            # no vlm inputs found
            return llm_chunk_size, None

        start = seq.num_history_ids
        end = start + llm_chunk_size
        out_multimodals: 'MultiModalInputs' = defaultdict(list)
        for modal_type, mm in self.multimodal_iter():
            assert mm.start >= start, 'multimodal data should be sorted by start'
            if mm.start >= end:
                # | start ... end ... mm.start ... mm.end |
                # if start is beyond threshold, stop
                break

            if mm.end > end:
                # | start ... mm.start ... end ... mm.end |
                # assume multimodals not overlap
                end = mm.start
                break

            # | start ... mm.start ... mm.end ... end |
            out_multimodals[modal_type].append(mm)

        return end - start, out_multimodals

    def is_last_chunk(self):
        """Is last chunk."""
        if self.seq is None:
            return True
        return self.seq.num_token_ids <= self.max_prefill_num

    def clear(self):
        """Clear."""
        self.seq: 'SchedulerSequence' = None
        self.multimodals: MultiModalInputs = defaultdict(list)
        self.next_step: int = 0
        self.max_prefill_num: int = self.max_prefill_token_num

    def update_step(self, inputs: ModelInputs):
        """Step chunker."""
        if self.seq is None:
            return
        if self.is_last_chunk():
            # last chunk should be treated as normal prefill
            return
        assert inputs.is_chunk
        chunk_size = inputs.max_q_seqlen
        self.next_step += chunk_size
        self.seq.set_step(self.next_step)

        # remove used multimodals
        for mms in self.multimodals.values():
            while len(mms) > 0 and mms[0].end <= self.next_step:
                mms.pop(0)
        self.multimodals = dict((k, v) for k, v in self.multimodals.items() if len(v) > 0)

    def check_enable(self):
        if not self.enabled():
            return
        if self.seq.status != MessageStatus.RUNNING:
            self.clear()


class InputsMakerAsync:

    def __init__(
        self,
        executor: 'ExecutorBase',
        scheduler: 'Scheduler',
        adapter_manager: 'AdapterManager',
        engine_strategy: 'EngineStrategy',
        sampling_strategy: 'SamplingStrategy',
        model_agent_strategy: 'ModelAgentStrategy',
        config: InputsMakerConfig,
    ):
        self.executor = executor
        self.scheduler = scheduler
        self.adapter_manager = adapter_manager
        self.config = config
        self.spec_decoding = config.spec_decoding

        # strategies
        self.engine_strategy = engine_strategy
        self.sampling_strategy = sampling_strategy
        self.model_agent_strategy = model_agent_strategy

        self._init_do_prefill(config)

        # record for next forward.
        self.next_is_prefill = True
        self.forward_inputs = None

        # running seqs
        # mark the seqs that have been sent to executor
        self.running_seqs: List['SchedulerSequence'] = []
        self.to_evict_seqs: List['SchedulerSequence'] = []

        # long context chunker
        self.long_context_chunker = LongContextChunker(config.max_prefill_token_num)

    def _init_do_prefill(self, config: InputsMakerConfig):
        if config.role == EngineRole.Prefill:
            self.do_prefill = self.do_prefill_pnode
        elif config.enable_chunked_prefill:
            self.do_prefill = self.do_prefill_chunked
        else:
            self.do_prefill = self.do_prefill_default

    def _create_vision_model_inputs(self, messages: 'SeqList', model_inputs: ModelInputs):
        """Create vision model inputs."""
        batch_size = len(messages)

        def __get_vlm_embeddings():
            """Get vlm input embeddings and indexings."""
            max_q_seq_length = model_inputs.seq_length.max().item()
            input_embeddings = [[
                emb.embeddings if isinstance(emb.embeddings, torch.Tensor) else torch.as_tensor(emb.embeddings)
                for emb in msg.input_embeddings
            ] for msg in messages]
            input_embedding_ranges = [
                torch.tensor([[emb.start, emb.end] for emb in msg.input_embeddings]) for msg in messages
            ]
            input_embedding_indexing = torch.zeros((batch_size, max_q_seq_length), dtype=torch.bool)
            for msg_id, msg in enumerate(messages):
                num_history_ids = msg.num_history_ids
                for emb in msg.input_embeddings:
                    # make slice index relative to embeddings
                    emb_start = emb.start - num_history_ids
                    emb_end = emb.end - num_history_ids
                    input_embedding_indexing[msg_id][emb_start:emb_end] = True
            return (input_embeddings, input_embedding_indexing, input_embedding_ranges)

        def __has_values(input_multimodals):
            for input_mm in input_multimodals:
                for val in input_mm.values():
                    if len(val) > 0:
                        return True
            return False

        has_embedding = any([len(msg.history_embeddings) > 0 for msg in messages])
        if has_embedding:
            has_embedding = any([len(msg.input_embeddings) > 0 for msg in messages])

        has_multimodal = any([not msg.history_multimodals.empty() for msg in messages])
        input_multimodals = None
        if has_multimodal:
            input_multimodals = [msg.get_input_multimodals() for msg in messages]
            has_multimodal = __has_values(input_multimodals)
            if not has_multimodal:
                # no multimodal inputs
                input_multimodals = None

        if not has_embedding and not has_multimodal:
            # no vision inputs
            return None

        if has_embedding:
            # for inputs with embeddings
            (input_embeddings, input_embedding_indexing, input_embedding_ranges) = __get_vlm_embeddings()
        else:
            input_embeddings = None
            input_embedding_indexing = None
            input_embedding_ranges = None

        history_lengths = model_inputs.history_lengths
        vision_embedding_inputs = VisionModelInputs(history_lengths=history_lengths,
                                                    input_embeddings=input_embeddings,
                                                    input_embedding_indexing=input_embedding_indexing,
                                                    input_embedding_ranges=input_embedding_ranges,
                                                    input_multimodals=input_multimodals)
        return vision_embedding_inputs

    @property
    def torch_int_dtype(self):
        """Return int32 for cuda, int64 for others."""
        if self.executor.device_type == 'cuda':
            return torch.int32
        return torch.int64

    def _set_adapter_ids(self, model_inputs: ModelInputs, messages: 'SeqList'):
        """Set adapter ids to model inputs."""
        if self.adapter_manager.num_adapters() <= 1:
            return
        adapter_names = [msg.adapter_name for msg in messages]
        local_adapter_ids = self.adapter_manager.get_adapter_ids(adapter_names)
        local_adapter_ids = model_inputs.seq_length.new_tensor(local_adapter_ids)
        model_inputs.local_adapter_ids = local_adapter_ids

    @torch.inference_mode()
    @record_function('create_model_inputs')
    def create_model_inputs(self, messages: 'SeqList', is_prefill: bool):
        """Create model inputs from messages.

        Args:
            messages (SeqList): The input messages.
        """
        batch_size = len(messages)
        # history lengths
        history_lengths = torch.tensor([msg.num_history_ids for msg in messages])

        # input ids
        token_ids = [msg.token_ids for msg in messages]

        input_ids = torch.as_tensor(np.concatenate(token_ids))[None]

        # seqlens
        is_decoding = not is_prefill
        if not is_decoding:
            seq_length = [len(tokens) for tokens in token_ids]
            seq_length = torch.tensor(seq_length, dtype=torch.long)
            max_q_seqlen = seq_length.max().item()
        else:
            max_q_seqlen = len(token_ids[0])
            seq_length = torch.full((batch_size, ), max_q_seqlen, dtype=torch.long)
        kv_seqlens = seq_length + history_lengths
        max_kv_seqlen = kv_seqlens.max().item()
        sum_kv_seqlen = kv_seqlens.sum().item()

        # block offsets
        block_offsets = self.scheduler.get_block_tables(messages)
        block_offsets = _tensorlize_block_offsets(block_offsets, dtype=self.torch_int_dtype)

        # num_ignored_history
        num_ignored_history = torch.tensor([msg.num_ignored_history for msg in messages])

        # model_metas
        model_metas = [msg.model_meta for msg in messages]

        # create model inputs for all required fields
        model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=seq_length,
            history_lengths=history_lengths,
            block_offsets=block_offsets,
            is_decoding=is_decoding,
            num_ignored_history=num_ignored_history,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            model_metas=model_metas,
        )

        # adapters
        self._set_adapter_ids(model_inputs, messages)

        # vision inputs
        vision_model_inputs = self._create_vision_model_inputs(messages, model_inputs)
        model_inputs.vision_inputs = vision_model_inputs

        # ssm
        if self.config.is_ssm:
            state_offsets = torch.tensor([msg.logical_state for msg in messages])
            model_inputs.state_offsets = state_offsets

        return model_inputs

    @torch.inference_mode()
    @record_function('create_model_inputs_long_context')
    def create_model_inputs_long_context(self,
                                         seq: 'SchedulerSequence',
                                         chunk_size: int,
                                         multimodals: Optional['MultiModalInputs'] = None):
        """Create model inputs for long context messages."""
        token_ids = seq.token_ids[:chunk_size]
        input_ids = torch.as_tensor(token_ids)[None]
        q_seqlens = torch.tensor([chunk_size])
        history_lens = torch.tensor([seq.num_history_ids])

        # block offsets
        block_offsets = self.scheduler.get_block_tables([seq])
        block_offsets = torch.as_tensor(block_offsets[0], dtype=self.torch_int_dtype)[None]

        # num_ignored_history
        num_ignored_history = torch.tensor([seq.num_ignored_history])

        # model_metas
        model_metas = [seq.model_meta]

        kv_seqlens = q_seqlens + history_lens
        max_kv_seqlen = kv_seqlens.item()
        sum_kv_seqlen = max_kv_seqlen

        model_inputs = ModelInputs(
            input_ids=input_ids,
            seq_length=q_seqlens,
            history_lengths=history_lens,
            block_offsets=block_offsets,
            is_decoding=False,
            num_ignored_history=num_ignored_history,
            max_q_seqlen=q_seqlens.item(),
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            model_metas=model_metas,
            is_chunk=True,
        )

        # adapters
        self._set_adapter_ids(model_inputs, [seq])

        # vision inputs
        if multimodals is not None and len(multimodals) > 0:
            vision_model_inputs = VisionModelInputs(
                history_lengths=model_inputs.history_lengths,
                input_multimodals=[multimodals],
            )
            model_inputs.vision_inputs = vision_model_inputs

        # ssm
        if self.config.is_ssm:
            model_inputs.state_offsets = torch.tensor([seq.logical_state])

        return model_inputs

    @torch.inference_mode()
    @record_function('create_model_inputs_delta')
    def create_model_inputs_delta(self):
        """Create model inputs delta from messages."""
        batch_size = len(self.running_seqs)
        assert batch_size > 0
        num_decode_tokens = self.engine_strategy.get_num_decode_tokens()
        max_q_seqlen = num_decode_tokens
        prealloc_size = self.engine_strategy.get_prealloc_size(True)
        valid_mask = self.scheduler.schedule_running(self.running_seqs,
                                                     num_decode_tokens=num_decode_tokens,
                                                     prealloc_size=prealloc_size)

        valid_mask = np.array(valid_mask)
        indices_cpu = np.arange(0, batch_size)[valid_mask]
        valid_seqs: List['SchedulerSequence'] = [self.running_seqs[i] for i in indices_cpu]
        invalid_seqs: List['SchedulerSequence'] = [self.running_seqs[i] for i in range(batch_size) if not valid_mask[i]]
        if len(valid_seqs) == 0:
            return None, valid_seqs, invalid_seqs

        # block offsets
        block_offsets = self.scheduler.get_block_tables(valid_seqs)
        block_offsets = _tensorlize_block_offsets(block_offsets, dtype=self.torch_int_dtype)

        # sliding window
        if self.scheduler.cache_config.window_size > 0:
            num_ignored_history = torch.tensor([msg.num_ignored_history for msg in valid_seqs])
        else:
            num_ignored_history = None

        kv_seqlens = [seq.num_all_ids + max_q_seqlen for seq in valid_seqs]
        sum_kv_seqlen = sum(kv_seqlens) + batch_size * max_q_seqlen
        max_kv_seqlen = max(kv_seqlens) + max_q_seqlen

        output = ModelInputsDelta(
            indices=None,
            block_offsets=block_offsets,
            indice_cpu=indices_cpu,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            num_ignored_history=num_ignored_history,
        )

        return output, valid_seqs, invalid_seqs

    def create_model_inputs_delta_valid_only(self):
        """Create model inputs delta for valid running seqs only.

        Only check validation, no resources will be scheduled.
        """
        from lmdeploy.pytorch.messages import MessageStatus
        batch_size = len(self.running_seqs)

        valid_mask = [seq.status == MessageStatus.RUNNING for seq in self.running_seqs]
        if all(valid_mask):
            return None, self.running_seqs, []

        valid_mask = np.array(valid_mask, dtype=bool)
        indices_cpu = np.arange(0, batch_size)[valid_mask]
        valid_seqs: List['SchedulerSequence'] = [self.running_seqs[i] for i in indices_cpu]
        invalid_seqs: List['SchedulerSequence'] = [self.running_seqs[i] for i in range(batch_size) if not valid_mask[i]]

        num_decode_tokens = self.engine_strategy.get_num_decode_tokens()
        max_q_seqlen = num_decode_tokens
        kv_seqlens = [seq.num_all_ids + max_q_seqlen for seq in valid_seqs]
        if len(kv_seqlens) == 0:
            sum_kv_seqlen = 0
            max_kv_seqlen = 0
        else:
            sum_kv_seqlen = sum(kv_seqlens) + batch_size * max_q_seqlen
            max_kv_seqlen = max(kv_seqlens) + max_q_seqlen

        output = ModelInputsDelta(
            indices=None,
            block_offsets=None,
            indice_cpu=indices_cpu,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            num_ignored_history=None,
        )

        return output, valid_seqs, invalid_seqs

    def update_running_seqs(self, running: 'SeqList', inputs: Optional[ModelInputs]):
        """Update running seqs."""
        if self.config.role == EngineRole.Prefill:
            # p node will not update running seqs
            return

        is_decoding = inputs is None
        if self.long_context_chunker.enabled() and not is_decoding:
            # long context chunk does not need to update running seqs
            self.long_context_chunker.update_step(inputs)
            return

        if is_decoding:
            self.running_seqs = running
        else:
            self.running_seqs += running

    def deactivate_evict_seqs(self):
        """Deactivate and evict seqs."""
        scheduler = self.scheduler
        to_evict_seqs = self.to_evict_seqs
        if len(to_evict_seqs) == 0:
            return
        # deactivate seqs(running -> ready)
        scheduler.deactivate_seqs(to_evict_seqs)
        # ready to waiting
        scheduler.evict_seqs(to_evict_seqs)
        self.to_evict_seqs.clear()

    @torch.inference_mode()
    @record_function('make_forward_inputs')
    def _make_forward_inputs(self, prefill: bool, enable_empty: bool = False):
        """Make forward inputs for ModelAgent._async_step_background()"""

        def __need_logits(seqs: 'SeqList'):
            """Need logits."""
            if self.spec_decoding:
                return True
            return any(seq.return_logits for seq in seqs)

        def __need_routed_experts(seqs: 'SeqList'):
            """Need routed experts."""
            return any(seq.return_routed_experts for seq in seqs)

        def __create_model_inputs(seqs):
            """Createe model inputs."""
            inputs = self.create_model_inputs(seqs, True)
            delta, valid_seqs, _ = self.create_model_inputs_delta_valid_only()
            self.running_seqs = valid_seqs
            extra_inputs = self.model_agent_strategy.make_extra_inputs(seqs, inputs)
            return inputs, delta, extra_inputs

        def __create_inputs_chunk(running: 'SeqList'):
            chunk_size, multimodals = self.long_context_chunker.next_chunk_size()
            inputs = self.create_model_inputs_long_context(running[0], chunk_size, multimodals)
            extra_inputs = self.model_agent_strategy.make_extra_inputs(running, inputs)
            return inputs, extra_inputs

        def __create_inputs_long_context_chunk():
            seq = self.long_context_chunker.seq
            running = [seq]
            if self.long_context_chunker.is_last_chunk():
                inputs, delta, extra_inputs = __create_model_inputs(running)
                self.long_context_chunker.clear()
            else:
                inputs, extra_inputs = __create_inputs_chunk(running)
                delta = None
            inputs.is_first_chunk = False
            return running, inputs, delta, extra_inputs

        def __create_inputs_prefill():
            if self.config.role == EngineRole.Prefill:
                prealloc_size = 0
            else:
                prealloc_size = self.engine_strategy.get_prealloc_size(True)
            scheduler_output = scheduler.schedule(is_prefill=prefill, prealloc_size=prealloc_size)
            running = scheduler_output.running
            swap_in_map = scheduler_output.swap_in_map
            swap_out_map = scheduler_output.swap_out_map

            inputs = None
            delta = None
            extra_inputs = None
            if len(running) == 1 and self.long_context_chunker.is_long_context(running[0]):
                # set long context chunker
                self.long_context_chunker.set_seq(running[0])
                inputs, extra_inputs = __create_inputs_chunk(running)
            elif len(running) > 0:
                # create inputs
                inputs, delta, extra_inputs = __create_model_inputs(running)
            return running, inputs, delta, extra_inputs, swap_in_map, swap_out_map

        scheduler = self.scheduler
        logger.debug(f'Make forward inputs with prefill={prefill}, enable_empty={enable_empty}')

        inputs = None
        delta = None
        swap_in_map = {}
        swap_out_map = {}

        self.long_context_chunker.check_enable()
        if self.long_context_chunker.enabled():
            # long context chunking
            running, inputs, delta, extra_inputs = __create_inputs_long_context_chunk()
        elif prefill:
            # prefill
            (
                running,
                inputs,
                delta,
                extra_inputs,
                swap_in_map,
                swap_out_map,
            ) = __create_inputs_prefill()

        # try decoding
        if inputs is None and len(self.running_seqs) > 0 and self.config.role != EngineRole.Prefill:
            prefill = False
            delta, running, invalid_seqs = self.create_model_inputs_delta()
            self.to_evict_seqs = invalid_seqs
            extra_inputs = None

        # skip if enable empty
        if inputs is None and delta is None:
            return None

        sampling_inputs = self.sampling_strategy.make_sampling_inputs(running)
        if inputs is not None:
            stopping_criteria = self.model_agent_strategy.make_stopping_criteria(running)
        else:
            stopping_criteria = None

        return_logits = __need_logits(running)
        return_routed_experts = __need_routed_experts(running)

        return dict(
            running=running,
            inputs=inputs,
            delta=delta,
            swap_in_map=swap_in_map,
            swap_out_map=swap_out_map,
            sampling_inputs=sampling_inputs,
            stopping_criteria=stopping_criteria,
            return_logits=return_logits,
            extra_inputs=extra_inputs,
            return_routed_experts=return_routed_experts,
        )

    def do_prefill_pnode(self):
        return True

    def do_prefill_default(self):
        # decoding if no waiting
        scheduler = self.scheduler

        # do decoding if not waiting
        if not scheduler.has_waiting():
            return False

        # do prefill if too much tokens
        waiting = scheduler.waiting
        token_count = 0
        for seq in waiting:
            token_count += seq.num_token_ids
            if token_count >= self.config.max_prefill_token_num:
                return True

        # prefill if no enough running
        num_ready = scheduler.num_ready()
        num_running = scheduler.num_running()
        max_batches = self.config.max_batches
        if num_ready + num_running < max_batches * 0.5:
            return True

        # decoding
        return False

    def do_prefill_chunked(self):
        """Chunked prefill strategy.

        both dp=1 and dp>1 are supported.
        """
        scheduler = self.scheduler
        return not scheduler.has_ready()

    async def _send_next_inputs_impl(self, prefill: bool = None, enable_empty: bool = False):
        forward_inputs = self._make_forward_inputs(prefill, enable_empty)
        if forward_inputs is None:
            return None, None
        next_running = forward_inputs.pop('running')
        inputs = forward_inputs['inputs']
        if logger.level <= logging.DEBUG and inputs is not None:
            logger.debug(f'Sending forward inputs: {inputs.log_info()}')
            session_ids = [seq.session_id for seq in next_running]
            logger.debug(f'Forward session_ids: {session_ids}')
        await self.executor.forward_async(forward_inputs)
        self.forward_inputs = forward_inputs
        return forward_inputs, next_running

    async def send_next_inputs(self):
        prefill = self.do_prefill()
        return await self._send_next_inputs_impl(prefill)

    async def prefetch_next_inputs(self):
        prefill = self.do_prefill()
        # send next forward
        logger.debug('Prefetching next forward inputs.')
        return await self._send_next_inputs_impl(prefill, True)


def build_inputs_maker(engine: 'Engine'):
    """Build inputs makers."""
    config = InputsMakerConfig.from_engine(engine)
    return InputsMakerAsync(
        executor=engine.executor,
        scheduler=engine.scheduler,
        adapter_manager=engine.adapter_manager,
        engine_strategy=engine.engine_strategy,
        sampling_strategy=engine.sampling_strategy,
        model_agent_strategy=engine.model_agent_strategy,
        config=config,
    )
