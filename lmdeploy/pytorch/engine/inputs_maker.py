# Copyright (c) OpenMMLab. All rights reserved.
"""Engine-loop input construction for the LMDeploy PyTorch backend.

This module converts scheduler decisions into model-agent inputs.  Most helpers
build tensor fields for full-batch ``ModelInputs``; ``InputsMakerAsync`` is the
coordinator that chooses prefill/chunk/decode work, attaches per-forward
metadata, dispatches it to the executor, and updates local running state.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import envs as _envs
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


def _compact_state_prefix_cache_restore_offsets(messages: list['SchedulerSequence']):
    """Build compact SSM restore src/dst index tensors."""
    src_offsets = []
    dst_offsets = []
    for msg in messages:
        state_idx = msg.prefix_cache.restore_state
        if state_idx >= 0:
            src_offsets.append(state_idx)
            dst_offsets.append(msg.logical_state)
    if len(src_offsets) == 0:
        return None, None
    return tuple(src_offsets), tuple(dst_offsets)


def _compact_state_prefix_cache_save_offsets(messages: list['SchedulerSequence'], save_state_offsets: list[int]):
    """Build compact SSM save src/dst index tensors."""
    src_offsets = []
    dst_offsets = []
    for msg, state_idx in zip(messages, save_state_offsets):
        if state_idx >= 0:
            src_offsets.append(msg.logical_state)
            dst_offsets.append(state_idx)
    if len(src_offsets) == 0:
        return None, None
    return tuple(src_offsets), tuple(dst_offsets)


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
    use_mrope: bool = False
    prefill_interval: int = 16

    @staticmethod
    def from_engine(engine: 'Engine'):
        cache_config = engine.cache_config
        model_config = engine.model_config
        prefill_interval = engine.engine_config.prefill_interval
        kwargs = dict()
        if prefill_interval is not None:
            if not isinstance(prefill_interval, int) or prefill_interval <= 0:
                raise ValueError('engine.engine_config.prefill_interval must be a positive int '
                                f'or None, but got {prefill_interval!r}')
            kwargs['prefill_interval'] = prefill_interval
        return InputsMakerConfig(
            spec_decoding=engine.specdecode_config is not None,
            max_batches=cache_config.max_batches,
            max_prefill_token_num=cache_config.max_prefill_token_num,
            role=cache_config.role,
            is_ssm=len(cache_config.states_shapes) > 0,
            dp=engine.dist_config.dp,
            enable_chunked_prefill=engine.misc_config.enable_chunked_prefill,
            use_mrope=model_config.use_mrope,
            **kwargs,
        )


class LongContextChunker:
    """Split a single long prefill into model-safe chunks.

    Multimodal spans are indivisible, so a span larger than
    ``max_prefill_token_num`` temporarily raises the chunk limit.  Prefix-cache
    restore can skip over the span itself, but the enlarged limit still needs
    to be derived from the whole request history so the remaining text tail is
    chunked the same way as the no-cache path.
    """

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
        """Set the sequence currently being chunked."""
        self.seq = seq
        self.next_step = seq.num_history_ids

        max_prefill_num = self.max_prefill_token_num
        input_mm = seq.get_input_multimodals()
        mm_for_chunk_limit = seq.get_chunk_limit_multimodals()
        self.multimodals = defaultdict(list)

        for value in mm_for_chunk_limit.values():
            max_mm_size = max([v.end - v.start for v in value], default=0)
            max_prefill_num = max(max_prefill_num, max_mm_size)

        has_multimodal = False
        for key, value in input_mm.items():
            # Only remaining multimodals are emitted by next_chunk_size().
            value = sorted(value, key=lambda x: x.start)
            self.multimodals[key] = value

            has_multimodal = has_multimodal or len(value) > 0

        self.max_prefill_num = max_prefill_num
        self.has_multimodal = has_multimodal

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
        """Get the next chunk size and its remaining multimodal payloads."""
        seq = self.seq
        if seq is None:
            return 0, None

        llm_chunk_size = min(seq.num_token_ids, self.max_prefill_num)

        if len(self.multimodals) == 0:
            # no vlm inputs found
            return llm_chunk_size, None

        start = seq.num_history_ids
        end = start + llm_chunk_size
        out_multimodals: MultiModalInputs = defaultdict(list)
        for modal_type, mm in self.multimodal_iter():
            assert mm.start >= start, 'multimodal data should be sorted by start'
            if mm.start >= end:
                # | start ... end ... mm.start ... mm.end |
                # if start is beyond threshold, stop
                break

            if mm.end > end:
                # | start ... mm.start ... end ... mm.end |
                # Do not split a multimodal span; recompute from its start in
                # the next chunk instead.
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
        self.seq: SchedulerSequence = None
        self.multimodals: MultiModalInputs = defaultdict(list)
        self.next_step: int = 0
        self.max_prefill_num: int = self.max_prefill_token_num
        self.has_multimodal = False

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
            # A stopped long request no longer has a valid continuation.  We do
            # not send a cleanup-only worker forward here: normal prefill/decode
            # ignore chunk carry, and the next first chunk resets carry before
            # use.  Avoiding a no-work forward also keeps DP ranks aligned.
            self.clear()


class InputsMakerAsync:
    """Coordinate prefill, decode, and long-context input dispatch.

    ``Scheduler`` owns admission, ordering, and cache/KV resources.  This class
    consumes the scheduler result and builds tensors only after resources have
    been granted.  Prefill-like work is represented by full ``ModelInputs``:
    prompt prefill, final long-context chunks, and eager non-final long chunks.
    Decode is represented by ``ModelInputsDelta`` and reuses persistent
    model-agent/strategy ``StepInputs`` that were created by earlier prefill and
    decode forwards.

    ``running_seqs`` is local engine-loop state, not the scheduler's source of
    truth.  It tracks sequences already sent to the executor so this class can
    build decode deltas, evict invalid decode requests, and update the local
    view after outputs return.  Every dispatched forward also carries the
    strategy-specific ``extra_inputs``, sampling inputs, and stopping criteria
    expected by the model agent.

    Long-context chunking is coordinated here because it spans scheduling
    policy and input construction.  ``LongContextChunker`` tracks one active
    long prefill and selects model-safe chunk boundaries, including indivisible
    multimodal spans.  Before tensors are created for each chunk, the scheduler
    reserves the chunk's KV ownership.  Non-final chunks are eager chunk
    forwards with no user-visible output; the final chunk is treated as normal
    prefill so it can merge into persistent decode state.

    The current first-slice chunked-prefill policy intentionally uses separate
    forwards instead of one mixed decode+prefill tensor batch.  After a
    non-final chunk, runnable decode is preferred and remains on the existing
    delta/CUDAGraph path; at most one eager non-final long chunk is sent after
    decode gets a chance to run.  Preserve chunk flags such as
    ``is_chunk_multimodal`` and ``is_last_chunk`` because VLM and speculative
    decoding paths interpret them downstream.
    """

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
        self.cache_config = scheduler.cache_config
        self.kernel_blocks_per_kv = self.cache_config.block_size // self.cache_config.kernel_block_size
        self.kernel_block_arange = torch.arange(self.kernel_blocks_per_kv, dtype=self.torch_int_dtype)

        # strategies
        self.engine_strategy = engine_strategy
        self.sampling_strategy = sampling_strategy
        self.model_agent_strategy = model_agent_strategy

        self._init_do_prefill(config)

        self._short_prefill_turns_per_long_chunk = max(1, _envs.opt_ttft_short_turns)
        self._init_runtime_state()

    def _init_runtime_state(self):
        """Initialize request-local scheduling state."""
        self._decode_count = 0
        self._last_forward_kind = None
        self._short_prefill_turns_since_long_chunk = 0
        self.next_is_prefill = True
        self.forward_inputs = None
        self.running_seqs: list[SchedulerSequence] = []
        self.to_evict_seqs: list[SchedulerSequence] = []
        self.long_context_chunker = LongContextChunker(self.config.max_prefill_token_num)

    def reset_runtime_state(self):
        """Discard request-local scheduling state after sleep cancels sessions."""
        self._decode_count = 0
        self._last_forward_kind = None
        self._short_prefill_turns_since_long_chunk = 0
        self.next_is_prefill = True
        self.forward_inputs = None
        self.running_seqs = []
        self.to_evict_seqs.clear()
        self.long_context_chunker.clear()

    def _init_do_prefill(self, config: InputsMakerConfig):
        if config.role == EngineRole.Prefill:
            self.do_prefill = self.do_prefill_pnode
        elif config.enable_chunked_prefill:
            self.do_prefill = self.do_prefill_chunked
        else:
            self.do_prefill = self.do_prefill_default

    def _has_pending_last_long_context_chunk(self):
        """Check whether a running long context has only its final chunk
        left."""
        return self.long_context_chunker.enabled() and self.long_context_chunker.is_last_chunk()

    def has_pending_long_context_chunk(self):
        """Check whether engine-local long-context chunk work can run."""
        self.long_context_chunker.check_enable()
        return self.long_context_chunker.enabled()

    def _should_defer_long_context_chunk(self, prefill: bool):
        """Check whether the active long-context chunk should yield this
        loop."""
        if self.config.role == EngineRole.Prefill:
            return False
        if not self.long_context_chunker.enabled():
            return False
        if self.long_context_chunker.is_last_chunk():
            if len(self.running_seqs) == 0:
                return False
            return not prefill
        return getattr(self, '_last_forward_kind', None) == 'long_context_chunk'

    def _is_long_context_chunk_turn_due(self):
        """Check if active long chunk should run before another short
        prefill."""
        return self._short_prefill_turns_since_long_chunk >= self._short_prefill_turns_per_long_chunk

    def _forward_kind(self, inputs: 'ModelInputs|None', delta: 'ModelInputsDelta|None'):
        """Classify a queued forward for long-context interleaving policy."""
        if inputs is None:
            if delta is not None:
                return 'decode'
            return None
        if inputs.is_chunk and not inputs.is_last_chunk:
            return 'long_context_chunk'
        if inputs.is_chunk:
            return 'last_long_context_chunk'
        if inputs.is_decoding:
            return 'decode'
        return 'prefill'

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

    def _map_to_kernel_block_offsets(self, block_offsets: torch.Tensor):
        """Converts manager block_offsets to kernel block_offsets.

        Example:

            # block_manager block size: 32 tokens,
            # Kernel block size: 16 tokens
            # kernel_blocks_per_kv = 2
            >>> block_manager block offsets = [0, 1, 3]
            >>> Result kernel block offsets = [0, 1, 2, 3, 6, 7]

            # Each block_manager block id maps to 2 kernel block id:
            # block_manager block id 0 -> kernel block id [0, 1]
            # block_manager block id 1 -> kernel block id [2, 3]
            # block_manager block id 3 -> kernel block id [6, 7]
        """
        if self.kernel_blocks_per_kv == 1:
            return block_offsets
        batch_size = block_offsets.shape[0]
        block_offsets = (block_offsets[:, :, None] * self.kernel_blocks_per_kv +
                         self.kernel_block_arange[None, None, :]).reshape(batch_size, -1)
        return block_offsets

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
        block_offsets = self._map_to_kernel_block_offsets(block_offsets)

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
            if (self.cache_config.enable_prefix_caching
                    and any(msg.prefix_cache.restore_state >= 0 for msg in messages)):
                # Pin restore checkpoints while the forward copies them into
                # runtime state slots; otherwise checkpoint eviction could race
                # with input prefetching for the next batch.
                self.scheduler.block_trie.acquire_state_checkpoint_restores(messages)
                if any(msg.prefix_cache.restore_state >= 0 and not msg.prefix_cache.restore_state_acquired
                       for msg in messages):
                    raise RuntimeError('Failed to acquire SSM prefix-cache restore checkpoint.')
                restore_src_offsets, restore_dst_offsets = _compact_state_prefix_cache_restore_offsets(messages)
                model_inputs.state_prefix_cache_offsets = restore_src_offsets
                model_inputs.state_prefix_cache_dst_offsets = restore_dst_offsets
            if self.cache_config.enable_prefix_caching and not is_decoding:
                # Prefill saves publish only after model_forward has copied the
                # runtime state to these reserved checkpoint offsets.
                save_state_offsets = [
                    self.scheduler.block_trie.reserve_state_checkpoint_for_seq(msg) for msg in messages
                ]
                save_src_offsets, save_dst_offsets = _compact_state_prefix_cache_save_offsets(messages,
                                                                                              save_state_offsets)
                model_inputs.state_prefix_cache_save_src_offsets = save_src_offsets
                model_inputs.state_prefix_cache_save_offsets = save_dst_offsets

        if self.config.use_mrope:
            mrope_pos_ids = [msg.mrope_pos_ids for msg in messages]
            mrope_pos_ids = torch.as_tensor(np.concatenate(mrope_pos_ids)).T
            model_inputs.mrope_pos_ids = mrope_pos_ids

        return model_inputs

    @torch.inference_mode()
    @record_function('create_model_inputs_long_context')
    def create_model_inputs_long_context(self,
                                         seq: 'SchedulerSequence',
                                         chunk_size: int,
                                         multimodals: 'MultiModalInputs|None' = None):
        """Create model inputs for long context messages."""
        token_ids = seq.token_ids[:chunk_size]
        input_ids = torch.as_tensor(token_ids)[None]
        q_seqlens = torch.tensor([chunk_size])
        history_lens = torch.tensor([seq.num_history_ids])

        # block offsets
        block_offsets = self.scheduler.get_block_tables([seq])
        block_offsets = torch.as_tensor(block_offsets[0], dtype=self.torch_int_dtype)[None]
        block_offsets = self._map_to_kernel_block_offsets(block_offsets)

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
            if self.cache_config.enable_prefix_caching and seq.prefix_cache.restore_state >= 0:
                # Long-context chunks use the same restore pinning contract as
                # normal prefill batches.
                self.scheduler.block_trie.acquire_state_checkpoint_restore_for_seq(seq)
                if not seq.prefix_cache.restore_state_acquired:
                    raise RuntimeError('Failed to acquire SSM prefix-cache restore checkpoint.')
                model_inputs.state_prefix_cache_offsets = (seq.prefix_cache.restore_state, )
                model_inputs.state_prefix_cache_dst_offsets = (seq.logical_state, )
            if self.cache_config.enable_prefix_caching:
                # Save at the exact state step produced by this chunk forward.
                checkpoint_step = seq.num_history_ids + chunk_size
                save_state = self.scheduler.block_trie.reserve_state_checkpoint_for_seq(seq, step=checkpoint_step)
                if save_state >= 0:
                    model_inputs.state_prefix_cache_save_src_offsets = (seq.logical_state, )
                    model_inputs.state_prefix_cache_save_offsets = (save_state, )

        # mrope
        if self.config.use_mrope:
            mrope_pos_ids = seq.mrope_pos_ids[:chunk_size]
            mrope_pos_ids = torch.as_tensor(mrope_pos_ids).T
            model_inputs.mrope_pos_ids = mrope_pos_ids

        return model_inputs

    @torch.inference_mode()
    @record_function('create_model_inputs_delta')
    def create_model_inputs_delta(self):
        """Create model inputs delta from messages."""
        batch_size = len(self.running_seqs)
        assert batch_size > 0
        num_decode_tokens = self.engine_strategy.get_num_decode_tokens()
        num_required_tokens = self.engine_strategy.get_num_required_tokens()
        max_q_seqlen = num_decode_tokens
        prealloc_size = self.engine_strategy.get_prealloc_size(True)
        valid_mask = self.scheduler.schedule_running(self.running_seqs,
                                                     num_required_tokens=num_required_tokens,
                                                     prealloc_size=prealloc_size)

        valid_mask = np.array(valid_mask)
        indices_cpu = np.arange(0, batch_size)[valid_mask]
        valid_seqs: list[SchedulerSequence] = [self.running_seqs[i] for i in indices_cpu]
        invalid_seqs: list[SchedulerSequence] = [self.running_seqs[i] for i in range(batch_size) if not valid_mask[i]]
        if len(valid_seqs) == 0:
            return None, valid_seqs, invalid_seqs

        # block offsets
        block_offsets = self.scheduler.get_block_tables(valid_seqs)
        block_offsets = _tensorlize_block_offsets(block_offsets, dtype=self.torch_int_dtype)
        block_offsets = self._map_to_kernel_block_offsets(block_offsets)

        # sliding window
        if self.scheduler.cache_config.window_size > 0:
            num_ignored_history = torch.tensor([msg.num_ignored_history for msg in valid_seqs])
        else:
            num_ignored_history = torch.zeros(len(valid_seqs), dtype=torch.long)

        # num_all_ids can be one decode step stale here: EngineLoop prefetches
        # the next inputs before _finish_forward_output() advances the sequence,
        # so +max_q_seqlen recovers this forward's kv length. The bug was adding
        # max_q_seqlen AGAIN in the reductions, plus using batch_size (which
        # counts scheduler-dropped invalid seqs) instead of reducing over the
        # valid seqs only (#4024).
        kv_seqlens = [seq.num_all_ids + max_q_seqlen for seq in valid_seqs]
        sum_kv_seqlen = sum(kv_seqlens)
        max_kv_seqlen = max(kv_seqlens)

        output = ModelInputsDelta(
            indices=None,
            block_offsets=block_offsets,
            indice_cpu=indices_cpu,
            max_q_seqlen=max_q_seqlen,
            max_kv_seqlen=max_kv_seqlen,
            sum_kv_seqlen=sum_kv_seqlen,
            num_ignored_history=num_ignored_history,
        )
        decode_state_interval = self.cache_config.prefix_cache_decode_state_interval
        if (self.cache_config.enable_prefix_caching and self.config.is_ssm and decode_state_interval > 0
                and not self.spec_decoding and num_decode_tokens == 1):
            save_state_offsets = [
                self.scheduler.block_trie.reserve_decode_state_checkpoint_for_seq(seq, decode_state_interval)
                for seq in valid_seqs
            ]
            if any(state_idx >= 0 for state_idx in save_state_offsets):
                save_src_offsets, save_dst_offsets = _compact_state_prefix_cache_save_offsets(valid_seqs,
                                                                                              save_state_offsets)
                output.state_prefix_cache_save_src_offsets = save_src_offsets
                output.state_prefix_cache_save_offsets = save_dst_offsets

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
        valid_seqs: list[SchedulerSequence] = [self.running_seqs[i] for i in indices_cpu]
        invalid_seqs: list[SchedulerSequence] = [self.running_seqs[i] for i in range(batch_size) if not valid_mask[i]]

        num_decode_tokens = self.engine_strategy.get_num_decode_tokens()
        max_q_seqlen = num_decode_tokens
        # Keep +max_q_seqlen (num_all_ids may be one decode step stale), but do
        # not add it a second time in the reductions or use batch_size (#4024).
        kv_seqlens = [seq.num_all_ids + max_q_seqlen for seq in valid_seqs]
        if len(kv_seqlens) == 0:
            sum_kv_seqlen = 0
            max_kv_seqlen = 0
        else:
            sum_kv_seqlen = sum(kv_seqlens)
            max_kv_seqlen = max(kv_seqlens)

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

    def update_running_seqs(self, running: 'SeqList', inputs: 'ModelInputs|None'):
        """Update running seqs."""
        if self.config.role == EngineRole.Prefill:
            # p node will not update running seqs
            return

        is_decoding = inputs is None
        if self.long_context_chunker.enabled() and not is_decoding and inputs.is_chunk:
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

        def __need_ce_loss(seqs: 'SeqList'):
            """Need input cross-entropy loss."""
            return any(seq.return_ce_loss for seq in seqs)

        def __create_model_inputs(seqs):
            """Createe model inputs."""
            inputs = self.create_model_inputs(seqs, True)
            delta, valid_seqs, _ = self.create_model_inputs_delta_valid_only()
            self.running_seqs = valid_seqs
            extra_inputs = self.model_agent_strategy.make_extra_inputs(seqs, inputs)
            return inputs, delta, extra_inputs

        def __create_inputs_chunk(running: 'SeqList', chunk_size: int, multimodals: 'MultiModalInputs|None'):
            inputs = self.create_model_inputs_long_context(running[0], chunk_size, multimodals)
            extra_inputs = self.model_agent_strategy.make_extra_inputs(running, inputs)
            return inputs, extra_inputs

        def __reserve_long_context_chunk(seq: 'SchedulerSequence', chunk_size: int, is_last_chunk: bool):
            if self.config.role == EngineRole.Prefill:
                prealloc_size = 0
            elif is_last_chunk:
                prealloc_size = self.engine_strategy.get_prealloc_size(True)
            else:
                prealloc_size = 0
            return scheduler.reserve_long_context_chunk(seq,
                                                        chunk_size,
                                                        prealloc_size=prealloc_size,
                                                        is_last_chunk=is_last_chunk)

        def __create_inputs_long_context_chunk():
            seq = self.long_context_chunker.seq
            chunk_size, multimodals = self.long_context_chunker.next_chunk_size()
            is_last_chunk = self.long_context_chunker.is_last_chunk()
            is_chunk_multimodal = self.long_context_chunker.has_multimodal
            if not __reserve_long_context_chunk(seq, chunk_size, is_last_chunk):
                return [], None, None, None
            running = [seq]
            if is_last_chunk:
                inputs, delta, extra_inputs = __create_model_inputs(running)
                inputs.is_chunk = True
                inputs.is_last_chunk = True
                self.long_context_chunker.clear()
            else:
                inputs, extra_inputs = __create_inputs_chunk(running, chunk_size, multimodals)
                delta = None
            inputs.is_first_chunk = False
            inputs.is_chunk_multimodal = is_chunk_multimodal
            self._short_prefill_turns_since_long_chunk = 0
            return running, inputs, delta, extra_inputs

        def __create_inputs_prefill(allow_long_prefill: bool = True, prefer_long_prefill: bool = False):
            if self.config.role == EngineRole.Prefill:
                prealloc_size = 0
            else:
                prealloc_size = self.engine_strategy.get_prealloc_size(True)
            scheduler_output = scheduler.schedule(is_prefill=True,
                                                  prealloc_size=prealloc_size,
                                                  allow_long_prefill=allow_long_prefill,
                                                  prefer_long_prefill=prefer_long_prefill)
            running = scheduler_output.running
            swap_in_map = scheduler_output.swap_in_map
            swap_out_map = scheduler_output.swap_out_map

            inputs = None
            delta = None
            extra_inputs = None
            if len(running) == 1 and self.long_context_chunker.is_long_context(running[0]):
                # set long context chunker
                self.long_context_chunker.set_seq(running[0])
                if self.long_context_chunker.is_last_chunk():
                    # A prefix-cache restore can skip past a large multimodal
                    # span, leaving a tail that fits the multimodal-expanded
                    # chunk limit.  Treat it as normal prefill so the model sees
                    # the same single tail chunk as the no-cache path.  Do not
                    # set chunk flags here: spec decoding uses them as a
                    # cross-chunk carry protocol.
                    self.long_context_chunker.clear()
                    inputs, delta, extra_inputs = __create_model_inputs(running)
                else:
                    chunk_size, multimodals = self.long_context_chunker.next_chunk_size()
                    inputs, extra_inputs = __create_inputs_chunk(running, chunk_size, multimodals)
                    inputs.is_first_chunk = True
                    inputs.is_chunk_multimodal = self.long_context_chunker.has_multimodal
                    self._short_prefill_turns_since_long_chunk = 0
            elif len(running) > 0:
                # create inputs
                inputs, delta, extra_inputs = __create_model_inputs(running)
            return running, inputs, delta, extra_inputs, swap_in_map, swap_out_map

        def __create_short_or_normal_prefill_turn():
            nonlocal attempted_short_or_normal_prefill
            attempted_short_or_normal_prefill = True
            result = __create_inputs_prefill(allow_long_prefill=False)
            _, prefill_inputs, prefill_delta, _, _, _ = result
            if prefill_inputs is not None or prefill_delta is not None:
                self._short_prefill_turns_since_long_chunk += 1
            return result

        def __is_empty_forward(forward_inputs: 'ModelInputs|None', forward_delta: 'ModelInputsDelta|None'):
            return forward_inputs is None and forward_delta is None

        def __try_active_long_context_chunk():
            nonlocal attempted_long_work
            nonlocal active_long_chunk_blocked_by_kv
            attempted_long_work = True
            result = __create_inputs_long_context_chunk()
            _, chunk_inputs, chunk_delta, _ = result
            active_long_chunk_blocked_by_kv = __is_empty_forward(chunk_inputs, chunk_delta)
            return result

        def __should_try_short_prefill_before_active_chunk():
            """Allow short/normal prefill quota before an active non-final
            chunk."""
            if self.long_context_chunker.is_last_chunk():
                return False
            if not scheduler.has_waiting():
                return False
            return not self._is_long_context_chunk_turn_due()

        def __has_no_forward():
            return __is_empty_forward(inputs, delta)

        def __can_fallback_to_short_after_long_work():
            if not __has_no_forward():
                return False
            if not attempted_long_work:
                return False
            if active_long_chunk_blocked_by_kv:
                return False
            if attempted_short_or_normal_prefill:
                return False
            return scheduler.has_waiting()

        def __can_try_short_prefill_after_defer():
            if not __has_no_forward():
                return False
            if not deferred_long_context_chunk:
                return False
            if self._is_long_context_chunk_turn_due():
                return False
            return scheduler.has_waiting()

        def __can_retry_deferred_active_chunk():
            return __has_no_forward() and deferred_long_context_chunk and self.long_context_chunker.enabled()

        scheduler = self.scheduler
        logger.debug(f'Make forward inputs with prefill={prefill}, enable_empty={enable_empty}')

        inputs = None
        delta = None
        running = []
        extra_inputs = None
        swap_in_map = {}
        swap_out_map = {}
        deferred_long_context_chunk = False
        attempted_long_work = False
        attempted_short_or_normal_prefill = False
        active_long_chunk_blocked_by_kv = False

        # Bounded opt-TTFT prefill policy: protect decode before continuing
        # non-final long chunks, then allow a bounded number of short/normal
        # prefill turns before forcing one long-work turn. A long-work turn
        # continues the active chunker first, otherwise it admits one waiting
        # long prefill through the scheduler.
        self.long_context_chunker.check_enable()
        if self.long_context_chunker.enabled():
            if self._should_defer_long_context_chunk(prefill):
                deferred_long_context_chunk = True
            elif __should_try_short_prefill_before_active_chunk():
                # After a decode turn, keep the short/normal prefill quota in
                # front of active long chunks; otherwise decode -> long can
                # repeat and small waiting requests remain gated by the active
                # chunker even while the long-work turn is not due.
                (
                    running,
                    inputs,
                    delta,
                    extra_inputs,
                    swap_in_map,
                    swap_out_map,
                ) = __create_short_or_normal_prefill_turn()
                if __is_empty_forward(inputs, delta):
                    running, inputs, delta, extra_inputs = __try_active_long_context_chunk()
            else:
                running, inputs, delta, extra_inputs = __try_active_long_context_chunk()
        elif prefill:
            # prefill
            has_waiting_long_prefill = scheduler.has_waiting_long_prefill()
            if has_waiting_long_prefill and not self._is_long_context_chunk_turn_due():
                (
                    running,
                    inputs,
                    delta,
                    extra_inputs,
                    swap_in_map,
                    swap_out_map,
                ) = __create_short_or_normal_prefill_turn()
                if __has_no_forward():
                    (
                        running,
                        inputs,
                        delta,
                        extra_inputs,
                        swap_in_map,
                        swap_out_map,
                    ) = __create_inputs_prefill(prefer_long_prefill=True)
            else:
                (
                    running,
                    inputs,
                    delta,
                    extra_inputs,
                    swap_in_map,
                    swap_out_map,
                ) = __create_inputs_prefill(prefer_long_prefill=has_waiting_long_prefill)
                attempted_long_work = has_waiting_long_prefill

        # Waiting-long admission failure can still fall back to short prefills.
        # Active-long reservation failure means KV is pinned by running work;
        # admit decode only so existing requests can drain blocks.
        if __can_fallback_to_short_after_long_work():
            (
                running,
                inputs,
                delta,
                extra_inputs,
                swap_in_map,
                swap_out_map,
            ) = __create_short_or_normal_prefill_turn()

        # try decoding
        if inputs is None and len(self.running_seqs) > 0 and self.config.role != EngineRole.Prefill:
            prefill = False
            delta, running, invalid_seqs = self.create_model_inputs_delta()
            self.to_evict_seqs = invalid_seqs
            extra_inputs = None

        if __can_try_short_prefill_after_defer():
            (
                running,
                inputs,
                delta,
                extra_inputs,
                swap_in_map,
                swap_out_map,
            ) = __create_short_or_normal_prefill_turn()

        if __can_retry_deferred_active_chunk():
            running, inputs, delta, extra_inputs = __try_active_long_context_chunk()

        # reset decode count when non-decoding inputs are produced
        if inputs is not None and not inputs.is_decoding:
            self._decode_count = 0

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
        return_ce_loss = __need_ce_loss(running)

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
            return_ce_loss=return_ce_loss,
        )

    def do_prefill_pnode(self):
        return True

    def do_prefill_default(self):
        # decoding if no waiting
        scheduler = self.scheduler
        pending_last_chunk = self._has_pending_last_long_context_chunk()

        # do decoding if not waiting
        if not scheduler.has_waiting() and not pending_last_chunk:
            self._decode_count = 0
            return False
        if pending_last_chunk:
            return True

        # force prefill if too many consecutive decode rounds
        if self._decode_count >= self.config.prefill_interval:
            return True

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
        self._decode_count += 1
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
        self._last_forward_kind = self._forward_kind(inputs, forward_inputs['delta'])
        self.scheduler.tick()
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
