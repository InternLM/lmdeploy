# Copyright (c) OpenMMLab. All rights reserved.
"""Engine-loop input construction for the LMDeploy PyTorch backend.

This module converts scheduler decisions into model-agent inputs.  Most helpers
build tensor fields for full-batch ``ModelInputs``; ``InputsMakerAsync`` is the
coordinator that chooses prefill/chunk/decode work, attaches per-forward
metadata, dispatches it to the executor, and updates local running state.
"""
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.profiler import record_function

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.pytorch.long_context import (
    get_long_context_chunk_limit,
    has_long_context_multimodal,
    plan_long_context_chunk,
    sort_long_context_multimodals,
)
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

        input_mm = seq.get_input_multimodals()
        # Only remaining multimodals are emitted by next_chunk_size().
        self.multimodals = sort_long_context_multimodals(input_mm)
        self.max_prefill_num = get_long_context_chunk_limit(seq, self.max_prefill_token_num)
        self.has_multimodal = has_long_context_multimodal(self.multimodals)

    def next_chunk_size(self):
        """Get the next chunk size and its remaining multimodal payloads."""
        seq = self.seq
        if seq is None:
            return 0, None

        plan = plan_long_context_chunk(seq, self.max_prefill_num, self.multimodals)
        return plan.chunk_size, plan.multimodals

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


@dataclass
class _ForwardInputsState:
    """Mutable branch state for one ``_ForwardInputsTask.run()`` call.

    ``_ForwardInputsTask`` may try several mutually exclusive work sources
    before it produces a forward payload: active long-context continuation,
    waiting prefill, decode, and retry paths after a deferred long chunk. These
    flags record which branches have already been attempted so the fallback
    checks can preserve the original scheduling order without threading many
    local variables through the helper calls.

    Args:
        prefill: Caller-selected top-level mode from ``do_prefill()``. Decode
            fallback sets this to ``False`` after a decode turn is chosen.
        enable_empty: Forwarded from prefetch/send code for parity with the
            existing call signature. It is currently diagnostic only and does
            not allow empty payloads.
        active_chunk_deferred: Whether an active long-context chunk yielded at
            the start of this task and may be retried after decode or a
            short-prefill opportunity.
        tried_long_work: Whether this task has already tried a
            long-context continuation or a waiting-long-prefill lane.
        tried_short_prefill: Whether this task has already called
            ``_try_short_prefill()``. Here "short" means the non-long-prefill
            lane selected with ``allow_long_prefill=False``.
        active_chunk_blocked_by_kv: Whether active chunk reservation failed
            because KV is pinned by running work. In that case the task should
            let decode drain resources instead of admitting short prefill.
    """

    prefill: bool
    enable_empty: bool = False
    active_chunk_deferred: bool = False
    tried_long_work: bool = False
    tried_short_prefill: bool = False
    active_chunk_blocked_by_kv: bool = False


@dataclass
class _ForwardInputsResult:
    """Selected work and payload fragments for one executor forward."""

    running: 'SeqList' = field(default_factory=list)
    inputs: ModelInputs | None = None
    delta: ModelInputsDelta | None = None
    extra_inputs: object | None = None
    swap_in_map: dict = field(default_factory=dict)
    swap_out_map: dict = field(default_factory=dict)

    def is_empty(self):
        return self.inputs is None and self.delta is None

    def set_work(self,
                 running: 'SeqList',
                 inputs: ModelInputs | None,
                 delta: ModelInputsDelta | None,
                 extra_inputs: object | None):
        """Replace forward work while preserving scheduler swap maps."""
        self.running = running
        self.inputs = inputs
        self.delta = delta
        self.extra_inputs = extra_inputs


class _ForwardInputsTask:
    """Per-call state machine for selecting the next forward payload."""

    def __init__(self,
                 maker: 'InputsMakerAsync',
                 prefill: bool,
                 enable_empty: bool = False):
        self.maker = maker
        self.scheduler = maker.scheduler
        self.state = _ForwardInputsState(prefill=prefill, enable_empty=enable_empty)
        self.result = _ForwardInputsResult()

    def run(self):
        """Select one executor forward while preserving scheduling order.

        The branch order is part of the opt-TTFT long-context policy:
        1. Pick the primary lane from the active chunker or ``do_prefill()``.
        2. Let failed waiting-long work fall back to short prefill, while
           active-chunk KV pressure falls through to decode.
        3. Try decode when no prefill/chunk payload was selected.
        4. For deferred active chunks, decode gets the first chance. If decode
           produces no payload, spend the still-unused short-prefill chance
           before retrying the chunk.
        """
        maker = self.maker
        state = self.state
        logger.debug(f'Make forward inputs with prefill={state.prefill}, '
                     f'enable_empty={state.enable_empty}')

        # 1. Pick the primary lane for this loop. Active chunks are
        # engine-local runnable work, so they are considered before waiting
        # scheduler prefill. Without an active chunk, honor do_prefill().
        maker.long_context_chunker.check_enable()
        if maker.long_context_chunker.enabled():
            self._select_active_chunk_work()
        elif state.prefill:
            self._select_prefill_work()

        # 2. Waiting-long admission failure can still fall back to short
        # prefill. Active chunk reservation failure means KV is pinned by
        # running work; let decode drain blocks instead of admitting more
        # prefill.
        if self._can_fallback_to_short_prefill():
            self._try_short_prefill()

        # 3. If no prefill/chunk payload was selected, run decode so active
        # requests make progress and can release cache pressure.
        if self._can_try_decode():
            self._try_decode()

        # 4. Stage 1 only marked the active chunk as deferred; it did not try
        # short prefill. Stage 3 gave decode the first chance. If decode
        # produced no payload and the long-work turn is still not due, spend
        # the still-unused short/normal-prefill chance before retrying the
        # deferred chunk.
        if self._can_try_short_after_defer():
            self._try_short_prefill()

        if self._can_retry_deferred_chunk():
            self._try_active_chunk()

        # A non-decode payload satisfies the prefill side of the starvation
        # guard, so future do_prefill() calls can count decode rounds again.
        if self.result.inputs is not None and not self.result.inputs.is_decoding:
            maker._decode_count = 0

        if self.result.is_empty():
            return None
        return self._build_payload()

    def _select_active_chunk_work(self):
        if self._should_defer_active_chunk():
            self.state.active_chunk_deferred = True
        elif self._should_try_short_prefill_first():
            # After a decode turn, keep the short/normal prefill quota in front
            # of active long chunks; otherwise decode -> long can repeat and
            # small waiting requests remain gated by the active chunker even
            # while the long-work turn is not due.
            self._try_short_prefill()
            if self.result.is_empty():
                self._try_active_chunk()
        else:
            self._try_active_chunk()

    def _should_defer_active_chunk(self):
        """Check whether the active long-context chunk should yield this loop."""
        maker = self.maker
        if maker.config.role == EngineRole.Prefill:
            return False
        if not maker.long_context_chunker.enabled():
            return False
        if maker.long_context_chunker.is_last_chunk():
            if len(maker.running_seqs) == 0:
                return False
            return not self.state.prefill
        return getattr(maker, '_last_forward_kind', None) == 'long_context_chunk'

    def _should_try_short_prefill_first(self):
        """Allow short/normal prefill quota before an active non-final chunk."""
        maker = self.maker
        if maker.long_context_chunker.is_last_chunk():
            return False
        if not self.scheduler.has_waiting():
            return False
        return not maker._is_long_context_chunk_turn_due()

    def _select_prefill_work(self):
        maker = self.maker
        has_waiting_long_prefill = self.scheduler.has_waiting_long_prefill()
        if has_waiting_long_prefill and not maker._is_long_context_chunk_turn_due():
            self._try_short_prefill()
            if self.result.is_empty():
                self.state.tried_long_work = True
                self.result = self._schedule_prefill(prefer_long_prefill=True)
        else:
            self.result = self._schedule_prefill(prefer_long_prefill=has_waiting_long_prefill)
            self.state.tried_long_work = has_waiting_long_prefill

    def _can_fallback_to_short_prefill(self):
        if self.state.active_chunk_blocked_by_kv and not self.state.tried_long_work:
            logger.warning('Unexpected forward-input state: active long-context chunk is marked '
                           'KV-blocked before long work was attempted.')
        if self.state.active_chunk_deferred and self.state.tried_long_work:
            logger.warning('Unexpected forward-input state: active long-context chunk was both '
                           'deferred and attempted in the primary lane.')
        if not self.result.is_empty():
            return False
        if not self.state.tried_long_work:
            return False
        if self.state.active_chunk_blocked_by_kv:
            return False
        if self.state.tried_short_prefill:
            return False
        return self.scheduler.has_waiting()

    def _can_try_decode(self):
        maker = self.maker
        if self.result.inputs is not None:
            return False
        if self.result.delta is not None:
            logger.warning('Unexpected forward-input state: decode fallback reached after a '
                           'delta-only payload was already selected.')
            return False
        if len(maker.running_seqs) == 0:
            return False
        return maker.config.role != EngineRole.Prefill

    def _can_try_short_after_defer(self):
        if not self.result.is_empty():
            return False
        if not self.state.active_chunk_deferred:
            return False
        if self.state.tried_short_prefill:
            logger.warning('Unexpected forward-input state: deferred active chunk is trying '
                           'short prefill after short/normal prefill was already attempted.')
            return False
        if self.state.active_chunk_blocked_by_kv:
            logger.warning('Unexpected forward-input state: deferred active chunk is also '
                           'marked KV-blocked.')
            return False
        if self.maker._is_long_context_chunk_turn_due():
            return False
        return self.scheduler.has_waiting()

    def _can_retry_deferred_chunk(self):
        if not self.result.is_empty():
            return False
        if not self.state.active_chunk_deferred:
            return False
        if not self.maker.long_context_chunker.enabled():
            logger.warning('Unexpected forward-input state: active long-context chunk was '
                           'deferred but the chunker is no longer enabled.')
            return False
        return True

    def _try_short_prefill(self):
        self.state.tried_short_prefill = True
        self.result = self._schedule_prefill(allow_long_prefill=False)
        if not self.result.is_empty():
            self.maker._short_prefill_turns_since_long_chunk += 1

    def _try_active_chunk(self):
        self.state.tried_long_work = True
        result = self._build_active_chunk()
        self.state.active_chunk_blocked_by_kv = result.is_empty()
        self.result.set_work(result.running, result.inputs, result.delta, result.extra_inputs)

    def _try_decode(self):
        maker = self.maker
        self.state.prefill = False
        delta, running, invalid_seqs = maker.create_model_inputs_delta()
        maker.to_evict_seqs = invalid_seqs
        self.result.set_work(running, None, delta, None)

    def _schedule_prefill(self,
                          allow_long_prefill: bool = True,
                          prefer_long_prefill: bool = False):
        maker = self.maker
        if maker.config.role == EngineRole.Prefill:
            prealloc_size = 0
        else:
            prealloc_size = maker.engine_strategy.get_prealloc_size(True)
        scheduler_output = self.scheduler.schedule(is_prefill=True,
                                                   prealloc_size=prealloc_size,
                                                   allow_long_prefill=allow_long_prefill,
                                                   prefer_long_prefill=prefer_long_prefill)
        running = scheduler_output.running
        result = _ForwardInputsResult(running=running,
                                      swap_in_map=scheduler_output.swap_in_map,
                                      swap_out_map=scheduler_output.swap_out_map)

        if len(running) == 1 and maker.long_context_chunker.is_long_context(running[0]):
            maker.long_context_chunker.set_seq(running[0])
            if maker.long_context_chunker.is_last_chunk():
                # A prefix-cache restore can skip past a large multimodal
                # span, leaving a tail that fits the multimodal-expanded chunk
                # limit.  Treat it as normal prefill so the model sees the same
                # single tail chunk as the no-cache path.  Do not set chunk
                # flags here: spec decoding uses them as a cross-chunk carry
                # protocol.
                maker.long_context_chunker.clear()
                result.inputs, result.delta, result.extra_inputs = self._build_prefill_inputs(running)
            else:
                chunk_size, multimodals = maker.long_context_chunker.next_chunk_size()
                result.inputs, result.extra_inputs = self._build_chunk_inputs(running, chunk_size, multimodals)
                result.inputs.is_first_chunk = True
                result.inputs.is_chunk_multimodal = maker.long_context_chunker.has_multimodal
                maker._short_prefill_turns_since_long_chunk = 0
        elif len(running) > 0:
            result.inputs, result.delta, result.extra_inputs = self._build_prefill_inputs(running)
        return result

    def _build_active_chunk(self):
        maker = self.maker
        seq = maker.long_context_chunker.seq
        chunk_size, multimodals = maker.long_context_chunker.next_chunk_size()
        is_last_chunk = maker.long_context_chunker.is_last_chunk()
        is_chunk_multimodal = maker.long_context_chunker.has_multimodal
        if not self._reserve_chunk(seq, chunk_size, is_last_chunk):
            return _ForwardInputsResult()

        running = [seq]
        if is_last_chunk:
            inputs, delta, extra_inputs = self._build_prefill_inputs(running)
            inputs.is_chunk = True
            inputs.is_last_chunk = True
            maker.long_context_chunker.clear()
        else:
            inputs, extra_inputs = self._build_chunk_inputs(running, chunk_size, multimodals)
            delta = None
        inputs.is_first_chunk = False
        inputs.is_chunk_multimodal = is_chunk_multimodal
        maker._short_prefill_turns_since_long_chunk = 0
        return _ForwardInputsResult(running=running,
                                    inputs=inputs,
                                    delta=delta,
                                    extra_inputs=extra_inputs)

    def _reserve_chunk(self,
                       seq: 'SchedulerSequence',
                       chunk_size: int,
                       is_last_chunk: bool):
        maker = self.maker
        if maker.config.role == EngineRole.Prefill:
            prealloc_size = 0
        elif is_last_chunk:
            prealloc_size = maker.engine_strategy.get_prealloc_size(True)
        else:
            prealloc_size = 0
        return self.scheduler.reserve_long_context_chunk(seq,
                                                         chunk_size,
                                                         prealloc_size=prealloc_size,
                                                         is_last_chunk=is_last_chunk)

    def _build_prefill_inputs(self, seqs: 'SeqList'):
        maker = self.maker
        inputs = maker.create_model_inputs(seqs, True)
        delta, valid_seqs, _ = maker.create_model_inputs_delta_valid_only()
        maker.running_seqs = valid_seqs
        extra_inputs = maker.model_agent_strategy.make_extra_inputs(seqs, inputs)
        return inputs, delta, extra_inputs

    def _build_chunk_inputs(self,
                            running: 'SeqList',
                            chunk_size: int,
                            multimodals: 'MultiModalInputs|None'):
        maker = self.maker
        inputs = maker.create_model_inputs_long_context(running[0], chunk_size, multimodals)
        extra_inputs = maker.model_agent_strategy.make_extra_inputs(running, inputs)
        return inputs, extra_inputs

    def _need_logits(self):
        if self.maker.spec_decoding:
            return True
        return any(seq.return_logits for seq in self.result.running)

    def _need_routed_experts(self):
        return any(seq.return_routed_experts for seq in self.result.running)

    def _need_ce_loss(self):
        return any(seq.return_ce_loss for seq in self.result.running)

    def _build_payload(self):
        maker = self.maker
        result = self.result
        sampling_inputs = maker.sampling_strategy.make_sampling_inputs(result.running)
        if result.inputs is not None:
            stopping_criteria = maker.model_agent_strategy.make_stopping_criteria(result.running)
        else:
            stopping_criteria = None

        return dict(
            running=result.running,
            inputs=result.inputs,
            delta=result.delta,
            swap_in_map=result.swap_in_map,
            swap_out_map=result.swap_out_map,
            sampling_inputs=sampling_inputs,
            stopping_criteria=stopping_criteria,
            return_logits=self._need_logits(),
            extra_inputs=result.extra_inputs,
            return_routed_experts=self._need_routed_experts(),
            return_ce_loss=self._need_ce_loss(),
        )


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

        # consecutive decode counter for prefill starvation prevention
        self._decode_count = 0
        self._last_forward_kind = None
        self._short_prefill_turns_since_long_chunk = 0
        self._short_prefill_turns_per_long_chunk = max(1, _envs.opt_ttft_short_turns)

        # record for next forward.
        self.next_is_prefill = True
        self.forward_inputs = None

        # running seqs
        # mark the seqs that have been sent to executor
        self.running_seqs: list[SchedulerSequence] = []
        self.to_evict_seqs: list[SchedulerSequence] = []

        # long context chunker
        self.long_context_chunker = LongContextChunker(config.max_prefill_token_num)

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
        return _ForwardInputsTask(self, prefill, enable_empty).run()

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
