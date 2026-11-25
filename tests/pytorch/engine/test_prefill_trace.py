# Copyright (c) OpenMMLab. All rights reserved.
"""Verify trace hooks across the real async dispatch/output boundaries."""

import asyncio
import pickle
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from lmdeploy.pytorch.engine.engine_loop import EngineLoop
from lmdeploy.pytorch.engine.inputs_maker import InputsMakerAsync
from lmdeploy.pytorch.engine.model_agent import BatchedOutputs
from lmdeploy.pytorch.messages import MessageStatus
from lmdeploy.pytorch.trace import PrefillTrace, begin_prefill


@pytest.mark.parametrize('enabled', [False, True])
def test_prefill_summary_stays_on_host_and_completes_before_response(monkeypatch, enabled):
    monkeypatch.setattr('lmdeploy.pytorch.engine.inputs_maker.TRACE_ENABLED', enabled)
    state = PrefillTrace() if enabled else None
    seq = NS(input_start_pos=0, input_end_pos=96, cached_tokens=16, prefill_trace=state)
    inputs = NS(seq_length=torch.tensor([80]), history_lengths=torch.tensor([16]),
                is_decoding=False, is_chunk=False, is_last_chunk=False, ssm_prefill_meta=None)
    sent = []

    async def forward(payload):
        sent.append(payload)
        assert state is None or state.summary is None

    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = NS(enable_decode_checkpoints=False)
    maker.scheduler = NS(resume_completed_migrations=lambda: None, tick=lambda: None)
    maker.executor = NS(forward_async=forward)
    maker.do_prefill = lambda: True
    maker._make_forward_inputs = lambda *args: dict(running=[seq], inputs=inputs, delta=None, cache_inputs=None)
    payload, running = asyncio.run(maker.send_next_inputs())
    assert running == [seq]
    assert 'prefill_trace' not in sent[0]
    assert ('prefill_trace' in payload) == enabled

    loop = EngineLoop.__new__(EngineLoop)
    loop.inputs_maker = NS(update_running_seqs=lambda *args: None)
    loop.state_checkpoints = Mock()
    loop._prefetch_next_inputs = AsyncMock(return_value=(None, None))
    loop.executor = NS(get_output_async=AsyncMock(
        return_value=NS(ssm_state_access_id=None, next_token_ids=torch.tensor([1]))))

    def make_output(*args):
        if enabled:
            assert state.summary is None  # Completion alone cannot freeze it.
            assert state.input_tokens == 80
            assert state.local_cached == 16
            assert state.chunks == 1
        return None

    loop._finish_forward_output = make_output
    asyncio.run(loop._main_loop_get_outputs(running, payload))
    assert 'prefill_trace' not in sent[0]  # Executor can retain its original dict.


@pytest.mark.parametrize('error', [RuntimeError, asyncio.CancelledError])
def test_failed_dispatch_does_not_claim_completed_prefill(monkeypatch, error):
    monkeypatch.setattr('lmdeploy.pytorch.engine.inputs_maker.TRACE_ENABLED', True)
    state = PrefillTrace()
    seq = NS(input_start_pos=0, input_end_pos=96, cached_tokens=0, prefill_trace=state)
    inputs = NS(seq_length=torch.tensor([96]), history_lengths=torch.tensor([0]),
                is_decoding=False, is_chunk=False, is_last_chunk=False)
    maker = InputsMakerAsync.__new__(InputsMakerAsync)
    maker.config = NS(enable_decode_checkpoints=False)
    maker.scheduler = NS(resume_completed_migrations=lambda: None, tick=lambda: None)
    maker.executor = NS(forward_async=AsyncMock(side_effect=error('executor unavailable')))
    maker.do_prefill = lambda: True
    maker._make_forward_inputs = lambda *args: dict(running=[seq], inputs=inputs, delta=None, cache_inputs=None)
    with pytest.raises(error, match='executor unavailable'):
        asyncio.run(maker.send_next_inputs())
    assert state.summary is None
    assert state.chunks == 0
    assert state.pending == 0
    assert state.invalid


@pytest.mark.parametrize('mode', ['tokenless', 'missing', 'output_error', 'prefetch_error'])
def test_prefill_accounting_settles_without_generated_tokens(monkeypatch, mode):
    monkeypatch.setattr('lmdeploy.pytorch.engine.engine_loop.TRACE_ENABLED', True)
    state = PrefillTrace()
    seq = NS(input_start_pos=0, input_end_pos=96, cached_tokens=0, prefill_trace=state)
    inputs = NS(seq_length=torch.tensor([96]), history_lengths=torch.tensor([0]),
                is_decoding=False, ssm_prefill_meta=None)
    payload = dict(inputs=inputs, delta=None, cache_inputs=None,
                   prefill_trace=begin_prefill([seq], inputs))
    loop = EngineLoop.__new__(EngineLoop)
    loop.inputs_maker = NS(update_running_seqs=lambda *args: None)
    loop.state_checkpoints = Mock()
    loop.scheduler = NS(update_connector_output=lambda output: None)
    loop._prefetch_next_inputs = AsyncMock(return_value=(None, None))
    loop.executor = NS(get_output_async=AsyncMock(return_value=BatchedOutputs(next_token_ids=None, stopped=None)))
    prefetched = None
    if mode == 'missing':
        loop.executor.get_output_async.return_value = None
    elif mode == 'output_error':
        prefetched = NS(input_start_pos=0, input_end_pos=96, cached_tokens=0, prefill_trace=PrefillTrace())
        next_payload = dict(prefill_trace=begin_prefill([prefetched], inputs))
        loop._prefetch_next_inputs.return_value = (next_payload, [prefetched])
        loop.executor.get_output_async.side_effect = asyncio.CancelledError()
    elif mode == 'prefetch_error':
        loop._prefetch_next_inputs.side_effect = asyncio.CancelledError()
    if mode.endswith('error'):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(loop._main_loop_get_outputs([seq], payload))
    else:
        asyncio.run(loop._main_loop_get_outputs([seq], payload))
    assert state.pending == 0
    assert state.summary is None
    assert state.chunks == (1 if mode == 'tokenless' else 0)
    assert state.invalid == (mode != 'tokenless')
    if prefetched is not None:
        assert prefetched.prefill_trace.invalid
        assert prefetched.prefill_trace.pending == 0


@pytest.mark.parametrize('chunked', [False, True])
def test_prefetched_decode_preemption_keeps_accounting_until_output_accepted(monkeypatch, chunked):
    monkeypatch.setattr('lmdeploy.pytorch.engine.engine_loop.TRACE_ENABLED', True)
    state = PrefillTrace()
    seq = NS(status=MessageStatus.RUNNING, num_token_ids=96, generated_ids=[1], session_id=7,
             input_start_pos=0, input_end_pos=96, cached_tokens=0, prefill_trace=state,
             resp=NS(_input_logprobs=None, data=None, first_token_timestamp=0),
             resp_cache=False, sampling_param=NS(num_logprobs=0), engine_events=[],
             routed_experts=None, return_ce_loss=False, return_logits=False)
    inputs = NS(seq_length=torch.tensor([96]), history_lengths=torch.tensor([0]),
                is_decoding=False, is_chunk=chunked, is_last_chunk=chunked,
                logits_indices=None, ssm_prefill_meta=None)
    native = BatchedOutputs(next_token_ids=torch.tensor([1]), stopped=torch.tensor([False]),
                            new_token_timestamp=10)
    loop = EngineLoop.__new__(EngineLoop)
    loop.config = NS(num_speculative_tokens=3, enable_metrics=False)
    loop.seq_strategy = NS(update_running=lambda **kwargs: None)
    loop.scheduler = NS(update_connector_output=lambda output: None,
                        cache_routed_experts=lambda seqs: None,
                        release_completed_prefill_reservations=lambda seqs: None)
    loop.inputs_maker = NS(update_running_seqs=lambda *args: None)
    loop.state_checkpoints = Mock()
    loop.executor = NS(get_output_async=AsyncMock(return_value=native))
    loop.resp_queue = asyncio.Queue()

    async def preempt():
        # Prefetch can reclaim this sequence before the current forward is
        # consumed. The real output path then rejects its generated token.
        state.preempt()
        seq.status = MessageStatus.WAITING
        return None, None

    loop._prefetch_next_inputs = preempt
    payload = dict(inputs=inputs, delta=None, cache_inputs=None,
                   prefill_trace=begin_prefill([seq], inputs))
    asyncio.run(loop._main_loop_get_outputs([seq], payload))
    assert loop.resp_queue.get_nowait() == {}
    assert state.chunks == 1 and state.tokens == 96
    assert state.summary is None
    assert seq.resp.first_token_timestamp == 0

    seq.status = MessageStatus.RUNNING
    loop._prefetch_next_inputs = AsyncMock(return_value=(None, None))
    payload['prefill_trace'] = begin_prefill([seq], inputs)
    native.new_token_timestamp = 20
    asyncio.run(loop._main_loop_get_outputs([seq], payload))
    assert loop.resp_queue.get_nowait()[7].token_ids == [1]
    assert seq.resp.first_token_timestamp == 20
    summary = state.summary
    assert summary.chunk_count == 2
    assert summary.computed_tokens == summary.computed_input_tokens == 192
    assert summary.recomputed_tokens == 96
    assert summary.preemption_count == summary.reschedule_count == 1
    state.preempt()  # Later decode work must not rewrite the accepted summary.
    assert begin_prefill([seq], inputs) is None
    assert state.summary is summary


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('terminal', [False, True])
def test_native_output_carries_summary_only_at_request_end(monkeypatch, enabled, terminal):
    monkeypatch.setattr('lmdeploy.pytorch.engine.engine_loop.TRACE_ENABLED', enabled)
    seq = NS(status=MessageStatus.RUNNING, num_token_ids=1, generated_ids=[1], session_id=7,
             input_start_pos=0, input_end_pos=96, cached_tokens=16,
             resp=NS(_input_logprobs=None, data=None, first_token_timestamp=0),
             resp_cache=False, sampling_param=NS(num_logprobs=0), engine_events=[],
             routed_experts=None, return_ce_loss=False, return_logits=False,
             prefill_trace=PrefillTrace() if enabled else None)
    if enabled:
        record = seq.prefill_trace.dispatch(seq, 16, 80, 1)
        seq.prefill_trace.complete(*record[1:], now=2)
        seq.prefill_trace.accept()

    def update_running(**kwargs):
        if terminal:
            seq.status = MessageStatus.STOPPED

    loop = EngineLoop.__new__(EngineLoop)
    loop.config = NS(num_speculative_tokens=None)
    loop.seq_strategy = NS(update_running=update_running)
    loop.scheduler = NS(cache_routed_experts=lambda seqs: None)
    native_output = BatchedOutputs(next_token_ids=torch.tensor([1]), stopped=torch.tensor([terminal]),
                                   new_token_timestamp=10)
    output = loop._make_infer_outputs(native_output, [seq], None, None)[7]
    metrics = pickle.loads(pickle.dumps(output.req_metrics))
    if enabled and terminal:
        assert metrics.prefill == seq.prefill_trace.summary
        assert metrics.prefill.computed_input_tokens == 80
    else:
        assert metrics.prefill is None
        assert 'prefill' not in vars(metrics)
