# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.messages import ResponseType
from lmdeploy.pytorch.engine.engine_loop import EngineLoop
from lmdeploy.pytorch.engine.model_agent.agent import BatchedLogProbs
from lmdeploy.pytorch.messages import (
    MessageStatus,
    SamplingParam,
    SchedulerSequence,
    SequenceMeta,
    UpdateTokenMode,
)
from lmdeploy.pytorch.strategies.ar.sequence import (
    ARSequenceStrategy,
    SchedulerSequenceDefault,
)


def _compact(actual_ids, width=1):
    if not actual_ids:
        return BatchedLogProbs(vals=torch.empty((0, width)),
                               indices=torch.empty((0, width),
                                                   dtype=torch.int32))
    indices = torch.tensor([[token_id] * width for token_id in actual_ids],
                           dtype=torch.int32)
    vals = torch.tensor([[-float(token_id)] * width
                         for token_id in actual_ids])
    return BatchedLogProbs(vals=vals, indices=indices)


def test_logprob_start_position_is_request_relative():
    seq = object.__new__(SchedulerSequence)
    seq.input_start_pos = 17
    seq.sampling_param = SamplingParam(num_logprobs=0,
                                       logprob_start_len=4)
    assert seq.logprob_start_pos == 21
    seq.sampling_param = SamplingParam(num_logprobs=0,
                                       logprob_start_len=-1)
    assert seq.logprob_start_pos == -1


def test_engine_loop_accumulates_mixed_rows_in_chunk_order():
    loop = EngineLoop.__new__(EngineLoop)
    messages = [
        SimpleNamespace(logprob_start_pos=0,
                        resp=SimpleNamespace(_input_logprobs=None)),
        SimpleNamespace(logprob_start_pos=0,
                        resp=SimpleNamespace(_input_logprobs=None)),
    ]
    inputs = SimpleNamespace(
        is_decoding=False,
        logits_indices=torch.tensor([0, 1, 3]),
        seq_logit_length=torch.tensor([2, 1]),
    )
    outputs = SimpleNamespace(logprobs=_compact([2, 3, 9]))

    loop._append_input_logprobs(outputs, messages, inputs)

    assert messages[0].resp._input_logprobs == [
        ([-2.0], [2]),
        ([-3.0], [3]),
    ]
    assert messages[1].resp._input_logprobs == [([-9.0], [9])]

    empty_inputs = SimpleNamespace(
        is_decoding=False,
        logits_indices=torch.empty(0, dtype=torch.long),
        seq_logit_length=torch.tensor([0, 0]),
    )
    loop._append_input_logprobs(SimpleNamespace(logprobs=_compact([])),
                                messages, empty_inputs)
    assert messages[0].resp._input_logprobs[-1] == ([-3.0], [3])

    unrequested = SimpleNamespace(
        logprob_start_pos=-1,
        resp=SimpleNamespace(_input_logprobs=None),
    )
    loop._append_input_logprobs(
        SimpleNamespace(logprobs=_compact([])),
        [unrequested],
        SimpleNamespace(is_decoding=False,
                        logits_indices=torch.empty(0, dtype=torch.long),
                        seq_logit_length=torch.tensor([0])),
    )
    assert unrequested.resp._input_logprobs is None


def test_engine_loop_rejects_malformed_compact_scoring_segments():
    messages = [
        SimpleNamespace(logprob_start_pos=0,
                        resp=SimpleNamespace(_input_logprobs=None)),
        SimpleNamespace(logprob_start_pos=0,
                        resp=SimpleNamespace(_input_logprobs=None)),
    ]
    inputs = SimpleNamespace(
        is_decoding=False,
        logits_indices=torch.arange(2),
        seq_logit_length=torch.tensor([1, 1]),
    )
    with pytest.raises(RuntimeError, match='compact output mismatch'):
        EngineLoop._append_input_logprobs(
            SimpleNamespace(logprobs=_compact([0])),
            messages, inputs)


@pytest.mark.parametrize(('actual_ids', 'expected'), [
    ([], None),
    ([2, 3], [([-2.0], [2]), ([-3.0], [3])]),
])
def test_final_scoring_prefill_uses_existing_logprob_carrier(actual_ids,
                                                             expected):
    loop = EngineLoop.__new__(EngineLoop)

    def finish_running(**kwargs):
        kwargs['running'][0].status = MessageStatus.STOPPED

    loop.seq_strategy = SimpleNamespace(update_running=finish_running)
    loop.scheduler = SimpleNamespace(
        block_trie=SimpleNamespace(cache_routed_experts=lambda running: None))
    loop.config = SimpleNamespace(num_speculative_tokens=None,
                                  enable_metrics=False)
    response = SimpleNamespace(data=None,
                               _input_logprobs=None,
                               is_done=False,
                               type=ResponseType.SUCCESS)
    msg = SimpleNamespace(
        session_id=7,
        status=MessageStatus.RUNNING,
        num_token_ids=3,
        generated_ids=np.empty((0,), dtype=np.int64),
        logprob_start_pos=0,
        resp=response,
        resp_cache=False,
        sampling_param=SamplingParam(max_new_tokens=0,
                                     num_logprobs=0,
                                     logprob_start_len=0),
        cached_tokens=0,
        engine_events=[],
        routed_experts=None,
        return_ce_loss=False,
        return_logits=False,
    )
    inputs = SimpleNamespace(
        is_decoding=False,
        is_chunk=False,
        is_last_chunk=False,
        logits_indices=torch.arange(len(actual_ids)),
        seq_logit_length=torch.tensor([len(actual_ids)]),
    )
    batched = SimpleNamespace(
        logprobs=_compact(actual_ids),
        logits=None,
        all_routed_experts=None,
        ce_loss=None,
        new_token_timestamp=0,
        stop_pos=torch.tensor([-1]),
        next_token_ids=torch.tensor([0]),
    )

    result = loop._make_infer_outputs(batched, [msg], inputs, delta=None)[7]

    assert result.finish
    assert result.token_ids.tolist() == []
    assert result.logprobs == expected

    sent = []
    loop._send_resp = sent.append
    loop._send_resps([result])
    if expected is None:
        assert response.data is None
    else:
        assert response.data['logprobs'] == [
            dict(zip(indices, vals)) for vals, indices in expected
        ]
    assert sent == [result]


def test_cancelled_scoring_response_drops_partial_accumulator(monkeypatch):
    """Cancellation must never expose rows accumulated by earlier chunks."""
    loop = EngineLoop.__new__(EngineLoop)
    loop.req_manager = object()
    sent = []
    monkeypatch.setattr(
        'lmdeploy.pytorch.engine.engine_loop.response_reqs',
        lambda req_manager, resp, resp_type, data: sent.append(
            (resp_type, data)))
    response = SimpleNamespace(
        data=None,
        _input_logprobs=[([-0.5], [2])],
        is_done=True,
        type=ResponseType.CANCEL,
    )
    out = SimpleNamespace(
        resp=response,
        finish=False,
        token_ids=np.empty((0,), dtype=np.int64),
        logits=None,
        cache_block_ids=None,
        req_metrics=None,
        routed_experts=None,
        ce_loss=None,
    )

    loop._send_resp(out)

    assert response._input_logprobs == [([-0.5], [2])]
    assert sent == [(ResponseType.CANCEL, {
        'token_ids': out.token_ids,
        'logits': None,
        'cache_block_ids': None,
        'req_metrics': None,
        'routed_experts': None,
        'logprobs': None,
        'ce_loss': None,
    })]


@pytest.mark.parametrize('sampling_param', [
    SamplingParam(max_new_tokens=0, logprob_start_len=-1),
    SamplingParam(max_new_tokens=0, num_logprobs=0, logprob_start_len=0),
])
def test_zero_token_prefill_finishes_without_committing_inert_token(
        sampling_param):
    strategy = ARSequenceStrategy()
    session = SimpleNamespace(
        seq_meta=SequenceMeta(block_size=16, strategy=strategy),
        session_id=1,
    )
    seq = SchedulerSequenceDefault(
        seq_id=1,
        session=session,
        sampling_param=sampling_param,
    )
    state = SimpleNamespace(status=MessageStatus.RUNNING)
    state.finish = lambda: setattr(state, 'status', MessageStatus.STOPPED)
    seq.set_state(state)
    seq.update_token_ids(np.array([1, 2, 3]), mode=UpdateTokenMode.INPUTS)
    outputs = SimpleNamespace(
        next_token_ids=torch.tensor([99]),
        stopped=torch.tensor([True]),
        model_metas=[{'turn': 1}],
        all_routed_experts=None,
    )
    inputs = SimpleNamespace(seq_length=torch.tensor([3]),
                             is_decoding=False)

    strategy.update_running([seq], outputs, inputs, delta=None)

    assert seq.status == MessageStatus.STOPPED
    assert seq.num_history_ids == 3
    assert seq.num_token_ids == 0
    assert seq.generated_ids.tolist() == []
    assert seq.model_meta == {'turn': 1}
