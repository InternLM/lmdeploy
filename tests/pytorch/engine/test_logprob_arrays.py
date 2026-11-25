# Copyright (c) OpenMMLab. All rights reserved.
"""Generated-token logprob rows from the engine loop to EngineOutput.

``logprobs_format='array'`` keeps a request's rows as a LogprobArrays; the
default ``'dict'`` format must keep producing the same dicts as before.
"""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lmdeploy.messages import EngineOutput, GenerationConfig, LogprobArrays, Response, ResponseType
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.engine_loop import EngineLoop
from lmdeploy.pytorch.engine.logits_process import FusedLogitsProcessor, SamplingInputs
from lmdeploy.pytorch.engine.model_agent.agent import BatchedLogProbs
from lmdeploy.pytorch.engine.mp_engine.base_worker import EngineOutputGather
from lmdeploy.pytorch.messages import MessageStatus, SamplingParam


def _batched_rows(raw_logprobs: torch.Tensor, token_ids: torch.Tensor, max_num_logprobs: int):
    """Rows as the sampler computes them for the whole batch."""
    processor = FusedLogitsProcessor(SamplingInputs(max_num_logprobs=max_num_logprobs))
    vals, indices = processor.compute_logprobs(raw_logprobs, token_ids)
    return BatchedLogProbs(vals=vals.clone(), indices=indices.clone())


def _msg(session_id, num_logprobs, logprobs_format='dict', num_tokens=1):
    return SimpleNamespace(
        session_id=session_id,
        status=MessageStatus.RUNNING,
        num_token_ids=4,
        generated_ids=np.arange(num_tokens, dtype=np.int64),
        resp=SimpleNamespace(data=None,
                             _input_logprobs=None,
                             is_done=False,
                             type=ResponseType.SUCCESS,
                             first_token_timestamp=None),
        resp_cache=False,
        sampling_param=SamplingParam(num_logprobs=num_logprobs, logprobs_format=logprobs_format),
        cached_tokens=0,
        engine_events=[],
        routed_experts=None,
        return_ce_loss=False,
        return_logits=False,
    )


def _loop():
    loop = EngineLoop.__new__(EngineLoop)
    loop.seq_strategy = SimpleNamespace(update_running=lambda **kwargs: None)
    loop.scheduler = SimpleNamespace(cache_routed_experts=lambda running: None)
    loop.config = SimpleNamespace(num_speculative_tokens=None, enable_metrics=False)
    return loop


def _infer(loop, running, logprobs, stop_pos=None):
    batched = SimpleNamespace(
        logprobs=logprobs,
        logits=None,
        all_routed_experts=None,
        ce_loss=None,
        new_token_timestamp=0,
        stop_pos=stop_pos,
        next_token_ids=torch.zeros(len(running), dtype=torch.long),
    )
    return loop._make_infer_outputs(batched, running, model_inputs=None, delta=None)


def test_mixed_batch_gives_each_request_its_own_columns():
    torch.manual_seed(0)
    raw = torch.randn(3, 50).log_softmax(-1).to(torch.bfloat16)
    sampled = torch.tensor([raw[0].argmax().item(), 7, 9])
    rows = _batched_rows(raw, sampled, max_num_logprobs=5)
    running = [_msg(1, 5, 'array'), _msg(2, 2, 'dict'), _msg(3, -1, 'array')]

    outputs = _infer(_loop(), running, rows)

    flat = outputs[1].logprobs
    assert isinstance(flat, LogprobArrays)
    assert flat.vals.dtype == np.float32 and flat.ids.dtype == np.int32
    assert flat.vals.shape == flat.ids.shape == (1, 6)
    assert flat.ids[0, 0] == sampled[0]
    topk = torch.topk(raw[0].float(), 5)
    np.testing.assert_array_equal(flat.ids[0, 1:], topk.indices.numpy())
    np.testing.assert_array_equal(flat.vals[0, 1:], topk.values.numpy())
    # The sampled token ranks first here, so it is in column 0 and in the top-k.
    assert flat.vals[0, 0] == flat.vals[0, 1]

    nested = outputs[2].logprobs
    assert len(nested) == 1
    vals, indices = nested[0]
    assert len(vals) == len(indices) == 3
    assert indices[0] == 7 and vals[0] == raw[1, 7].item()
    assert indices[1:] == torch.topk(raw[1].float(), 2).indices.tolist()

    assert outputs[3].logprobs is None


def test_dict_format_matches_the_previous_conversion():
    torch.manual_seed(1)
    raw = torch.randn(2, 40).log_softmax(-1)
    sampled = torch.tensor([3, 11])
    rows = _batched_rows(raw, sampled, max_num_logprobs=4)
    running = [_msg(1, 4), _msg(2, 4)]
    loop = _loop()
    outputs = _infer(loop, running, rows)

    # Former engine loop: per-token lists, then dict(zip(indices, vals)).
    expected = []
    for idx in range(2):
        vals = rows.vals[idx:idx + 1].tolist()
        indices = rows.indices[idx:idx + 1].tolist()
        expected.append([dict(zip(i[:5], v[:5])) for v, i in zip(vals, indices)])

    loop._send_resp = lambda out: None
    loop._send_resps(list(outputs.values()))
    assert running[0].resp.data['logprobs'] == expected[0]
    assert running[1].resp.data['logprobs'] == expected[1]


def test_array_rows_drop_rejected_draft_and_post_stop_rows():
    # Two decode positions per request, as with one speculative token.
    torch.manual_seed(2)
    raw = torch.randn(4, 30).log_softmax(-1)
    tokens = torch.tensor([5, -1, 6, 8])
    rows = _batched_rows(raw, tokens.clamp_min(0), max_num_logprobs=3)
    rows.indices[1, 0] = -1  # rejected draft position of request 1
    running = [_msg(1, 3, 'array', num_tokens=1), _msg(2, 3, 'array', num_tokens=1)]

    outputs = _infer(_loop(), running, rows, stop_pos=torch.tensor([-1, 0]))

    assert outputs[1].logprobs.ids[:, 0].tolist() == [5]
    # Request 2 stopped at its first position.
    assert outputs[2].logprobs.ids[:, 0].tolist() == [6]
    np.testing.assert_array_equal(outputs[2].logprobs.vals[0], rows.vals[2].numpy())


def test_steps_queued_before_a_read_concatenate_in_order():
    loop = _loop()
    loop._send_resp = lambda out: None
    msg = _msg(1, 2, 'array')
    parts = []
    for step in range(3):
        raw = torch.randn(1, 20).log_softmax(-1)
        rows = _batched_rows(raw, torch.tensor([step]), max_num_logprobs=2)
        parts.append(_infer(loop, [msg], rows)[1])
    loop._send_resps(parts)

    queued = msg.resp.data['logprobs']
    assert len(queued) == 3 and all(isinstance(part, LogprobArrays) for part in queued)
    merged = LogprobArrays.concat(queued)
    assert merged.ids[:, 0].tolist() == [0, 1, 2]
    assert merged.vals.shape == (3, 3)


class _ReqSender:

    sender_id = 0

    def __init__(self, responses):
        self.responses = list(responses)

    def send_async(self, req_type, data):
        return SimpleNamespace(is_done=True)

    async def async_recv(self, resp, wait_main=True):
        return self.responses.pop(0)


def _rows(token_ids, k=2):
    ids = np.array([[t] + list(range(100, 100 + k)) for t in token_ids], dtype=np.int32)
    vals = -np.arange(ids.size, dtype=np.float32).reshape(ids.shape)
    return LogprobArrays(vals=vals, ids=ids)


def test_engine_instance_concatenates_popped_parts():
    instance = EngineInstance.__new__(EngineInstance)
    instance.max_input_len = 100
    instance.engine = SimpleNamespace(vocab_size=1000, req_manager=SimpleNamespace(senders={}))
    instance._get_extra_outputs = lambda *args: {}
    a, b, c = _rows([7]), _rows([8]), _rows([9])
    instance.req_sender = _ReqSender([
        SimpleNamespace(type=ResponseType.SUCCESS,
                        data=dict(token_ids=np.array([7, 8]), logprobs=[a, b])),
        SimpleNamespace(type=ResponseType.FINISH,
                        data=dict(token_ids=np.array([7, 8, 9]), logprobs=[c])),
    ])

    async def collect():
        gen_config = GenerationConfig(logprobs=2, logprobs_format='array')
        return [out async for out in instance.async_stream_infer(1, [1, 2], gen_config=gen_config)]

    instance.engine.req_manager.senders[0] = None
    outputs = asyncio.run(collect())
    assert [out.token_ids for out in outputs] == [[7, 8], [9]]
    assert outputs[0].logprobs.ids[:, 0].tolist() == [7, 8]
    assert outputs[1].logprobs.ids[:, 0].tolist() == [9]


def test_mp_gather_concatenates_array_parts_and_keeps_dicts():
    gather = EngineOutputGather()
    gather.add(0, EngineOutput(ResponseType.SUCCESS, [7], logprobs=_rows([7])))
    gather.add(0, EngineOutput(ResponseType.SUCCESS, [8, 9], logprobs=_rows([8, 9])))
    out = gather.pop(0, EngineOutput(ResponseType.SUCCESS, []))
    assert out.token_ids == [7, 8, 9]
    assert out.logprobs.ids[:, 0].tolist() == [7, 8, 9]

    gather.add(1, EngineOutput(ResponseType.SUCCESS, [7], logprobs=[{7: -0.5}]))
    gather.add(1, EngineOutput(ResponseType.SUCCESS, [8], logprobs=[{8: -0.25}]))
    assert gather.pop(1, EngineOutput(ResponseType.SUCCESS, [])).logprobs == [{7: -0.5}, {8: -0.25}]


def test_logprob_arrays_slice_concat_and_extend():
    rows = _rows([1, 2, 3])
    assert len(rows) == 3 and bool(rows)
    assert rows[:-1].ids[:, 0].tolist() == [1, 2]
    assert rows[-1:].ids[:, 0].tolist() == [3]
    assert not rows[3:]
    with pytest.raises(TypeError):
        rows[0]
    with pytest.raises(TypeError):
        [].extend(rows)
    assert LogprobArrays.concat([rows]) is rows

    resp = Response(text='a', generate_token_len=1, input_token_len=1, logprobs=rows[:1])
    resp.extend(Response(text='b', generate_token_len=3, input_token_len=1, logprobs=rows[1:]))
    np.testing.assert_array_equal(resp.logprobs.ids, rows.ids)


def test_generation_config_logprobs_format():
    assert GenerationConfig().logprobs_format == 'dict'
    assert SamplingParam.from_gen_config(GenerationConfig(logprobs=3,
                                                          logprobs_format='array')).logprobs_format == 'array'
    with pytest.raises(ValueError):
        GenerationConfig(logprobs_format='flat')
    with pytest.raises(ValueError):
        GenerationConfig(logprobs=1, logprob_start_len=0, logprobs_format='array')
