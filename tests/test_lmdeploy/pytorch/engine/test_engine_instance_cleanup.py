import asyncio

import numpy as np

from lmdeploy.messages import ResponseType
from lmdeploy.pytorch.engine.engine_instance import EngineInstance
from lmdeploy.pytorch.engine.mp_engine.base import MPEngineInstance, SessionState
from lmdeploy.pytorch.engine.request import RequestType, Response


class _FakeReqSender:

    def __init__(self, responses):
        self.responses = list(responses)
        self.sent_async = []
        self.sender_id = 0

    def send_async(self, req_type, data):
        self.sent_async.append((req_type, data))
        return object()

    async def async_recv(self, resp, wait_main=True):
        return self.responses.pop(0)


class _FakeReqManager:

    def __init__(self):
        self.senders = {0: object()}


class _FakeEngine:
    max_session_len = 1024

    class _Config:
        enable_transfer_obj_ref = False
        distributed_executor_backend = None

    engine_config = _Config()

    def __init__(self):
        self.req_manager = _FakeReqManager()


def _response(resp_type, data=None):
    return Response(resp_type, 0, asyncio.Event(), data=data)


async def _consume(generator):
    outputs = []
    async for item in generator:
        outputs.append(item)
    return outputs


def test_engine_instance_ends_session_after_finish():
    req_sender = _FakeReqSender([
        _response(ResponseType.FINISH, data={'token_ids': np.array([10, 11])}),
    ])
    instance = EngineInstance.__new__(EngineInstance)
    instance.engine = _FakeEngine()
    instance.req_sender = req_sender
    instance.max_input_len = 1024
    instance._enable_transfer_obj_ref = False

    outputs = asyncio.run(_consume(instance.async_stream_infer(7, [1, 2])))

    assert outputs[-1].status == ResponseType.FINISH
    assert (RequestType.END_SESSION, {'session_id': 7, 'response': False}) in req_sender.sent_async


def test_engine_instance_ends_session_when_generator_is_closed():
    req_sender = _FakeReqSender([
        _response(ResponseType.SUCCESS, data={'token_ids': np.array([10])}),
    ])
    instance = EngineInstance.__new__(EngineInstance)
    instance.engine = _FakeEngine()
    instance.req_sender = req_sender
    instance.max_input_len = 1024
    instance._enable_transfer_obj_ref = False

    async def _run():
        gen = instance.async_stream_infer(8, [1, 2])
        first = await gen.__anext__()
        await gen.aclose()
        return first

    first = asyncio.run(_run())

    assert first.status == ResponseType.SUCCESS
    assert (RequestType.END_SESSION, {'session_id': 8, 'response': False}) in req_sender.sent_async


class _FakeMPEngine:

    def __init__(self):
        self.session_states = {9: SessionState()}
        self.pending_cancel_sessions = set()

    async def _collective_rpc_streaming_async(self, func, init_done, *args, **kwargs):
        self.pending_cancel_sessions.add(kwargs['session_id'])
        init_done.set()
        yield 'done'


def test_mp_engine_instance_clears_local_session_state_after_stream():
    engine = _FakeMPEngine()
    instance = MPEngineInstance(engine)

    outputs = asyncio.run(_consume(instance.async_stream_infer(9, input_ids=[1])))

    assert outputs == ['done']
    assert 9 not in engine.session_states
    assert 9 not in engine.pending_cancel_sessions
