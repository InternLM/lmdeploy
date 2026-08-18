# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from types import SimpleNamespace

import numpy as np

from lmdeploy.messages import ResponseType
from lmdeploy.pytorch.engine.engine import Engine, InferOutput
from lmdeploy.pytorch.engine.engine_loop import EngineLoop
from lmdeploy.pytorch.engine.request import Request, RequestType, Response
from lmdeploy.pytorch.messages import MessageStatus


class FakeReqManager:

    def __init__(self):
        self.rejected = []
        self.responses = []

    def reject_request(self, resp, req_type=None, reason=''):
        self.rejected.append((resp, req_type, reason))
        resp.type = ResponseType.CANCEL
        resp.is_done = True
        resp.event.set()

    def response(self, resp):
        self.responses.append(resp)
        resp.event.set()


class FakeState:

    def __init__(self, seq):
        self.seq = seq

    def stop(self):
        self.seq.status = MessageStatus.STOPPED


class FakeSeq:

    def __init__(self, status, resp):
        self.status = status
        self.resp = resp
        self.state = FakeState(self)


class FakeScheduler:

    def __init__(self, seq):
        self.sessions = {1: SimpleNamespace(sequences={0: seq})}

    def stop_session(self, session_id):
        for seq in self.sessions[session_id].sequences.values():
            seq.state.stop()


def make_response(status=ResponseType.INTERNAL_ENGINE_ERROR):
    return Response(type=status, sender_id=0, event=asyncio.Event())


def make_stop_request():
    return Request(
        type=RequestType.STOP_SESSION,
        sender_id=0,
        data=dict(session_id=1),
        resp=make_response(),
    )


def run_stop_session(seq_status):
    stream_resp = make_response()
    seq = FakeSeq(seq_status, stream_resp)
    req_manager = FakeReqManager()
    engine = SimpleNamespace(
        scheduler=FakeScheduler(seq),
        req_manager=req_manager,
        _response=lambda resp, resp_type: Engine._response(engine, resp, resp_type),
    )
    req = make_stop_request()

    Engine._on_stop_session(engine, [req])
    return stream_resp, req.resp, req_manager


def test_stopped_sequence_is_not_cancelled_by_abort():
    stream_resp, stop_resp, req_manager = run_stop_session(MessageStatus.STOPPED)

    assert stop_resp.type == ResponseType.SUCCESS
    assert stream_resp.type == ResponseType.INTERNAL_ENGINE_ERROR
    assert stream_resp.is_done is False
    assert stream_resp.event.is_set() is False
    assert req_manager.rejected == []


def test_to_be_migrated_sequence_is_not_cancelled_by_abort():
    stream_resp, stop_resp, req_manager = run_stop_session(MessageStatus.TO_BE_MIGRATED)

    assert stop_resp.type == ResponseType.SUCCESS
    assert stream_resp.type == ResponseType.INTERNAL_ENGINE_ERROR
    assert stream_resp.is_done is False
    assert stream_resp.event.is_set() is False
    assert req_manager.rejected == []


def test_running_sequence_is_still_cancelled_by_abort():
    stream_resp, stop_resp, req_manager = run_stop_session(MessageStatus.RUNNING)

    assert stop_resp.type == ResponseType.SUCCESS
    assert stream_resp.type == ResponseType.CANCEL
    assert stream_resp.is_done is True
    assert stream_resp.event.is_set() is True
    assert len(req_manager.rejected) == 1


def test_finish_output_wins_over_stale_cancel_response():
    req_manager = FakeReqManager()
    loop = SimpleNamespace(req_manager=req_manager)
    resp = make_response(ResponseType.CANCEL)
    resp.is_done = True
    out = InferOutput(session_id=1, resp=resp, token_ids=np.array([7]), finish=True)

    EngineLoop._send_resp(loop, out)

    assert resp.type == ResponseType.FINISH
    assert resp.data['token_ids'].tolist() == [7]
    assert req_manager.responses == [resp]


def test_cancel_output_without_finish_stays_cancelled():
    req_manager = FakeReqManager()
    loop = SimpleNamespace(req_manager=req_manager)
    resp = make_response(ResponseType.CANCEL)
    resp.is_done = True
    out = InferOutput(session_id=1, resp=resp, token_ids=np.array([]), finish=False)

    EngineLoop._send_resp(loop, out)

    assert resp.type == ResponseType.CANCEL
    assert resp.data['token_ids'].tolist() == []
    assert req_manager.responses == [resp]
