# Copyright (c) OpenMMLab. All rights reserved.
"""A bad DistServe migration must fail the request, not EngineLoop."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from lmdeploy.messages import ResponseType
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.engine.engine_loop import EngineLoop


def _migration_msg(session_id, *, remote_block_ids, dummy=False):
    return SimpleNamespace(
        session_id=session_id,
        num_token_ids=9,
        migration_request=MigrationRequest(
            protocol=MigrationProtocol.NVLINK,
            remote_engine_id='http://127.0.0.1:19001',
            remote_session_id=123456789,
            remote_token_id=1,
            remote_block_ids=remote_block_ids,
            is_dummy_prefill=dummy,
        ),
        resp=SimpleNamespace(type=None, is_done=False, err_msg=''),
    )


def _loop_with_blocks(decode_block_ids, ended):
    loop = object.__new__(EngineLoop)
    loop.scheduler = SimpleNamespace(
        sessions={1: object(), 2: object()},
        get_block_tables=lambda seqs: [list(decode_block_ids)],
        end_session=lambda session_id: ended.append(session_id),
    )
    loop.executor = SimpleNamespace(migrate=lambda inputs: (_ for _ in ()).throw(AssertionError('migrate')))
    loop.engine_conn = SimpleNamespace(zmq_send=lambda **kwargs: (_ for _ in ()).throw(AssertionError('zmq')))
    loop.resp_queue = asyncio.Queue()
    return loop


def test_mismatched_remote_block_ids_fail_request_not_loop():
    async def _run():
        ended = []
        loop = _loop_with_blocks([0], ended)
        bad = _migration_msg(1, remote_block_ids=[])
        dummy = _migration_msg(2, remote_block_ids=[], dummy=True)
        succeeded = await loop._migration_loop_migrate([bad, dummy])
        return loop, ended, bad, dummy, succeeded

    loop, ended, bad, dummy, succeeded = asyncio.run(_run())
    assert succeeded == [dummy]
    assert ended == [1]
    assert bad.resp.type == ResponseType.INTERNAL_ENGINE_ERROR
    assert bad.resp.is_done is True
    assert 'prefill block ids' in bad.resp.err_msg
    failed = loop.resp_queue.get_nowait()
    assert 1 in failed
    assert dummy.resp.type is None
