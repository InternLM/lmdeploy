# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import suppress

from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool, PDConnectionStatus
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage


def _msg(p_url='http://p', d_url='http://d'):
    return PDConnectionMessage(p_url=p_url, d_url=d_url)


async def _close_pool(pool: PDConnectionPool):
    if pool._perform_conn_task is not None:
        pool._perform_conn_task.cancel()
        with suppress(asyncio.CancelledError):
            await pool._perform_conn_task
    for task in list(pool._bg_tasks):
        task.cancel()
    if getattr(pool, 'conn_sess', None) is not None:
        await pool.conn_sess.close()


def _wait_for_conn_tasks():
    return [task for task in asyncio.all_tasks() if 'wait_for_conn' in getattr(task.get_coro(), '__qualname__', '')]


def _worker_tasks():
    return [task for task in asyncio.all_tasks() if task.get_name().startswith('pd-conn-worker:')]


async def _run_coalesce_same_link_starts_one_handshake():
    pool = PDConnectionPool()
    handshake_started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def fake_handshake(conn_req):
        calls.append((conn_req.p_url, conn_req.d_url))
        handshake_started.set()
        await release.wait()

    pool._handshake = fake_handshake

    try:
        tasks = [asyncio.create_task(pool.connect(_msg())) for _ in range(32)]
        await asyncio.wait_for(handshake_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert len(calls) == 1
        assert pool.waiting_conn.qsize() == 0
        assert pool.pool[('http://p', 'http://d')].status == PDConnectionStatus.Connecting
        assert len(_worker_tasks()) == 1
        assert _wait_for_conn_tasks() == []

        release.set()
        await asyncio.gather(*tasks)
        assert pool.is_connected('http://p', 'http://d')
        assert len(calls) == 1
    finally:
        await _close_pool(pool)


def test_coalesce_same_link_starts_one_handshake():
    asyncio.run(_run_coalesce_same_link_starts_one_handshake())


async def _run_connect_wait_timeout_does_not_spawn_duplicate_workers():
    pool = PDConnectionPool()
    pool.connect_wait_timeout = 0.05
    pool.max_retry_cnt = 8
    handshake_started = asyncio.Event()
    calls = []

    async def fake_handshake(conn_req):
        calls.append((conn_req.p_url, conn_req.d_url))
        handshake_started.set()
        await asyncio.sleep(3600)

    pool._handshake = fake_handshake

    try:
        tasks = [asyncio.create_task(pool.connect(_msg())) for _ in range(16)]
        await asyncio.wait_for(handshake_started.wait(), timeout=1)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(exc, TimeoutError) for exc in results)
        assert len(calls) == 1
        assert len(_worker_tasks()) == 1
        assert _wait_for_conn_tasks() == []
        assert pool.pool[('http://p', 'http://d')].status == PDConnectionStatus.Connecting
    finally:
        await _close_pool(pool)


def test_connect_wait_timeout_does_not_spawn_duplicate_workers():
    asyncio.run(_run_connect_wait_timeout_does_not_spawn_duplicate_workers())


async def _run_warmup_coalesces_concurrent_callers():
    pool = PDConnectionPool()
    handshake_started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def fake_handshake(conn_req):
        calls.append((conn_req.p_url, conn_req.d_url))
        handshake_started.set()
        await release.wait()

    pool._handshake = fake_handshake
    messages = [
        _msg('http://p1', 'http://d1'),
        _msg('http://p1', 'http://d2'),
        _msg('http://p2', 'http://d1'),
        _msg('http://p2', 'http://d2'),
    ]

    try:
        callers = [asyncio.create_task(pool.warmup_connections(messages, concurrency=8)) for _ in range(12)]
        await asyncio.wait_for(handshake_started.wait(), timeout=1)
        for _ in range(50):
            if len(calls) == len(messages):
                break
            await asyncio.sleep(0.01)
        assert pool._warmup_task is not None
        assert sum(1 for task in callers if not task.done()) == 12
        # Unique P-D pairs, not callers × pairs.
        assert sorted(set(calls)) == sorted((m.p_url, m.d_url) for m in messages)
        assert len(calls) == len(messages)
        assert len(_worker_tasks()) == len(messages)
        assert _wait_for_conn_tasks() == []

        release.set()
        await asyncio.gather(*callers)
        assert all(pool.is_connected(m.p_url, m.d_url) for m in messages)
        assert len(calls) == len(messages)
    finally:
        await _close_pool(pool)


def test_warmup_coalesces_concurrent_callers():
    asyncio.run(_run_warmup_coalesces_concurrent_callers())
