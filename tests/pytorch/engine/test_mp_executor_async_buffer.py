# Copyright (c) OpenMMLab. All rights reserved.
"""Ring-wrap waits must let the model event loop make progress."""
import asyncio
import copy
import multiprocessing as mp
import threading

import pytest

from lmdeploy.pytorch.engine.executor import mp_executor as mod


@pytest.mark.parametrize('operation', ['set_async', 'wait_async'])
@pytest.mark.parametrize('blocked_phase', [1, 2])
def test_wrap_barrier_keeps_event_loop_live(operation, blocked_phase):
    release = threading.Event()
    calls = []

    class Barrier:
        def wait(self):
            calls.append(len(calls) + 1)
            if len(calls) == blocked_phase:
                # Bounded failure on old code, rather than hanging the test runner.
                assert release.wait(1), 'barrier blocked the model event loop'

    async def run():
        notifier = mod.Notifier(1, mp.get_context('spawn'))
        notifier._event_id = mod.NUM_SHARED_BLOCK - 1
        notifier.events[-1].set()
        notifier.bar = Barrier()

        async def model_progress():
            while len(calls) < blocked_phase:
                await asyncio.sleep(.001)
            release.set()

        task = asyncio.create_task(model_progress())
        try:
            if operation == 'set_async':
                await notifier.set_async()
            else:
                async with notifier.wait_async():
                    pass
            await task
            assert calls == [1, 2]
            assert notifier._event_id == 0
        finally:
            release.set()
            task.cancel()

    asyncio.run(run())


@pytest.mark.parametrize('receiver_mask', [1, 0])
def test_async_messages_cross_wrap_without_interleaving(monkeypatch, receiver_mask):
    # Small real shared-memory ring, real multiprocessing barriers, independent
    # reader/writer cursors; concurrent sends and multi-packet messages.
    monkeypatch.setattr(mod, 'NUM_SHARED_BLOCK', 2)
    monkeypatch.setattr(mod, 'SHARED_BLOCK_SIZE', 64)
    monkeypatch.setattr(mod, 'SHARED_BLOCK_REAL_SIZE', 72)
    notifier = mod.Notifier(1, mp.get_context('spawn'))
    writer = mod.SharedBuffer(-1, notifier)
    reader = mod.SharedBuffer(0, copy.copy(notifier), writer.name())
    watchdog = threading.Timer(10, notifier.bar.abort)
    watchdog.start()
    payloads = [(i, bytes([i]) * 300) for i in range(8)]

    async def run():
        async def receive():
            return [await reader.receive_async() for _ in payloads]

        receiver = asyncio.create_task(receive())
        await asyncio.gather(*(writer.send_async(p, receiver_mask) for p in payloads))
        assert await receiver == (payloads if receiver_mask else [None] * len(payloads))
        assert writer._buf_id == reader._buf_id
        assert writer.notifier._event_id == reader.notifier._event_id

    try:
        asyncio.run(run())
    finally:
        watchdog.cancel()
        reader.close()
        writer.close()


def _receive_in_process(rank, notifier, name, results):
    mod.NUM_SHARED_BLOCK = 2
    mod.SHARED_BLOCK_SIZE = 64
    mod.SHARED_BLOCK_REAL_SIZE = 72
    reader = mod.SharedBuffer(rank, notifier, name)

    async def run():
        return [await reader.receive_async() for _ in range(6)]

    try:
        results.put((rank, asyncio.run(run())))
    finally:
        # The parent owns barrier shutdown, after every receiver has exited.
        reader.shm.close()


def test_async_broadcast_wrap_across_spawned_processes(monkeypatch):
    monkeypatch.setattr(mod, 'NUM_SHARED_BLOCK', 2)
    monkeypatch.setattr(mod, 'SHARED_BLOCK_SIZE', 64)
    monkeypatch.setattr(mod, 'SHARED_BLOCK_REAL_SIZE', 72)
    ctx = mp.get_context('spawn')
    notifier = mod.Notifier(2, ctx)
    writer = mod.SharedBuffer(-1, notifier)
    results = ctx.Queue()
    children = [ctx.Process(target=_receive_in_process, args=(rank, notifier, writer.name(), results))
                for rank in range(2)]
    watchdog = threading.Timer(40, notifier.bar.abort)
    payloads = [(i, bytes([i]) * 300) for i in range(6)]

    async def send():
        for i, payload in enumerate(payloads):
            await writer.send_async(payload, receiver_mask=3 if i % 2 == 0 else 1)

    try:
        for child in children:
            child.start()
        watchdog.start()
        asyncio.run(send())
        received = dict(results.get(timeout=10) for _ in children)
        assert received[0] == payloads
        assert received[1] == [x if i % 2 == 0 else None for i, x in enumerate(payloads)]
        for child in children:
            child.join(10)
            assert child.exitcode == 0
    finally:
        watchdog.cancel()
        notifier.close()
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(10)
        writer.close()
        results.close()
        results.join_thread()


@pytest.mark.parametrize('large_message', [False, True])
@pytest.mark.parametrize('receiver_mask', [1, 0])
def test_sync_async_rpc_share_one_writer(monkeypatch, large_message, receiver_mask):
    monkeypatch.setattr(mod, 'NUM_SHARED_BLOCK', 2)
    monkeypatch.setattr(mod, 'SHARED_BLOCK_SIZE', 256)
    monkeypatch.setattr(mod, 'SHARED_BLOCK_REAL_SIZE', 264)
    notifier = mod.Notifier(1, mp.get_context('spawn'))
    writer = mod.SharedBuffer(-1, notifier)
    reader = mod.SharedBuffer(0, copy.copy(notifier), writer.name())
    writer._buf_id = reader._buf_id = 1
    writer.notifier._event_id = reader.notifier._event_id = 1
    executor = mod.MPExecutor.__new__(mod.MPExecutor)
    executor.comm_buf = writer
    watchdog = threading.Timer(10, notifier.bar.abort)
    watchdog.start()
    payload = 'x' * (1200 if large_message else 1)

    async def run():
        first = asyncio.create_task(executor.collective_rpc_async(
            'forward_async', args=(payload,), receiver_mask=receiver_mask, return_mask=0))
        while notifier.bar.n_waiting != 1:
            await asyncio.sleep(.001)
        second = asyncio.create_task(asyncio.to_thread(
            executor.collective_rpc, 'get_checkpoint_engine_status', None, None, receiver_mask, 0))
        await asyncio.sleep(.05)
        # Without a reader neither message may pass ring reclamation. Previously
        # the two senders satisfied the barrier themselves and overwrote forward.
        assert not first.done() and not second.done()
        assert writer._buf_id == 1 and notifier._event_id == 1
        received = [await reader.receive_async() for _ in range(2)]
        await asyncio.gather(first, second)
        if receiver_mask:
            assert [r['method'] for r in received] == ['forward_async', 'get_checkpoint_engine_status']
            assert received[0]['args'] == (payload,)
        else:
            assert received == [None, None]

    try:
        asyncio.run(run())
    finally:
        watchdog.cancel()
        writer.close()
        reader.close()


def test_cancelled_sender_does_not_release_message_ownership(monkeypatch):
    monkeypatch.setattr(mod, 'NUM_SHARED_BLOCK', 2)
    notifier = mod.Notifier(1, mp.get_context('spawn'))
    writer = mod.SharedBuffer(-1, notifier)
    reader = mod.SharedBuffer(0, copy.copy(notifier), writer.name())
    writer._buf_id = reader._buf_id = 1
    writer.notifier._event_id = reader.notifier._event_id = 1
    watchdog = threading.Timer(10, notifier.bar.abort)
    watchdog.start()

    async def run():
        first = asyncio.create_task(writer.send_async('first'))
        while notifier.bar.n_waiting != 1:
            await asyncio.sleep(.001)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, .5)
        second = asyncio.create_task(writer.send_async('second'))
        assert await reader.receive_async() == 'first'
        assert await reader.receive_async() == 'second'
        await second

    try:
        asyncio.run(run())
    finally:
        watchdog.cancel()
        writer.close()
        reader.close()


def test_close_aborts_active_and_queued_senders(monkeypatch):
    monkeypatch.setattr(mod, 'NUM_SHARED_BLOCK', 2)
    notifier = mod.Notifier(1, mp.get_context('spawn'))
    writer = mod.SharedBuffer(-1, notifier)
    writer._buf_id = notifier._event_id = 1

    async def run():
        first = asyncio.create_task(writer.send_async('no receiver'))
        while notifier.bar.n_waiting != 1:
            await asyncio.sleep(.001)
        second = asyncio.create_task(asyncio.to_thread(writer.send, 'queued sync sender'))
        await asyncio.sleep(.02)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, .5)
        await asyncio.wait_for(asyncio.to_thread(writer.close), 2)
        with pytest.raises((asyncio.CancelledError, threading.BrokenBarrierError, RuntimeError)):
            await asyncio.wait_for(second, 2)
        assert all(not t.is_alive() for t in writer._writer._threads)
        with pytest.raises(RuntimeError, match='closed'):
            writer.send('after close')
        with pytest.raises(RuntimeError, match='closed'):
            await writer.send_async('after close')

    try:
        asyncio.run(run())
    finally:
        writer.close()
