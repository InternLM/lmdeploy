"""MP worker command loop must execute sleep inline, before later commands."""
import asyncio

from lmdeploy.pytorch.engine.executor.mp_executor import ExecutorProc


def test_worker_command_loop_executes_sleep_inline_before_later_writes():
    async def scenario():
        proc = ExecutorProc.__new__(ExecutorProc)
        events = []
        started, release = asyncio.Event(), asyncio.Event()

        class FakeComm:
            def __init__(self):
                self._cmds = [
                    dict(method='sleep', args=[1], kwargs={}, return_mask=0),
                    dict(method='update_weights_from_ipc', args=(), kwargs={}, return_mask=0),
                ]

            async def receive_async(self):
                if self._cmds:
                    return self._cmds.pop(0)
                await asyncio.Event().wait()  # no further commands

        class FakeWorker:
            async def sleep(self, level=1):
                events.append('sleep:start')
                started.set()
                await release.wait()
                events.append('sleep:end')

            def update_weights_from_ipc(self, *args, **kwargs):
                events.append('update:start')

        task = asyncio.create_task(
            proc._main_loop_impl(0, comm_buf=FakeComm(), ret_buf=None, worker=FakeWorker()))
        await asyncio.wait_for(started.wait(), 5)
        # While the worker is still inside sleep, the command loop must not
        # have picked up (let alone started) the later weight write.
        await asyncio.sleep(0.05)
        assert events == ['sleep:start']
        release.set()
        await asyncio.sleep(0.05)
        assert events == ['sleep:start', 'sleep:end', 'update:start']
        task.cancel()
    asyncio.run(scenario())
