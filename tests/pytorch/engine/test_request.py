# yapf: disable
import asyncio

import pytest

from lmdeploy.pytorch.engine.request import RequestManager, RequestType, ResponseType

# yapf: enable


class TestRequestHander:

    @pytest.fixture
    def event_loop(self):
        old_loop = asyncio.get_event_loop()
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            yield new_loop
        finally:
            new_loop.stop()
            asyncio.set_event_loop(old_loop)

    @pytest.fixture
    def manager(self):
        yield RequestManager()

    def test_bind(self, manager, event_loop):

        def __stop_engine_callback(reqs, **kwargs):
            for req in reqs:
                resp = req.resp
                resp.type = ResponseType.SUCCESS
                resp.data = f'{req.data} success'
                manager.response(resp)

        async def __dummy_loop():
            while True:
                try:
                    await manager.step()
                except Exception:
                    return

        sender = manager.build_sender()
        manager.set_main_loop_func(__dummy_loop)

        # test not bind
        resp = sender.send_async(RequestType.STOP_ENGINE, None)
        resp = sender.recv(resp)
        assert resp.type == ResponseType.HANDLER_NOT_EXIST

        assert manager.is_loop_alive()

        # test bind success
        sender.send_async(RequestType.STOP_ENGINE, None)
        manager.bind_func(RequestType.STOP_ENGINE, __stop_engine_callback)
        resp = sender.send_async(RequestType.STOP_ENGINE, 'test')
        resp = sender.recv(resp)
        assert resp.data == 'test success'

        # cleanup, cancel main task
        task_to_cancel = manager._loop_task
        manager.stop_loop()
        asyncio.run
        event_loop.run_until_complete(asyncio.gather(task_to_cancel, return_exceptions=True))

    def test_blocked_add_requests_are_cancelled_immediately(self, manager):
        sender = manager.build_sender()
        manager.block_request_types({RequestType.ADD_SESSION, RequestType.ADD_MESSAGE})

        add_session_resp = sender.send_async(RequestType.ADD_SESSION, dict(session_id=1))
        add_message_resp = sender.send_async(RequestType.ADD_MESSAGE, dict(session_id=1))

        assert add_session_resp.type == ResponseType.CANCEL
        assert add_session_resp.is_done
        assert add_session_resp.event.is_set()
        assert add_message_resp.type == ResponseType.CANCEL
        assert add_message_resp.is_done
        assert add_message_resp.event.is_set()
        assert manager._loop_task is None

    def test_cleanup_requests_are_allowed_while_add_requests_blocked(self, manager, event_loop):

        def __success_callback(reqs, **kwargs):
            for req in reqs:
                req.resp.type = ResponseType.SUCCESS
                manager.response(req.resp)

        async def __dummy_loop():
            while True:
                try:
                    await manager.step()
                except Exception:
                    return

        sender = manager.build_sender()
        manager.set_main_loop_func(__dummy_loop)
        manager.bind_func(RequestType.STOP_SESSION, __success_callback)
        manager.bind_func(RequestType.END_SESSION, __success_callback)
        manager.block_request_types({RequestType.ADD_SESSION, RequestType.ADD_MESSAGE})

        stop_resp = sender.send(RequestType.STOP_SESSION, dict(session_id=1))
        end_resp = sender.send(RequestType.END_SESSION, dict(session_id=1))

        assert stop_resp.type == ResponseType.SUCCESS
        assert end_resp.type == ResponseType.SUCCESS

        task_to_cancel = manager._loop_task
        manager.stop_loop()
        event_loop.run_until_complete(asyncio.gather(task_to_cancel, return_exceptions=True))

    def test_queued_add_request_is_cancelled_when_blocked_before_processing(self, manager, event_loop):

        async def __idle_loop():
            await asyncio.Event().wait()

        sender = manager.build_sender()
        manager.set_main_loop_func(__idle_loop)
        resp = sender.send_async(RequestType.ADD_SESSION, dict(session_id=1))

        manager.block_request_types({RequestType.ADD_SESSION})
        event_loop.run_until_complete(manager.step())

        assert resp.type == ResponseType.CANCEL
        assert resp.is_done
        assert resp.event.is_set()

        task_to_cancel = manager._loop_task
        manager.stop_loop()
        event_loop.run_until_complete(asyncio.gather(task_to_cancel, return_exceptions=True))

    def test_unblock_request_types_restores_normal_processing(self, manager, event_loop):

        def __success_callback(reqs, **kwargs):
            for req in reqs:
                req.resp.type = ResponseType.SUCCESS
                manager.response(req.resp)

        async def __dummy_loop():
            while True:
                try:
                    await manager.step()
                except Exception:
                    return

        sender = manager.build_sender()
        manager.block_request_types({RequestType.ADD_SESSION})
        blocked_resp = sender.send_async(RequestType.ADD_SESSION, dict(session_id=1))
        assert blocked_resp.type == ResponseType.CANCEL

        manager.unblock_request_types({RequestType.ADD_SESSION})
        manager.set_main_loop_func(__dummy_loop)
        manager.bind_func(RequestType.ADD_SESSION, __success_callback)
        resp = sender.send(RequestType.ADD_SESSION, dict(session_id=1))

        assert resp.type == ResponseType.SUCCESS

        task_to_cancel = manager._loop_task
        manager.stop_loop()
        event_loop.run_until_complete(asyncio.gather(task_to_cancel, return_exceptions=True))
