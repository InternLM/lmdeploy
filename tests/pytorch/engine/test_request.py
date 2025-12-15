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

        manager.stop_loop()
