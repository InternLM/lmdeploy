import asyncio

import pytest

from lmdeploy.pytorch.engine.request import (RequestManager, RequestType,
                                             Response, ResponseType)


class TestRequestHander:

    @pytest.fixture
    def event_loop(self):
        old_loop = asyncio.get_event_loop()
        new_loop = asyncio.new_event_loop()
        yield new_loop
        new_loop.stop()
        asyncio.set_event_loop(old_loop)

    @pytest.fixture
    def thread_safe(self, request):
        yield request.param

    @pytest.fixture
    def manager(self, thread_safe):
        yield RequestManager(thread_safe=thread_safe)

    @pytest.mark.parametrize('thread_safe', [True, False])
    def test_bind(self, manager, event_loop):

        def __stop_engine_callback(reqs, **kwargs):
            for req in reqs:
                manager.response(
                    Response(type=ResponseType.SUCCESS,
                             sender_id=req.sender_id,
                             req_id=req.req_id,
                             data=f'{req.data} success'))

        async def __dummy_loop():
            while True:
                manager.step()
                await asyncio.sleep(0.1)

        asyncio.set_event_loop(event_loop)
        sender = manager.build_sender()
        manager.start_loop(__dummy_loop)

        # test not bind
        req_id = sender.send_async(RequestType.STOP_ENGINE, None)
        resp = sender.recv(req_id)
        assert resp.type == ResponseType.HANDLER_NOT_EXIST

        assert manager.is_loop_alive()

        # test bind success
        sender.send_async(RequestType.STOP_ENGINE, None)
        manager.bind_func(RequestType.STOP_ENGINE, __stop_engine_callback)
        req_id = sender.send_async(RequestType.STOP_ENGINE, 'test')
        resp = sender.recv(req_id)
        assert resp.data == 'test success'
