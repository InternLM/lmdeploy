import pytest

from lmdeploy.pytorch.engine.request import (RequestManager, RequestType,
                                             ResponseType)


class TestRequestHander:

    @pytest.fixture
    def manager(self):
        yield RequestManager()

    def _stop_engine_callback(self, reqs, **kwargs):
        raise RuntimeError('stop_engine')

    def test_bind(self, manager):
        sender = manager.build_sender()

        # test not bind
        req_id = sender.send_async(RequestType.STOP_ENGINE, None)
        manager.step()
        resp = sender.recv(req_id)
        assert resp.type == ResponseType.HANDLER_NOT_EXIST

        # test bind success
        sender.send_async(RequestType.STOP_ENGINE, None)
        manager.bind_func(RequestType.STOP_ENGINE, self._stop_engine_callback)
        with pytest.raises(RuntimeError):
            manager.step()
