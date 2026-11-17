import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
import torch
from fastapi import APIRouter, FastAPI

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.metrics.metrics_processor import metrics_processor
from lmdeploy.metrics.stats import SchedulerStats
from lmdeploy.serve.core.async_engine import AsyncEngine
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.endpoints import auxiliary


@pytest.fixture
def scoring_app(monkeypatch):
    monkeypatch.setattr(metrics_processor, 'scheduler_stats', SchedulerStats())

    async def infer(*args, **kwargs):
        yield EngineOutput(ResponseType.FINISH, [], logits=torch.tensor([[1.0], [2.0], [3.0]]), ce_loss=3.0)

    handle = SimpleNamespace(async_stream_infer=Mock(side_effect=infer), async_cancel=AsyncMock())
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.epoch = 0
    engine.backend = 'pytorch'
    engine.backend_config = SimpleNamespace(enable_prefix_caching=False)
    engine.speculative_config = None
    engine.arch = 'InternLM2ForRewardModel'
    engine.model_name = 'test-reward-model'
    engine.session_mgr = SessionManager()
    engine.session_mgr.build_request_handle_pool(SimpleNamespace(create_instance=lambda: handle), 1)
    app, router = FastAPI(), APIRouter()
    auxiliary.register(router, SimpleNamespace(async_engine=engine))
    app.include_router(router)
    return app, engine, handle


@pytest.mark.parametrize('endpoint', ['/get_ppl', '/pooling'])
@pytest.mark.parametrize('operation', ['abort', 'abort_all', 'close'])
def test_scoring_cancelled_while_waiting_for_handle_returns_499(endpoint, operation, scoring_app, monkeypatch):
    async def run():
        app, engine, handle = scoring_app
        manager = engine.session_mgr
        pool = manager.request_handle_pool
        held = await pool.get()
        waiting = asyncio.Event()
        original_get = pool.get

        async def get():
            waiting.set()
            return await original_get()

        monkeypatch.setattr(pool, 'get', get)
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            task = asyncio.create_task(client.post(endpoint, json={'input': [1, 2, 3]}))
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                session, = manager.sessions.values()
                if operation == 'abort_all':
                    await engine.stop_all_session()
                elif operation == 'close':
                    await session.async_close()
                else:
                    await session.async_abort()
            finally:
                pool.put(held)
            response = await asyncio.wait_for(task, 5)

        assert response.status_code == 499
        assert response.json() == {
            'message': 'The request was cancelled.',
            'type': 'invalid_request_error',
            'code': 499,
            'param': None,
            'object': 'error',
        }
        handle.async_stream_infer.assert_not_called()
        handle.async_cancel.assert_not_awaited()
        assert pool.num_dispatched == 0
        assert manager.sessions == {}

    asyncio.run(run())


@pytest.mark.parametrize('endpoint', ['/get_ppl', '/pooling'])
def test_scoring_success_returns_200(endpoint, scoring_app):
    async def run():
        app, engine, handle = scoring_app
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            response = await client.post(endpoint, json={'input': [1, 2, 3]})
        assert response.status_code == 200
        if endpoint == '/get_ppl':
            assert response.json() == {'ppl': 1.5}
        else:
            assert response.json()['data'] == [{'index': 0, 'object': 'pooling', 'data': 3.0}]
        handle.async_cancel.assert_not_awaited()
        assert engine.session_mgr.request_handle_pool.num_dispatched == 0
        assert engine.session_mgr.sessions == {}

    asyncio.run(run())


@pytest.mark.parametrize('endpoint', ['/get_ppl', '/pooling'])
def test_scoring_request_error_preserves_status_and_message(endpoint, scoring_app, monkeypatch):
    async def run():
        app, engine, _ = scoring_app
        method = 'async_get_ppl' if endpoint == '/get_ppl' else 'async_get_reward_score'
        error = RequestError(ErrorCode.ENGINE_UNAVAILABLE, 'Scoring engine unavailable.')
        monkeypatch.setattr(engine, method, AsyncMock(side_effect=error))
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            response = await client.post(endpoint, json={'input': [1, 2, 3]})
        assert response.status_code == 503
        assert response.json()['code'] == 503
        assert response.json()['message'] == error.message

    asyncio.run(run())


@pytest.mark.parametrize('prefix_caching', [False, True])
def test_get_ppl_value_error_returns_400(prefix_caching, scoring_app):
    async def run():
        app, engine, handle = scoring_app
        engine.backend_config.enable_prefix_caching = prefix_caching
        input_ids = [1, 2, 3] if prefix_caching else [1]
        expected = 'prefix caching' if prefix_caching else 'at least 2 tokens'
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            response = await client.post('/get_ppl', json={'input': input_ids})
        assert response.status_code == 400
        assert response.json()['code'] == 400
        assert expected in response.json()['message']
        handle.async_stream_infer.assert_not_called()

    asyncio.run(run())
