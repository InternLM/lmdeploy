"""Regression tests for DistServe Prefill cache-free session identity.

The OpenAI response ``id`` is the client session_id (default -1). DistServe
cache-free looks up ``scheduler.sessions[remote_session_id]``, so using that
public id misses the real Prefill session and leaks preserved metadata.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import APIRouter

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.config import EngineRole, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.paging.seq_states.states import ToBeMigratedState
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.serve.openai.endpoints.completions import register
from lmdeploy.serve.openai.protocol import CompletionRequest
from lmdeploy.serve.proxy import proxy as proxy_mod


class _Session:

    def __init__(self, session_id):
        self.session_id = session_id

    async def async_abort(self):
        pass


class _SessionManager:

    def __init__(self):
        self.removed = []

    def has(self, session_id):
        return False

    def remove(self, session):
        self.removed.append(session)


class _RawRequest:

    def __init__(self, payload=None):
        self._payload = payload or {}

    async def json(self):
        return dict(self._payload)

    async def is_disconnected(self):
        return False


def _scheduler():
    return Scheduler(
        SchedulerConfig(max_batches=4, max_session_len=128, max_request_output_len=64),
        CacheConfig(max_batches=4, block_size=16, num_cpu_blocks=0, num_gpu_blocks=16),
        SequenceMeta(16, strategy=ARSequenceStrategy()),
    )


def _cache_free(scheduler, session_id):
    """Mirror EngineP2PConnection.handle_zmq_recv cache-free lookup."""
    if session_id in scheduler.sessions:
        scheduler.end_session(session_id)
        return True
    return False


def test_public_completion_id_misses_preserved_prefill_session():
    scheduler = _scheduler()
    session = scheduler.add_session(7)
    seq = session.add_sequence([1, 2, 3], preserve_cache=True)
    # Prefill finish with preserve_cache keeps the session for migration.
    seq.state.to_state(ToBeMigratedState)

    public_id = -1
    assert not _cache_free(scheduler, public_id)
    assert 7 in scheduler.sessions
    assert seq.seq_id in scheduler.seq_manager._seq_map

    assert _cache_free(scheduler, 7)
    assert 7 not in scheduler.sessions
    assert seq.seq_id not in scheduler.seq_manager._seq_map


def test_completions_with_cache_returns_engine_session_id_not_public_id():
    engine_session_id = 7

    class _AsyncEngine:
        model_name = 'fake-model'

        async def preprocess(self, prompt, session, **kwargs):
            return SimpleNamespace(session=session, gen_config=kwargs['gen_config'])

        async def generate(self, prepared, **kwargs):
            yield SimpleNamespace(response='ok',
                                  token_ids=[1],
                                  input_token_len=2,
                                  generate_token_len=1,
                                  finish_reason='stop',
                                  logprobs=None,
                                  cache_block_ids=[0],
                                  cached_tokens=0)

    class _ServerContext:

        def __init__(self):
            self.async_engine = _AsyncEngine()
            self.engine_config = SimpleNamespace(logprobs_mode=None, adapters=[])
            self.session_manager = _SessionManager()
            self.default_gen_config = {}

        def create_session(self, session_id=None):
            return _Session(engine_session_id)

    async def _run():
        context = _ServerContext()
        router = APIRouter()
        register(router, context)
        endpoint = router.routes[0].endpoint
        return await endpoint(
            CompletionRequest(model='fake-model', prompt='hi', session_id=-1, max_tokens=1),
            _RawRequest({'with_cache': True, 'preserve_cache': True}),
        )

    response = asyncio.run(_run())
    assert response['id'] == '-1'
    assert response['cache_session_id'] == engine_session_id


def _proxy_manager(generate, stream_generate=None, pool=None):
    pool = pool or PDConnectionPool()
    pool.is_connected = lambda *args: True

    async def check_model(model):
        return None

    return SimpleNamespace(
        serving_strategy=ServingStrategy.DistServe,
        check_request_model=check_model,
        dummy_prefill=False,
        migration_protocol=MigrationProtocol.RDMA,
        rdma_config=None,
        pd_connection_pool=pool,
        generate=generate,
        stream_generate=stream_generate,
        get_node_url=lambda model, role: 'p' if role == EngineRole.Prefill else 'd',
        pre_call=lambda url: 0,
        post_call=lambda *args: None,
        create_background_tasks=lambda *args: None,
    )


@pytest.mark.parametrize('stream', [False, True])
def test_proxy_completions_uses_cache_session_id_for_free(monkeypatch, stream):
    owners = []
    pool = PDConnectionPool()
    pool.is_connected = lambda *args: True
    pool.shelf_prefill_session = Mock(wraps=pool.shelf_prefill_session)
    pool.unshelf_prefill_session = Mock(wraps=pool.unshelf_prefill_session)

    async def generate(payload, url, endpoint):
        if url == 'd':
            owners.append(payload['migration_request']['remote_session_id'])
            return json.dumps({'choices': [{'finish_reason': 'stop'}]})
        return json.dumps({
            'id': '-1',
            'cache_session_id': 7,
            'cache_block_ids': [0],
            'remote_token_ids': [1],
        })

    async def stream_generate(payload, url, endpoint):
        owners.append(payload['migration_request']['remote_session_id'])
        yield b'data: {"choices": [{"finish_reason": "stop"}]}\n\n'
        yield b'data: [DONE]\n\n'

    monkeypatch.setattr(
        proxy_mod, 'node_manager',
        _proxy_manager(generate, stream_generate=stream_generate, pool=pool))

    async def _run():
        response = await proxy_mod.completions_v1(
            CompletionRequest(model='fake-model', prompt='hi', max_tokens=1, stream=stream))
        if stream:
            async for _ in response.body_iterator:
                pass
        return response

    response = asyncio.run(_run())
    assert response.status_code == 200
    assert owners == [7]
    pool.shelf_prefill_session.assert_called_once_with(('p', 'd'), 7)
    pool.unshelf_prefill_session.assert_called_once_with(('p', 'd'), 7)


@pytest.mark.parametrize('owner', [None, -1, '-1', True])
def test_proxy_completions_rejects_public_id_fallback(monkeypatch, owner):
    decode_calls = []
    pool = PDConnectionPool()
    pool.is_connected = lambda *args: True

    async def generate(payload, url, endpoint):
        decode_calls.append(url)
        result = {'id': '-1', 'cache_block_ids': [0], 'remote_token_ids': [1]}
        if owner is not None:
            result['cache_session_id'] = owner
        return json.dumps(result)

    monkeypatch.setattr(proxy_mod, 'node_manager', _proxy_manager(generate, pool=pool))

    async def _run():
        return await proxy_mod.completions_v1(CompletionRequest(model='fake-model', prompt='hi', max_tokens=1))

    response = asyncio.run(_run())
    assert response.status_code == 502
    assert b'cache_session_id' in response.body
    assert decode_calls == ['p']
    assert not pool.migration_session_shelf
