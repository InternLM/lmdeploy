# Copyright (c) OpenMMLab. All rights reserved.
"""Cache-owner IDs travel from the real API to migration ACK and scheduler GC.

Only generation/transport are simulated; no model, CUDA, or RDMA is needed.
"""
import asyncio
import importlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi import APIRouter, FastAPI

from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig
from lmdeploy.pytorch.disagg.config import EngineRole, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.engine.engine import Engine
from lmdeploy.pytorch.messages import SequenceMeta
from lmdeploy.pytorch.paging.scheduler import Scheduler
from lmdeploy.pytorch.paging.seq_states.states import ToBeMigratedState
from lmdeploy.pytorch.strategies.ar.sequence import ARSequenceStrategy
from lmdeploy.serve.managers import SessionManager
from lmdeploy.serve.openai.api_server import ServerContext
from lmdeploy.serve.openai.chat_completions import register as register_chat
from lmdeploy.serve.openai.endpoints.completions import register as register_completion
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest, DeltaMessage


class _Parser:
    tool_parser_cls = None

    def __init__(self, request):
        self.request = request
        self.tool_parser = None
        self.reasoning_tokens = 0

    def stream_chunk(self, text, token_ids, **kwargs):
        return [(DeltaMessage(content=text), False)]

    def parse_complete(self, text, token_ids=None):
        return text, None, None

    def validate_complete(self, **kwargs):
        return True


@pytest.fixture
def prefill():
    scheduler = Scheduler(
        SchedulerConfig(max_batches=4, max_session_len=128, max_request_output_len=64),
        CacheConfig(max_batches=4, block_size=16, num_cpu_blocks=0, num_gpu_blocks=16),
        SequenceMeta(16, strategy=ARSequenceStrategy()))
    engine = object.__new__(Engine)
    engine.scheduler = scheduler

    class Generation:
        model_name = 'fake-model'
        epoch = 0
        backend_config = SimpleNamespace(role=EngineRole.Prefill, logprobs_mode=None, adapters=[])
        tokenizer = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(model=None)))

        def __init__(self):
            self.session_mgr = SessionManager()
            self.ids = []

        async def preprocess(self, prompt, session, **kwargs):
            return SimpleNamespace(session=session, gen_config=kwargs['gen_config'])

        async def generate(self, prepared, **kwargs):
            sid = prepared.session.session_id
            self.ids.append(sid)
            seq = scheduler.add_session(sid).add_sequence([1, 2], preserve_cache=prepared.gen_config.preserve_cache)
            seq.state.to_state(ToBeMigratedState)
            engine._on_end_session([SimpleNamespace(data={'session_id': sid, 'response': False})])
            yield SimpleNamespace(response='ok', token_ids=[1], input_token_len=2, generate_token_len=1,
                                  finish_reason='stop', logprobs=None, cache_block_ids=[0], cached_tokens=0,
                                  routed_experts=None)

    context = ServerContext()
    context.async_engine = Generation()
    context.response_parser_cls = _Parser
    router = APIRouter()
    register_chat(router, context)
    register_completion(router, context)
    app = FastAPI()
    app.include_router(router)
    return context, engine, app


def _request(chat, **kwargs):
    if chat:
        return ChatCompletionRequest(model='fake-model', messages=[dict(role='user', content='hi')], **kwargs)
    return CompletionRequest(model='fake-model', prompt='hi', **kwargs)


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('user_id', [-1, 42])
@pytest.mark.parametrize('stream', [False, True])
def test_prefill_cache_handle_is_not_the_public_response_id(prefill, chat, user_id, stream):
    context, engine, app = prefill

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://prefill') as client:
            for _ in range(3):
                payload = _request(chat, session_id=user_id, stream=stream).model_dump()
                payload.update(with_cache=True, preserve_cache=True)
                response = await client.post('/v1/chat/completions' if chat else '/v1/completions', json=payload)
                assert response.status_code == 200, response.text
                if stream:
                    chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                              if line.startswith('data: ') and line != 'data: [DONE]']
                    result = next(chunk for chunk in chunks if 'cache_session_id' in chunk)
                else:
                    result = response.json()
                owner = result['cache_session_id']
                assert owner == context.async_engine.ids[-1]
                assert result['id'].startswith('chatcmpl-') if chat else result['id'] == str(user_id)
                assert len(engine.scheduler.sessions) == 1
                assert engine.end_session(owner)
                assert not engine.scheduler.sessions
                assert not engine.scheduler.seq_manager._seq_map
                assert not context.session_manager.sessions

    asyncio.run(run())


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('user_id', [-1, 42])
@pytest.mark.parametrize('stream', [False, True])
def test_proxy_passes_internal_cache_owner_to_decode(prefill, monkeypatch, chat, user_id, stream):
    proxy = importlib.import_module('lmdeploy.serve.proxy.proxy')
    context, engine, app = prefill
    pool = PDConnectionPool()
    pool.is_connected = lambda *args: True
    pool.shelf_prefill_session = Mock(wraps=pool.shelf_prefill_session)
    pool.unshelf_prefill_session = Mock(wraps=pool.unshelf_prefill_session)
    owners = []

    async def check_model(model):
        return None

    def acknowledge(payload):
        owner = payload['migration_request']['remote_session_id']
        assert owner == context.async_engine.ids[-1]
        owners.append(owner)
        assert engine.end_session(owner)

    async def generate(payload, url, endpoint):
        if url == 'd':
            acknowledge(payload)
            return json.dumps(dict(choices=[dict(finish_reason='stop')]))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://prefill') as client:
            response = await client.post(endpoint, json=payload)
            assert response.status_code == 200
            return response.text

    async def stream_generate(payload, url, endpoint):
        acknowledge(payload)
        yield b'data: {"choices": [{"finish_reason": "stop"}]}\n\n'
        yield b'data: [DONE]\n\n'

    manager = SimpleNamespace(
        serving_strategy=ServingStrategy.DistServe, check_request_model=check_model,
        dummy_prefill=False, migration_protocol=MigrationProtocol.RDMA, rdma_config=None,
        pd_connection_pool=pool, generate=generate, stream_generate=stream_generate,
        get_node_url=lambda model, role: 'p' if role == EngineRole.Prefill else 'd',
        pre_call=lambda url: 0, post_call=lambda *args: None,
        create_background_tasks=lambda *args: None)
    monkeypatch.setattr(proxy, 'node_manager', manager)

    async def run():
        request = _request(chat, session_id=user_id, stream=stream, max_tokens=8)
        handler = proxy.chat_completions_v1 if chat else proxy.completions_v1
        response = await handler(request)
        assert response.status_code == 200
        if stream:
            async for _ in response.body_iterator:
                pass
        assert owners == [context.async_engine.ids[-1]]
        pool.shelf_prefill_session.assert_called_once_with(('p', 'd'), owners[0])
        pool.unshelf_prefill_session.assert_called_once_with(('p', 'd'), owners[0])
        assert not engine.scheduler.sessions
        assert not engine.scheduler.seq_manager._seq_map

    asyncio.run(run())


@pytest.mark.parametrize('chat', [False, True])
@pytest.mark.parametrize('owner', [None, -1, '42', True])
def test_proxy_rejects_invalid_cache_owner_without_public_id_fallback(monkeypatch, chat, owner):
    proxy = importlib.import_module('lmdeploy.serve.proxy.proxy')
    pool = PDConnectionPool()
    pool.is_connected = lambda *args: True
    calls = []

    async def check_model(model):
        return None

    async def generate(payload, url, endpoint):
        calls.append(url)
        result = dict(id='42', cache_block_ids=[0], remote_token_ids=[1])
        if owner is not None:
            result['cache_session_id'] = owner
        return json.dumps(result)

    manager = SimpleNamespace(
        serving_strategy=ServingStrategy.DistServe, check_request_model=check_model,
        dummy_prefill=False, pd_connection_pool=pool, generate=generate,
        get_node_url=lambda model, role: 'p' if role == EngineRole.Prefill else 'd',
        pre_call=lambda url: 0, post_call=lambda *args: None)
    monkeypatch.setattr(proxy, 'node_manager', manager)

    async def run():
        handler = proxy.chat_completions_v1 if chat else proxy.completions_v1
        response = await handler(_request(chat))
        assert response.status_code == 502
        assert 'cache_session_id' in response.body.decode()
        assert calls == ['p']
        assert not pool.migration_session_shelf

    asyncio.run(run())
