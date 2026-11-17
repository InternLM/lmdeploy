# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lmdeploy.model import BaseChatTemplate
from lmdeploy.serve.processors import MultimodalProcessor


@pytest.fixture
def processor():
    processor = MultimodalProcessor(SimpleNamespace(encode=lambda prompt, **kwargs: list(prompt.encode())),
                                    BaseChatTemplate())
    yield processor
    processor.close()


def test_rendering_and_tokenization_leave_event_loop_responsive(processor):
    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        started = asyncio.Event()
        release = threading.Event()
        worker_threads = []

        def render(prompt, **kwargs):
            assert threading.get_ident() != loop_thread
            worker_threads.append(threading.get_ident())
            return 'rendered ' + prompt

        def encode(prompt, **kwargs):
            assert threading.get_ident() == worker_threads[0]
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5), 'Event loop did not resume during tokenization'
            return [len(prompt)]

        processor.chat_template = SimpleNamespace(messages2prompt=render)
        processor.tokenizer.encode = encode
        task = asyncio.create_task(processor.get_prompt_input('hello', do_preprocess=True))
        try:
            await asyncio.wait_for(started.wait(), 5)
            assert not task.done()
        finally:
            release.set()
        assert await task == {'prompt': 'rendered hello', 'input_ids': [14]}

    asyncio.run(run())


@pytest.mark.parametrize('fail_cancelled_work', [False, True])
def test_cancellation_keeps_executor_bounded(processor, monkeypatch, fail_cancelled_work):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        calls = []
        unhandled = []
        loop.set_exception_handler(lambda loop, context: unhandled.append(context))
        submit = Mock(wraps=processor._prompt_executor.submit)
        monkeypatch.setattr(processor._prompt_executor, 'submit', submit)

        def encode(prompt, **kwargs):
            calls.append(prompt)
            if prompt == 'first':
                loop.call_soon_threadsafe(started.set)
                assert release.wait(5)
                if fail_cancelled_work:
                    raise ValueError('cancelled worker failed')
            return [len(prompt)]

        processor.tokenizer.encode = encode
        first = asyncio.create_task(processor.get_prompt_input('first', do_preprocess=False))
        waiting = []
        try:
            await asyncio.wait_for(started.wait(), 5)
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            waiting = [asyncio.create_task(processor.get_prompt_input(str(i), do_preprocess=False))
                       for i in range(8)]
            await asyncio.sleep(0)
            assert submit.call_count == 1
            for task in waiting[:-1]:
                task.cancel()
            await asyncio.gather(*waiting[:-1], return_exceptions=True)
            assert submit.call_count == 1
        finally:
            release.set()
        assert await asyncio.wait_for(waiting[-1], 5) == {'prompt': '7', 'input_ids': [1]}
        assert calls == ['first', '7']
        assert submit.call_count == 2
        await asyncio.sleep(0)
        assert unhandled == []

    asyncio.run(run())


def test_worker_exception_releases_slot(processor):
    def encode(prompt, **kwargs):
        if prompt == 'invalid':
            raise ValueError('invalid prompt')
        return [1]

    async def run():
        processor.tokenizer.encode = encode
        with pytest.raises(ValueError, match='invalid prompt'):
            await processor.get_prompt_input('invalid', do_preprocess=False)
        result = await asyncio.wait_for(processor.get_prompt_input('valid', do_preprocess=False), 5)
        assert result == {'prompt': 'valid', 'input_ids': [1]}

    asyncio.run(run())


@pytest.mark.parametrize('effort,template_effort,expected', [
    ('low', 'high', 'Low'), (None, 'high', 'High'), ('max', None, 'Max'),
])
def test_text_template_options_and_inputs_are_preserved(processor, hf_media_template,
                                                       effort, template_effort, expected):
    processor.chat_template = hf_media_template()
    # Separators between structured parts belong to the model's template.
    # A newline inside one part must survive both string and structured input.
    messages = [dict(role='user', content=[dict(type='text', text='hello\nworld')])]
    original_messages = deepcopy(messages)
    kwargs = {'reasoning_effort': template_effort}
    original_kwargs = dict(kwargs)
    result = asyncio.run(processor.get_prompt_input(
        messages, do_preprocess=True, reasoning_effort=effort, chat_template_kwargs=kwargs))
    assert f'Reasoning Effort: {expected}' in result['prompt']
    assert 'hello\nworld' in result['prompt']
    assert result['input_ids'] == list(result['prompt'].encode())
    assert messages == original_messages
    assert kwargs == original_kwargs


def test_string_template_merges_text_parts_with_newlines(processor):
    messages = [dict(role='user', content=[dict(type='text', text='hello'), dict(type='text', text='world')])]
    original_messages = deepcopy(messages)
    result = asyncio.run(processor.get_prompt_input(messages, do_preprocess=True))
    assert result['prompt'] == 'hello\nworld'
    assert messages == original_messages


def test_engine_close_releases_text_worker(processor):
    from lmdeploy.serve.core.async_engine import AsyncEngine

    threads = []

    def encode(prompt, **kwargs):
        threads.append(threading.current_thread())
        return [1]

    async def run():
        processor.tokenizer.encode = encode
        await processor.get_prompt_input('hello', do_preprocess=False)
        engine = AsyncEngine.__new__(AsyncEngine)
        engine.prompt_processor = processor
        engine.session_mgr = Mock()
        engine.engine = Mock()
        engine.close()
        assert not threads[0].is_alive()
        engine.session_mgr.clear.assert_called_once()
        engine.engine.close.assert_called_once()
        # Closing is idempotent, and failed submissions must not consume the slot.
        processor.close()
        for _ in range(2):
            with pytest.raises(RuntimeError, match='shutdown'):
                await asyncio.wait_for(processor.get_prompt_input('closed', do_preprocess=False), 5)

    asyncio.run(run())


@pytest.mark.parametrize('multimodal', [False, True])
def test_prompt_worker_does_not_share_tokenizer_with_event_loop(monkeypatch, multimodal):
    from fastapi import APIRouter

    from lmdeploy.messages import PytorchEngineConfig
    from lmdeploy.serve.anthropic.adapter import count_input_tokens
    from lmdeploy.serve.core import async_engine
    from lmdeploy.serve.core.vl_async_engine import VLAsyncEngine
    from lmdeploy.serve.openai.endpoints.auxiliary import register
    from lmdeploy.serve.openai.protocol import EncodeRequest, PoolingRequest, PPLRequest

    async def run():
        loop = asyncio.get_running_loop()
        started, release = asyncio.Event(), threading.Event()

        class Tokenizer:
            # Model a tokenizer whose mutable state cannot be borrowed twice.
            def __init__(self, *args, **kwargs):
                self.borrowed = threading.Lock()

            def encode(self, prompt, **kwargs):
                if not self.borrowed.acquire(blocking=False):
                    raise RuntimeError('Already borrowed')
                try:
                    if prompt == 'worker':
                        loop.call_soon_threadsafe(started.set)
                        assert release.wait(5), 'Event loop blocked behind the prompt worker'
                    return list(prompt.encode())
                finally:
                    self.borrowed.release()

            def decode(self, ids, **kwargs):
                if not self.borrowed.acquire(blocking=False):
                    raise RuntimeError('Already borrowed')
                try:
                    return bytes(ids).decode()
                finally:
                    self.borrowed.release()

        config = PytorchEngineConfig(session_len=4096, max_batch_size=1, enable_metrics=False)
        backend = SimpleNamespace(engine_config=config, create_instance=Mock(), close=Mock())
        monkeypatch.setattr(async_engine, 'Tokenizer', Tokenizer)
        monkeypatch.setattr(async_engine, 'get_chat_template', lambda *a, **kw: BaseChatTemplate())
        monkeypatch.setattr(async_engine, 'get_hf_gen_cfg', lambda *a, **kw: {})
        monkeypatch.setattr(async_engine, 'get_model_arch', lambda *a, **kw: ('LlamaForCausalLM', SimpleNamespace()))
        monkeypatch.setattr(async_engine.AsyncEngine, '_build_pytorch', lambda *a, **kw: backend)
        if multimodal:
            from lmdeploy.vl import engine as vl_engine

            monkeypatch.setattr(vl_engine, 'ImageEncoder', lambda *a, **kw: SimpleNamespace())
        engine_cls = VLAsyncEngine if multimodal else async_engine.AsyncEngine
        engine = engine_cls('model', backend='pytorch', backend_config=config)
        engine.async_get_reward_score = AsyncMock(return_value=[0.5])
        engine.async_get_ppl = AsyncMock(return_value=1.5)
        router = APIRouter()
        register(router, SimpleNamespace(async_engine=engine))
        endpoints = {route.path: route.endpoint for route in router.routes}
        task = asyncio.create_task(engine.prompt_processor.get_prompt_input('worker', do_preprocess=False))
        try:
            await asyncio.wait_for(started.wait(), 5)
            # Auxiliary APIs and streaming decode must work while preprocessing
            # holds its tokenizer, even for tokenizers that mutate on encode.
            for prompt in ('hello', ['hello', 'world']):
                response = await endpoints['/v1/encode'](EncodeRequest(input=prompt))
                assert response.length == (5 if isinstance(prompt, str) else [5, 5])
            await endpoints['/pooling'](PoolingRequest(input='hello'))
            engine.async_get_reward_score.assert_awaited_once_with([list(b'hello')])
            await endpoints['/get_ppl'](PPLRequest(input='hello'))
            engine.async_get_ppl.assert_awaited_once_with(list(b'hello'))
            assert count_input_tokens(engine, [dict(role='user', content='hello')]) == 5
            assert engine.tokenizer.decode(list(b'hello')) == 'hello'
            assert not task.done()
        finally:
            release.set()
            await asyncio.wait_for(task, 5)
            engine.close()

    asyncio.run(run())
