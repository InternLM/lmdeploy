# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import threading
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

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
        submit = Mock(wraps=processor._text_executor.submit)
        monkeypatch.setattr(processor._text_executor, 'submit', submit)

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
    messages = [dict(role='user', content=[dict(type='text', text='hello'), dict(type='text', text='world')])]
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
