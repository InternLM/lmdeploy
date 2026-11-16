# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from lmdeploy.model import BaseChatTemplate
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.vl.engine import ImageEncoder
from lmdeploy.vl.media.image import ImageMediaIO
from lmdeploy.vl.model.base import VisionModel


@pytest.fixture
def processor():
    encoder = ImageEncoder.__new__(ImageEncoder)
    encoder.executor = ThreadPoolExecutor(max_workers=1)
    encoder._uses_new_preprocess = True
    encoder.max_batch_size = 1
    encoder.model = SimpleNamespace(get_input_prompt=VisionModel.get_input_prompt,
                                    has_input_ids=VisionModel.has_input_ids,
                                    _turbomind_native_vision=False)
    processor = MultimodalProcessor(SimpleNamespace(encode=lambda prompt, **kwargs: [len(prompt)]),
                                    BaseChatTemplate(), vl_encoder=encoder, backend='pytorch')
    yield processor
    processor.close()
    encoder.executor.shutdown(wait=True, cancel_futures=True)


@pytest.fixture
def messages():
    encoded = ImageMediaIO().encode_base64(Image.new('RGB', (2, 2)))
    return [dict(role='user', content=[
        dict(type='image_url', image_url=dict(url=f'data:image/png;base64,{encoded}')),
        dict(type='text', text='describe'),
    ])]


@pytest.mark.parametrize('backend,new_preprocess,native_vision', [
    ('pytorch', True, False),
    ('turbomind', True, True),
    ('pytorch', False, False),
    ('turbomind', False, False),
    ('turbomind', False, True),
])
def test_multimodal_rendering_and_packing_leave_event_loop_responsive(
        processor, messages, backend, new_preprocess, native_vision):
    async def run():
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        rendering = asyncio.Event()
        release = threading.Event()
        prompt_threads = []
        stages = []
        original_messages = deepcopy(messages)
        template_kwargs = {'reasoning_effort': 'high'}

        def prepare(parsed):
            assert threading.get_ident() != loop_thread
            prompt_threads.append(threading.get_ident())
            assert isinstance(parsed[0]['content'][0]['data'], Image.Image)
            stages.append('prepare')
            return parsed

        def render(parsed, **kwargs):
            assert threading.get_ident() == prompt_threads[0]
            assert kwargs['reasoning_effort'] == 'low'
            stages.append('render')
            loop.call_soon_threadsafe(rendering.set)
            assert release.wait(5), 'Event loop did not resume during multimodal rendering'
            return 'rendered'

        def preprocess(parsed, input_prompt=None, mm_processor_kwargs=None):
            assert threading.get_ident() not in (loop_thread, prompt_threads[0])
            stages.append('preprocess')
            if new_preprocess:
                assert input_prompt == 'rendered'
                return {'input_ids': [8]}
            return parsed

        def forward(parsed, max_batch_size):
            assert threading.get_ident() not in (loop_thread, prompt_threads[0])
            stages.append('forward')
            return parsed

        def encode(prompt, **kwargs):
            assert threading.get_ident() == prompt_threads[0]
            stages.append('encode')
            return [len(prompt)]

        def pack(parsed, chat_template, tokenizer, tools=None, chat_template_kwargs=None):
            assert threading.get_ident() == prompt_threads[0]
            stages.append('pack')
            prompt = chat_template.messages2prompt(parsed, tools=tools, **chat_template_kwargs)
            return {'input_ids': tokenizer.encode(prompt)}

        encoder = processor.vl_encoder
        encoder._uses_new_preprocess = new_preprocess
        encoder.model._turbomind_native_vision = native_vision
        encoder.model.preprocess = preprocess
        encoder.model.forward = forward
        encoder.model.to_pytorch = pack
        encoder.model.to_turbomind = pack
        processor.backend = backend
        processor.chat_template = SimpleNamespace(prepare_multimodal_messages=prepare, messages2prompt=render)
        processor.tokenizer.encode = encode
        task = asyncio.create_task(processor.get_prompt_input(
            messages, do_preprocess=True, reasoning_effort='low', chat_template_kwargs=template_kwargs))
        try:
            await asyncio.wait_for(rendering.wait(), 5)
            assert not task.done()
        finally:
            release.set()
        assert await asyncio.wait_for(task, 5) == {'input_ids': [8]}
        if new_preprocess:
            assert stages == ['prepare', 'render', 'preprocess']
        else:
            inference = ['forward'] if backend == 'turbomind' and not native_vision else []
            assert stages == ['prepare', 'preprocess', *inference, 'pack', 'render', 'encode']
        assert messages == original_messages
        assert template_kwargs == {'reasoning_effort': 'high'}

    asyncio.run(run())


@pytest.mark.parametrize('backend', ['pytorch', 'turbomind'])
@pytest.mark.parametrize('new_preprocess', [True, False])
def test_multimodal_input_ids_do_not_render_or_retokenize(processor, messages, backend, new_preprocess):
    async def run():
        loop_thread = threading.get_ident()
        input_ids = [1, 2, 3]
        messages[0]['content'].insert(0, dict(type='text', text=input_ids))
        original_messages = deepcopy(messages)

        def preprocess(parsed, input_prompt=None, mm_processor_kwargs=None):
            assert threading.get_ident() != loop_thread
            if new_preprocess:
                assert input_prompt == input_ids
                return {'input_ids': input_prompt}
            return parsed

        def pack(parsed):
            assert threading.get_ident() != loop_thread
            return {'input_ids': parsed[0]['content'][0]['text']}

        encoder = processor.vl_encoder
        encoder._uses_new_preprocess = new_preprocess
        encoder.model._turbomind_native_vision = True
        encoder.model.preprocess = preprocess
        encoder.model.to_pytorch_with_input_ids = pack
        processor.backend = backend
        processor.chat_template.messages2prompt = Mock(side_effect=AssertionError('Unexpected rendering'))
        processor.tokenizer.encode = Mock(side_effect=AssertionError('Unexpected tokenization'))
        result = await processor.get_prompt_input(messages, do_preprocess=True)
        assert result == {'input_ids': input_ids}
        assert messages == original_messages

    asyncio.run(run())


@pytest.mark.parametrize('new_preprocess', [True, False])
def test_cancelled_multimodal_work_keeps_shared_prompt_worker_bounded(
        processor, messages, monkeypatch, new_preprocess):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        submit = Mock(wraps=processor._prompt_executor.submit)
        monkeypatch.setattr(processor._prompt_executor, 'submit', submit)
        worker_threads = []

        def render(parsed, **kwargs):
            worker_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return 'rendered'

        def encode(prompt, **kwargs):
            assert threading.get_ident() == worker_threads[0]
            return [len(prompt)]

        def pack(parsed, chat_template, tokenizer, **kwargs):
            return {'input_ids': tokenizer.encode(chat_template.messages2prompt(parsed))}

        encoder = processor.vl_encoder
        encoder._uses_new_preprocess = new_preprocess
        encoder.model.preprocess = Mock(side_effect=lambda parsed: parsed)
        encoder.model.to_pytorch = pack
        processor.chat_template = SimpleNamespace(prepare_multimodal_messages=lambda parsed: parsed,
                                                  messages2prompt=render)
        processor.tokenizer.encode = encode
        task = asyncio.create_task(processor.get_prompt_input(messages, do_preprocess=True))
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            submissions = submit.call_count
            text = asyncio.create_task(processor.get_prompt_input('next', do_preprocess=False))
            await asyncio.sleep(0)
            assert submit.call_count == submissions
            assert not text.done()
        finally:
            release.set()
        assert await asyncio.wait_for(text, 5) == {'prompt': 'next', 'input_ids': [4]}
        assert submit.call_count == submissions + 1
        assert encoder.model.preprocess.call_count == (0 if new_preprocess else 1)

    asyncio.run(run())
