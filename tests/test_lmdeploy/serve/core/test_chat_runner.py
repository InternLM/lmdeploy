# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.serve.core.chat_runner import ChatRunner, ChatRunnerOptions
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage, DeltaToolCall


class _FakeSession:

    def __init__(self, session_id=0):
        self.session_id = session_id
        self.aborted = False

    async def async_abort(self):
        self.aborted = True


class _FakeSessionManager:

    def __init__(self):
        self.removed = []
        self.session = _FakeSession()

    def get(self, session_id=None):
        if session_id is not None:
            self.session.session_id = session_id
        return self.session

    def remove(self, session):
        self.removed.append(session)


class _FakeEngine:

    model_name = 'fake-model'
    backend_config = SimpleNamespace(adapters=[])

    def __init__(self, outputs=None):
        self.outputs = outputs or [
            SimpleNamespace(
                response='ok',
                token_ids=[1],
                input_token_len=3,
                generate_token_len=1,
                finish_reason='stop',
                cached_tokens=0,
                logprobs=[{
                    1: -0.1,
                }],
                routed_experts=[[1]],
                cache_block_ids=['cache'],
            )
        ]
        self.session_mgr = _FakeSessionManager()
        self.preprocess_messages = None
        self.preprocess_kwargs = None
        self.generator_closed = False

    async def preprocess(self, messages, session, **kwargs):
        self.preprocess_messages = messages
        self.preprocess_kwargs = kwargs
        return SimpleNamespace(messages=messages, session=session)

    def generate(self, request, **kwargs):
        async def _generator():
            try:
                for output in self.outputs:
                    yield output
            finally:
                self.generator_closed = True

        return _generator()


class _FakeServerContext:

    def __init__(self, parser_cls, outputs=None):
        self.async_engine = _FakeEngine(outputs)
        self.default_gen_config = {}
        self.response_parser_cls = parser_cls

    @property
    def session_manager(self):
        return self.async_engine.session_mgr

    def create_session(self, session_id=None):
        return self.session_manager.get(session_id)


class _Parser:
    supports_required_tool_choice = False
    tool_parser_cls = object()
    tool_parser = object()
    reasoning_tokens = 2

    def __init__(self, request):
        self.request = request

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        return [(DeltaMessage(role='assistant', content=delta_text), False)]

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        return text, None, None

def _request(**kwargs):
    defaults = {
        'model': 'fake-model',
        'messages': [{
            'role': 'user',
            'content': 'hi',
        }],
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _tools():
    return [{
        'type': 'function',
        'function': {
            'name': 'search',
            'parameters': {
                'type': 'object',
            },
        },
    }]


def test_runner_forwards_parser_adjusted_response_format_to_engine():
    response_format = {'type': 'json_object'}

    class _AdjustingParser(_Parser):

        def __init__(self, request):
            super().__init__(request)
            self.request = request.model_copy(update={'response_format': response_format})

    context = _FakeServerContext(_AdjustingParser)

    asyncio.run(ChatRunner.prepare(context, _request()))

    assert context.async_engine.preprocess_kwargs['gen_config'].response_format == response_format


def test_runner_normalizes_string_stop_word_for_engine():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request(stop='END')))

    assert context.async_engine.preprocess_kwargs['gen_config'].stop_words == ['END']


def test_runner_maps_return_logprob_to_engine_logprobs():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request(return_logprob=True)))

    assert context.async_engine.preprocess_kwargs['gen_config'].logprobs == 1


def test_runner_maps_chat_logprobs_to_engine_logprobs():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request(logprobs=True, top_logprobs=3)))

    assert context.async_engine.preprocess_kwargs['gen_config'].logprobs == 3


def test_runner_always_preprocesses_chat_messages():
    context = _FakeServerContext(_Parser)

    asyncio.run(ChatRunner.prepare(context, _request()))

    assert context.async_engine.preprocess_kwargs['do_preprocess'] is True


def test_runner_skips_preprocess_for_raw_input_ids():
    context = _FakeServerContext(_Parser)

    chat_runner = asyncio.run(
        ChatRunner.prepare(
            context,
            _request(messages=[]),
            ChatRunnerOptions(input_ids=[1, 2, 3], do_preprocess=False),
        ))

    assert chat_runner.request.messages == []
    assert context.async_engine.preprocess_messages is None
    assert context.async_engine.preprocess_kwargs['do_preprocess'] is False
    assert context.async_engine.preprocess_kwargs['input_ids'] == [1, 2, 3]


def test_runner_rejects_required_tool_choice_for_unsupported_response_parser():
    context = _FakeServerContext(_Parser)

    with pytest.raises(RequestError) as exc_info:
        asyncio.run(ChatRunner.prepare(
            context,
            _request(tool_choice='required', tools=_tools()),
        ))

    assert exc_info.value.code == ErrorCode.INVALID_REQUEST
    assert 'does not support `tool_choice="required"`' in exc_info.value.message


def test_runner_stream_chunks_preserve_metadata():
    context = _FakeServerContext(_Parser)

    async def _run():
        chat_runner = await ChatRunner.prepare(
            context,
            _request(return_token_ids=True, return_routed_experts=True, return_logprob=True),
        )
        return [chunk async for chunk in chat_runner.stream()]

    chunks = asyncio.run(_run())

    assert len(chunks) == 1
    assert chunks[0].delta_message.content == 'ok'
    assert chunks[0].token_ids == [1]
    assert chunks[0].logprobs == [{1: -0.1}]
    assert chunks[0].routed_experts == [[1]]
    assert chunks[0].reasoning_tokens == 2
    assert context.async_engine.generator_closed is True
    assert context.session_manager.removed == [context.session_manager.session]


def test_runner_close_cleans_prepared_unconsumed_request():
    context = _FakeServerContext(_Parser)

    async def _run():
        chat_runner = await ChatRunner.prepare(context, _request())
        await chat_runner.close()
        await chat_runner.close()

    asyncio.run(_run())

    assert context.session_manager.removed == [context.session_manager.session]


def test_runner_parser_complete_error_raises_request_error():
    class _FailingParser(_Parser):

        def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
            raise RuntimeError('bad tool payload')

    async def _run():
        chat_runner = await ChatRunner.prepare(_FakeServerContext(_FailingParser), _request())
        return await chat_runner.collect()

    with pytest.raises(RequestError) as exc_info:
        asyncio.run(_run())

    assert exc_info.value.code == ErrorCode.INVALID_REQUEST
    assert 'bad tool payload' in exc_info.value.message


def test_runner_engine_error_is_not_reclassified_as_request_error():
    class _FailingEngine(_FakeEngine):

        def generate(self, request, **kwargs):
            async def _generator():
                if False:
                    yield None
                raise RuntimeError('engine exploded')

            return _generator()

    context = _FakeServerContext(_Parser)
    context.async_engine = _FailingEngine()

    async def _run():
        chat_runner = await ChatRunner.prepare(context, _request())
        return await chat_runner.collect()

    with pytest.raises(RuntimeError, match='engine exploded'):
        asyncio.run(_run())


def test_runner_binds_effective_preprocessed_decode_options():
    class StrictParser(_Parser):
        requires_output_chunks = True

        def bind_generation_config(self, config):
            self.bound_config = config

    class AdjustingEngine(_FakeEngine):

        async def preprocess(self, messages, session, **kwargs):
            result = await super().preprocess(messages, session, **kwargs)
            # AsyncEngine preprocess also copies and adjusts GenerationConfig.
            result.gen_config = SimpleNamespace(skip_special_tokens=False)
            self.effective_config = result.gen_config
            return result

    context = _FakeServerContext(StrictParser)
    context.async_engine = AdjustingEngine()

    async def run():
        runner = await ChatRunner.prepare(context, _request())
        assert runner.response_parser.bound_config is context.async_engine.effective_config
        assert runner.gen_config is not runner.response_parser.bound_config
        await runner.close()

    asyncio.run(run())


@pytest.mark.parametrize('stream', [False, True])
def test_runner_preserves_empty_batches_and_final_boundary(stream):
    class StrictParser(_Parser):
        requires_output_chunks = True

        def __init__(self, request):
            super().__init__(request)
            self.rows = []

        def stream_chunk(self, text, ids, *, final, engine_final):
            assert final == engine_final
            self.rows.append((text, ids, engine_final))
            return super().stream_chunk(text, ids)

        def parse_complete(self, text, ids, *, output_chunks):
            self.rows = output_chunks
            assert text == 'X</think>🧑'
            assert ids == [1, 2, 3, 4]
            return text, None, None

    rows = [('', [1, 2], False), ('X</think>🧑', [3, 4], False), ('', [], True)]
    outputs = [SimpleNamespace(response=text, token_ids=ids, finish_reason='stop' if final else None,
                               input_token_len=3, generate_token_len=4, cached_tokens=0,
                               logprobs=[{i: -0.1} for i in ids], routed_experts=None,
                               cache_block_ids=None) for text, ids, final in rows]
    engine = _FakeEngine(outputs)
    parser = StrictParser(_request(return_token_ids=True))

    async def run():
        runner = ChatRunner(parser.request, parser, None, engine.session_mgr.session,
                            engine.generate(None), engine.session_mgr)
        if stream:
            chunks = [x async for x in runner.stream()]
            assert [i for x in chunks for i in x.token_ids] == [1, 2, 3, 4]
            assert [p for x in chunks for p in x.logprobs or []] == [{i: -0.1} for i in [1, 2, 3, 4]]
        else:
            result = await runner.collect()
            assert result.token_ids == [1, 2, 3, 4]
        assert parser.rows == rows
        assert engine.generator_closed
        assert engine.session_mgr.removed == [engine.session_mgr.session]

    asyncio.run(run())


def test_legacy_parser_does_not_receive_new_keywords():
    class LegacyParser(_Parser):

        def stream_chunk(self, text, ids, *, final):
            return super().stream_chunk(text, ids)

        def parse_complete(self, text, ids):
            return text, None, None

    async def run():
        for stream in (True, False):
            runner = await ChatRunner.prepare(_FakeServerContext(LegacyParser), _request())
            if stream:
                assert [x.delta_message.content async for x in runner.stream()] == ['ok']
            else:
                assert (await runner.collect()).text == 'ok'

    asyncio.run(run())


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('stop_sequence', [None, 'END'])
def test_runner_preserves_stop_sequence_with_token_id_parser(stream, stop_sequence):
    call = DeltaToolCall(index=0, id='call_search', function={'name': 'search', 'arguments': '{}'})

    class StrictParser(_Parser):
        requires_output_chunks = True

        def stream_chunk(self, text, ids, *, final, engine_final):
            assert final and engine_final
            return [(DeltaMessage(tool_calls=[call]), True)]

        def parse_complete(self, text, ids, *, output_chunks):
            assert output_chunks == [('ok', [1], True)]
            return '', [call], None

    engine = _FakeEngine()
    engine.outputs[0].stop_sequence = stop_sequence
    parser = StrictParser(_request(tools=_tools()))

    async def run():
        runner = ChatRunner(parser.request, parser, None, engine.session_mgr.session,
                            engine.generate(None), engine.session_mgr)
        if stream:
            chunks = [chunk async for chunk in runner.stream()]
            assert chunks[-1].delta_message.tool_calls == [call]
            result = chunks[-1]
        else:
            result = await runner.collect()
            assert result.tool_calls == [call]
        assert result.finish_reason == ('stop' if stop_sequence is not None else 'tool_calls')
        assert result.stop_sequence == stop_sequence
        assert runner._closed and engine.generator_closed
        assert engine.session_mgr.removed == [engine.session_mgr.session]

    asyncio.run(run())
