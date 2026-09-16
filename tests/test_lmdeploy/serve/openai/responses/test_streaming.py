# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from lmdeploy.serve.core.chat_runner import ChatStreamChunk
from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall
from lmdeploy.serve.openai.responses import ResponsesRequest
from lmdeploy.serve.openai.responses.streaming import stream_response


async def _parsed_stream(result_generator, response_parser):
    streaming_tools = False
    async for res in result_generator:
        token_ids = res.token_ids if getattr(res, 'token_ids', None) is not None else []
        stream_deltas = response_parser.stream_chunk(
            res.response or '',
            token_ids,
            final=res.finish_reason is not None,
        )
        if not stream_deltas:
            if res.finish_reason is None and not token_ids:
                continue
            stream_deltas = [(DeltaMessage(role='assistant', content=''), False)]

        for delta_index, (delta_message, tool_emitted) in enumerate(stream_deltas):
            if tool_emitted:
                streaming_tools = True
            is_last_delta = delta_index == len(stream_deltas) - 1
            finish_reason = res.finish_reason if is_last_delta else None
            if finish_reason == 'stop' and streaming_tools:
                finish_reason = 'tool_calls'
            yield ChatStreamChunk(
                delta_message=delta_message,
                tool_emitted=tool_emitted,
                finish_reason=finish_reason,
                token_ids=token_ids,
                logprobs=getattr(res, 'logprobs', None),
                input_token_len=res.input_token_len,
                generate_token_len=res.generate_token_len,
                cached_tokens=getattr(res, 'cached_tokens', 0),
                routed_experts=getattr(res, 'routed_experts', None) if finish_reason is not None else None,
                reasoning_tokens=getattr(response_parser, 'reasoning_tokens', None),
                is_last_delta=is_last_delta,
            )


def test_responses_streaming_sse_shape(sse_payloads,
                                       passthrough_response_parser_cls):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='Hello ',
            input_token_len=8,
            generate_token_len=1,
            finish_reason=None,
        )
        yield SimpleNamespace(
            response='world!',
            input_token_len=8,
            generate_token_len=2,
            finish_reason='stop',
        )

    async def _collect_events():
        parser = passthrough_response_parser_cls(request)
        parser.reasoning_tokens = 2
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    events = asyncio.run(_collect_events())
    body = ''.join(events)
    payloads = sse_payloads(events)

    assert 'event: response.created' in body
    assert 'event: response.in_progress' in body
    assert 'event: response.output_item.added' in body
    assert 'event: response.content_part.added' in body
    assert 'event: response.output_text.delta' in body
    assert 'event: response.completed' in body
    assert any(payload.get('delta') == 'Hello ' for payload in payloads)
    added_item = next(payload['item'] for payload in payloads
                      if payload['type'] == 'response.output_item.added')
    done_item = next(payload['item'] for payload in payloads
                     if payload['type'] == 'response.output_item.done')
    completed_response = payloads[-1]['response']
    assert done_item['id'] == added_item['id']
    assert payloads[-1]['type'] == 'response.completed'
    assert completed_response['output_text'] == 'Hello world!'
    assert completed_response['usage']['output_tokens_details']['reasoning_tokens'] == 2


def test_responses_streaming_length_finish_reason_emits_incomplete_event(
        sse_payloads, passthrough_response_parser_cls):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='partial',
            input_token_len=8,
            generate_token_len=1,
            finish_reason='length',
        )

    async def _collect_events():
        parser = passthrough_response_parser_cls(request)
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    payloads = sse_payloads(asyncio.run(_collect_events()))

    assert payloads[-1]['type'] == 'response.incomplete'
    assert payloads[-1]['response']['status'] == 'incomplete'
    assert payloads[-1]['response']['incomplete_details'] == {
        'reason': 'max_output_tokens'
    }


def test_responses_streaming_error_finish_reason_emits_failed_event(
        sse_payloads, passthrough_response_parser_cls):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='',
            input_token_len=8,
            generate_token_len=0,
            finish_reason='error',
        )

    async def _collect_events():
        parser = passthrough_response_parser_cls(request)
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    payloads = sse_payloads(asyncio.run(_collect_events()))

    assert payloads[-1]['type'] == 'response.failed'
    assert payloads[-1]['response']['status'] == 'failed'
    assert payloads[-1]['response']['error']['code'] == 'server_error'


def test_responses_streaming_exception_emits_failed_event(
        sse_payloads, passthrough_response_parser_cls):
    request = ResponsesRequest(model='fake-model', input='Hi', stream=True)

    async def _result_generator():
        if False:
            yield None
        raise RequestError(ErrorCode.INTERNAL_ERROR)

    async def _collect_events():
        parser = passthrough_response_parser_cls(request)
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    payloads = sse_payloads(asyncio.run(_collect_events()))
    assert payloads[-1]['type'] == 'response.failed'
    assert payloads[-1]['response']['error'] == {
        'code': 'server_error',
        'message': 'An internal server error occurred.',
    }


def test_responses_streaming_empty_output_announces_message_item(
        sse_payloads, passthrough_response_parser_cls):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='',
            input_token_len=8,
            generate_token_len=0,
            finish_reason='stop',
        )

    async def _collect_events():
        parser = passthrough_response_parser_cls(request)
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    payloads = sse_payloads(asyncio.run(_collect_events()))

    assert any(payload['type'] == 'response.output_item.added'
               and payload['item']['type'] == 'message'
               for payload in payloads)
    assert any(payload['type'] == 'response.output_item.done'
               and payload['item']['type'] == 'message'
               for payload in payloads)
    assert payloads[-1]['type'] == 'response.completed'
    assert payloads[-1]['response']['output'][0]['type'] == 'message'
    assert payloads[-1]['response']['output'][0]['content'][0]['text'] == ''


def test_responses_streaming_tool_call_events(sse_payloads):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    class _ToolParser:
        reasoning_tokens = 0

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int],
                         **kwargs):
            if delta_text == 'tool-start':
                return [(
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='call_123',
                                function=DeltaFunctionCall(
                                    name='search', arguments='{"query":'),
                            )
                        ],
                    ),
                    True,
                )]
            return [(
                DeltaMessage(
                    role='assistant',
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id='call_123',
                            function=DeltaFunctionCall(
                                arguments='"lmdeploy"}'),
                        )
                    ],
                ),
                True,
            )]

    async def _result_generator():
        yield SimpleNamespace(
            response='tool-start',
            token_ids=[101],
            input_token_len=8,
            generate_token_len=1,
            finish_reason=None,
        )
        yield SimpleNamespace(
            response='tool-end',
            token_ids=[102],
            input_token_len=8,
            generate_token_len=2,
            finish_reason='stop',
        )

    async def _collect_events():
        parser = _ToolParser()
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    events = asyncio.run(_collect_events())
    body = ''.join(events)
    payloads = sse_payloads(events)

    assert 'event: response.output_item.added' in body
    assert 'event: response.function_call_arguments.delta' in body
    assert 'event: response.function_call_arguments.done' in body
    added = next(payload for payload in payloads
                 if payload['type'] == 'response.output_item.added')
    done = next(payload for payload in payloads
                if payload['type'] == 'response.output_item.done')
    assert added['item']['type'] == 'function_call'
    assert added['item']['name'] == 'search'
    assert added['item']['status'] == 'in_progress'
    assert next(payload for payload in payloads if payload['type'] ==
                'response.function_call_arguments.done')['name'] == 'search'
    assert done['item']['arguments'] == '{"query":"lmdeploy"}'
    assert done['item']['status'] == 'completed'
    assert payloads[-1]['response']['output'][0]['type'] == 'function_call'
    assert payloads[-1]['response']['output'][0]['status'] == 'completed'


@pytest.mark.parametrize(('parallel_tool_calls', 'expected_call_ids'), [
    (False, ['call_123']),
    (None, ['call_123', 'call_456']),
])
def test_responses_streaming_parallel_tool_calls_filtering(
        parallel_tool_calls, expected_call_ids, sse_payloads):
    request = ResponsesRequest(
        model='fake-model',
        input='Hi there',
        stream=True,
        parallel_tool_calls=parallel_tool_calls,
    )

    class _ParallelToolParser:
        reasoning_tokens = 0

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int],
                         **kwargs):
            return [(
                DeltaMessage(
                    role='assistant',
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id='call_123',
                            function=DeltaFunctionCall(name='search',
                                                       arguments='{}'),
                        ),
                        DeltaToolCall(
                            index=1,
                            id='call_456',
                            function=DeltaFunctionCall(name='lookup',
                                                       arguments='{}'),
                        ),
                    ],
                ),
                True,
            )]

    async def _result_generator():
        yield SimpleNamespace(
            response='tools',
            token_ids=[101],
            input_token_len=8,
            generate_token_len=1,
            finish_reason='stop',
        )

    async def _collect_events():
        parser = _ParallelToolParser()
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    payloads = sse_payloads(asyncio.run(_collect_events()))
    added_items = [
        payload['item'] for payload in payloads
        if payload['type'] == 'response.output_item.added'
    ]
    completed_output = payloads[-1]['response']['output']

    assert [item['call_id'] for item in added_items] == expected_call_ids
    assert [item['call_id'] for item in completed_output] == expected_call_ids


def test_responses_streaming_text_indices_follow_text_item_order(sse_payloads):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    class _ToolThenTextParser:
        reasoning_tokens = 0

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int],
                         **kwargs):
            if delta_text == 'tool':
                return [(
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='call_123',
                                function=DeltaFunctionCall(name='search',
                                                           arguments='{}'),
                            )
                        ],
                    ),
                    True,
                )]
            return [(DeltaMessage(role='assistant',
                                  content='visible text'), False)]

    async def _result_generator():
        yield SimpleNamespace(
            response='tool',
            token_ids=[101],
            input_token_len=8,
            generate_token_len=1,
            finish_reason=None,
        )
        yield SimpleNamespace(
            response='text',
            token_ids=[102],
            input_token_len=8,
            generate_token_len=2,
            finish_reason='stop',
        )

    async def _collect_events():
        parser = _ToolThenTextParser()
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    payloads = sse_payloads(asyncio.run(_collect_events()))

    text_events = [
        payload for payload in payloads if payload['type'] in (
            'response.output_text.delta',
            'response.output_text.done',
            'response.content_part.done',
        )
    ]
    text_item_done = next(payload for payload in payloads
                          if payload['type'] == 'response.output_item.done'
                          and payload['item']['type'] == 'message')
    completed_output = payloads[-1]['response']['output']

    assert {payload['output_index'] for payload in text_events} == {1}
    assert text_item_done['output_index'] == 1
    assert completed_output[0]['type'] == 'function_call'
    assert completed_output[1]['type'] == 'message'


def test_responses_streaming_accepts_parser_delta_list(sse_payloads):
    request = ResponsesRequest(model='fake-model',
                               input='Hi there',
                               stream=True)

    class _MultiDeltaParser:
        reasoning_tokens = 0

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int],
                         **kwargs):
            return [
                (
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='call_123',
                                function=DeltaFunctionCall(name='search',
                                                           arguments='{}'),
                            )
                        ],
                    ),
                    True,
                ),
                (DeltaMessage(role='assistant',
                              content='visible text'), False),
            ]

    async def _result_generator():
        yield SimpleNamespace(
            response='mixed',
            token_ids=[101],
            input_token_len=8,
            generate_token_len=1,
            finish_reason='stop',
        )

    async def _collect_events():
        parser = _MultiDeltaParser()
        return [
            event async for event in stream_response(
                _parsed_stream(_result_generator(), parser),
                request=request,
                model_name='fake-model',
                created_time=123,
            )
        ]

    completed_output = sse_payloads(asyncio.run(
        _collect_events()))[-1]['response']['output']

    assert completed_output[0]['type'] == 'function_call'
    assert completed_output[1]['type'] == 'message'
    assert completed_output[1]['content'][0]['text'] == 'visible text'
