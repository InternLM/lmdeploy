from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseFunctionToolCall

from lmdeploy.serve.openai.protocol import DeltaFunctionCall, DeltaMessage, DeltaToolCall, FunctionCall, ToolCall
from lmdeploy.serve.openai.responses import (
    ResponsesRequest,
    create_responses_router,
)
from lmdeploy.serve.openai.responses.protocol import ResponseInputOutputItem
from lmdeploy.serve.openai.responses.request import (
    _messages_from_input,
    _openai_tools_from_responses,
    _to_generation_config,
    _tool_choice_from_responses,
    _validate_text_v1_request,
)
from lmdeploy.serve.openai.responses.response import _make_response
from lmdeploy.serve.openai.responses.streaming import _stream_response


class _FakeAsyncEngine:

    model_name = 'fake-model'
    backend_config = SimpleNamespace(adapters=[])
    tokenizer = SimpleNamespace(model=SimpleNamespace(model=None))

    def __init__(self):
        self.generate_kwargs = None
        self.prompt = None

    def generate(self, prompt, session, **kwargs):
        self.prompt = prompt
        self.generate_kwargs = kwargs

        async def _generator():
            yield SimpleNamespace(
                response='ok',
                token_ids=[1],
                input_token_len=1,
                generate_token_len=1,
                finish_reason='stop',
            )

        return _generator()


class _PassthroughResponseParser:

    tool_parser_cls = None
    last_request = None

    def __init__(self, request, tokenizer=None):
        self.request = request
        type(self).last_request = request

    def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
        return [(DeltaMessage(content=delta_text), False)] if delta_text else []

    def parse_complete(self, text: str, token_ids: list[int] | None = None, **kwargs):
        return text, None, None


class _FakeServerContext:

    response_parser_cls = _PassthroughResponseParser

    def __init__(self):
        self.async_engine = _FakeAsyncEngine()

    def create_session(self, session_id):
        return _FakeSession(session_id)


class _FakeSession:

    def __init__(self, session_id):
        self.session_id = session_id

    async def async_abort(self):
        pass


class _FakeRawRequest:

    async def is_disconnected(self):
        return False


def _responses_endpoint():
    context = _FakeServerContext()
    router = create_responses_router(context)
    return router.routes[0].endpoint, context


def _sse_payloads(events: list[str]):
    payloads = []
    for event in events:
        for line in event.splitlines():
            if line.startswith('data: '):
                payloads.append(json.loads(line.removeprefix('data: ')))
    return payloads


def test_responses_request_uses_structured_input_item_alias():
    item: ResponseInputOutputItem = ResponseFunctionToolCall(
        id='fc_1',
        call_id='call_1',
        name='search',
        arguments='{"query":"lmdeploy"}',
        type='function_call',
    )
    request = ResponsesRequest(model='fake-model', input=[item])

    assert _messages_from_input(request) == [{
        'role': 'assistant',
        'content': None,
        'tool_calls': [{
            'id': 'call_1',
            'type': 'function',
            'function': {
                'name': 'search',
                'arguments': '{"query":"lmdeploy"}',
            },
        }],
    }]


def test_responses_string_input_maps_to_user_message():
    request = ResponsesRequest(model='fake-model', input='Hi there')

    assert _messages_from_input(request) == [{'role': 'user', 'content': 'Hi there'}]


def test_responses_maps_instructions_and_typed_message_input():
    request = ResponsesRequest(
        model='fake-model',
        instructions='You are concise.',
        input=[{
            'type': 'message',
            'role': 'user',
            'content': [{
                'type': 'input_text',
                'text': 'Say hello.',
            }],
        }],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'system',
            'content': 'You are concise.',
        },
        {
            'role': 'user',
            'content': 'Say hello.',
        },
    ]


def test_responses_merges_multiple_system_messages():
    request = ResponsesRequest(
        model='fake-model',
        instructions='You are concise.',
        input=[
            {
                'type': 'message',
                'role': 'developer',
                'content': 'Follow the repo instructions.',
            },
            {
                'type': 'message',
                'role': 'system',
                'content': 'Use plain text.',
            },
            {
                'type': 'message',
                'role': 'user',
                'content': 'Say hello.',
            },
        ],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'system',
            'content': 'You are concise.\n\nFollow the repo instructions.\n\nUse plain text.',
        },
        {
            'role': 'user',
            'content': 'Say hello.',
        },
    ]


def test_responses_moves_developer_messages_before_conversation():
    request = ResponsesRequest(
        model='fake-model',
        input=[
            {
                'type': 'message',
                'role': 'user',
                'content': 'Say hello.',
            },
            {
                'type': 'message',
                'role': 'developer',
                'content': 'Follow the repo instructions.',
            },
        ],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'system',
            'content': 'Follow the repo instructions.',
        },
        {
            'role': 'user',
            'content': 'Say hello.',
        },
    ]


def test_responses_maps_function_call_history_to_chat_messages():
    request = ResponsesRequest(
        model='fake-model',
        input=[
            {
                'type': 'message',
                'role': 'user',
                'content': 'Search for lmdeploy.',
            },
            {
                'type': 'function_call',
                'call_id': 'call_123',
                'name': 'search',
                'arguments': '{"query":"lmdeploy"}',
            },
            {
                'type': 'function_call_output',
                'call_id': 'call_123',
                'output': '{"result":"ok"}',
            },
        ],
    )

    assert _messages_from_input(request) == [
        {
            'role': 'user',
            'content': 'Search for lmdeploy.',
        },
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [{
                'id': 'call_123',
                'type': 'function',
                'function': {
                    'name': 'search',
                    'arguments': '{"query":"lmdeploy"}',
                },
            }],
        },
        {
            'role': 'tool',
            'tool_call_id': 'call_123',
            'content': '{"result":"ok"}',
        },
    ]


def test_responses_rejects_non_string_function_call_arguments():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'function_call',
            'call_id': 'call_123',
            'name': 'search',
            'arguments': {
                'query': 'lmdeploy',
            },
        }],
    )

    try:
        _messages_from_input(request)
    except ValueError as err:
        assert 'Unsupported `arguments` in function_call item' in str(err)
    else:
        raise AssertionError('non-string function_call arguments should be rejected')


def test_responses_maps_function_call_output_text_parts():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'function_call_output',
            'call_id': 'call_123',
            'output': [{
                'type': 'input_text',
                'text': '{"result":"ok"}',
            }],
        }],
    )

    assert _messages_from_input(request) == [{
        'role': 'tool',
        'tool_call_id': 'call_123',
        'content': '{"result":"ok"}',
    }]


def test_responses_maps_function_tools_to_openai_tools():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        tools=[{
            'type': 'function',
            'name': 'search',
            'description': 'demo',
            'parameters': {
                'type': 'object',
            },
        }, {
            'type': 'web_search',
        }],
    )

    tools = _openai_tools_from_responses(request)

    assert tools is not None
    assert len(tools) == 1
    assert tools[0].function.name == 'search'
    assert tools[0].function.description == 'demo'
    assert tools[0].function.parameters == {'type': 'object'}


def test_responses_tool_choice_without_function_tools_validation():
    assert _tool_choice_from_responses('auto', None) == 'none'
    assert _tool_choice_from_responses('none', None) == 'none'

    for tool_choice in ('required', {'type': 'function', 'name': 'search'}):
        try:
            _tool_choice_from_responses(tool_choice, None)
        except ValueError as err:
            assert 'tools' in str(err)
        else:
            raise AssertionError(f'{tool_choice!r} should require function tools')


def test_responses_named_tool_choice_must_match_function_tools():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        tools=[{
            'type': 'function',
            'name': 'search',
        }],
        tool_choice={
            'type': 'function',
            'name': 'missing',
        },
    )
    tools = _openai_tools_from_responses(request)

    try:
        _tool_choice_from_responses(request.tool_choice, tools)
    except ValueError as err:
        assert 'not found' in str(err)
    else:
        raise AssertionError('named tool_choice should match one of the function tools')


def test_responses_explicit_unsupported_tool_choice_is_rejected():
    try:
        _tool_choice_from_responses({'type': 'web_search_preview'}, None)
    except ValueError as err:
        assert 'Unsupported tool_choice type' in str(err)
    else:
        raise AssertionError('explicit unsupported tool_choice should be rejected')


def test_responses_tool_validation_uses_tools_error_param():
    import asyncio

    endpoint, _ = _responses_endpoint()
    response = asyncio.run(
        endpoint(
            ResponsesRequest(
                model='fake-model',
                input='Hi',
                tools=[{
                    'type': 'function',
                }],
            ),
            _FakeRawRequest(),
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'tools'


def test_responses_tool_choice_validation_uses_tool_choice_error_param():
    import asyncio

    endpoint, _ = _responses_endpoint()
    response = asyncio.run(
        endpoint(
            ResponsesRequest(
                model='fake-model',
                input='Hi',
                tools=[{
                    'type': 'function',
                    'name': 'search',
                }],
                tool_choice={
                    'type': 'function',
                    'name': 'missing',
                },
            ),
            _FakeRawRequest(),
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'tool_choice'


def test_responses_tool_choice_none_does_not_require_tool_parser():
    import asyncio

    endpoint, context = _responses_endpoint()
    response = asyncio.run(
        endpoint(
            ResponsesRequest(
                model='fake-model',
                input='Hi',
                tools=[{
                    'type': 'function',
                    'name': 'search',
                }],
                tool_choice='none',
            ),
            _FakeRawRequest(),
        )
    )

    assert response['output_text'] == 'ok'
    assert context.async_engine.generate_kwargs['tools'] is None


def test_responses_uses_parser_adjusted_messages_for_generation():
    import asyncio

    class _AdjustingResponseParser(_PassthroughResponseParser):

        adjusted_messages = [{'role': 'user', 'content': 'adjusted'}]

        def __init__(self, request, tokenizer=None):
            super().__init__(request, tokenizer)
            self.request.messages = self.adjusted_messages

    endpoint, context = _responses_endpoint()
    context.response_parser_cls = _AdjustingResponseParser

    response = asyncio.run(endpoint(ResponsesRequest(model='fake-model', input='original'), _FakeRawRequest()))

    assert response['output_text'] == 'ok'
    assert context.async_engine.prompt is _AdjustingResponseParser.adjusted_messages


def test_responses_generation_config_mapping():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        max_output_tokens=32,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        stop=['!'],
        repetition_penalty=1.1,
        text={
            'format': {
                'type': 'json_schema',
                'name': 'answer',
                'schema': {
                    'type': 'object',
                },
                'strict': True,
            }
        },
    )

    gen_config = _to_generation_config(request)

    assert gen_config.max_new_tokens == 32
    assert gen_config.temperature == 0.2
    assert gen_config.top_p == 0.9
    assert gen_config.top_k == 20
    assert gen_config.stop_words == ['!']
    assert gen_config.repetition_penalty == 1.1
    assert gen_config.response_format == {
        'type': 'json_schema',
        'json_schema': {
            'name': 'answer',
            'schema': {
                'type': 'object',
            },
            'strict': True,
        },
    }


def test_responses_non_stream_response_shape():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi there',
    )

    response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='Hello world!',
        input_tokens=8,
        output_tokens=2,
        finish_reason='stop',
    ).model_dump(exclude_none=True)

    assert response['object'] == 'response'
    assert response['status'] == 'completed'
    assert response['output_text'] == 'Hello world!'
    assert response['usage']['input_tokens'] == 8
    assert response['usage']['output_tokens'] == 2
    assert response['usage']['total_tokens'] == 10
    assert response['output'][0]['type'] == 'message'
    assert response['output'][0]['content'][0]['text'] == 'Hello world!'


def test_responses_length_finish_reason_sets_incomplete_details():
    request = ResponsesRequest(model='fake-model', input='Hi there')

    response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='partial',
        input_tokens=8,
        output_tokens=2,
        finish_reason='length',
    ).model_dump(exclude_none=True)

    assert response['status'] == 'incomplete'
    assert response['incomplete_details'] == {'reason': 'max_output_tokens'}
    assert response['output'][0]['status'] == 'incomplete'


def test_responses_error_finish_reasons_do_not_complete_successfully():
    request = ResponsesRequest(model='fake-model', input='Hi there')

    error_response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        input_tokens=8,
        output_tokens=0,
        finish_reason='error',
    ).model_dump(exclude_none=True)
    abort_response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        input_tokens=8,
        output_tokens=0,
        finish_reason='abort',
    ).model_dump(exclude_none=True)

    assert error_response['status'] == 'failed'
    assert error_response['error']['code'] == 'server_error'
    assert abort_response['status'] == 'cancelled'
    assert abort_response['error']['code'] == 'server_error'


def test_responses_tool_call_response_shape():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
    )

    response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search', arguments='{"query":"lmdeploy"}'),
            )
        ],
        input_tokens=8,
        output_tokens=2,
        finish_reason='tool_calls',
    ).model_dump(exclude_none=True)

    assert response['output_text'] == ''
    assert response['output'][0] == {
        'id': 'call_123',
        'type': 'function_call',
        'call_id': 'call_123',
        'name': 'search',
        'arguments': '{"query":"lmdeploy"}',
        'status': 'completed',
    }


def test_responses_parallel_tool_calls_false_keeps_first_tool_call():
    request = ResponsesRequest(model='fake-model', input='Hi', parallel_tool_calls=False)

    response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search', arguments='{"query":"lmdeploy"}'),
            ),
            ToolCall(
                id='call_456',
                function=FunctionCall(name='lookup', arguments='{"query":"vllm"}'),
            ),
        ],
        input_tokens=8,
        output_tokens=2,
        finish_reason='tool_calls',
    ).model_dump(exclude_none=True)

    assert response['parallel_tool_calls'] is False
    assert len(response['output']) == 1
    assert response['output'][0]['call_id'] == 'call_123'


def test_responses_parallel_tool_calls_none_keeps_all_tool_calls():
    request = ResponsesRequest(model='fake-model', input='Hi', parallel_tool_calls=None)

    response_model = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search', arguments='{"query":"lmdeploy"}'),
            ),
            ToolCall(
                id='call_456',
                function=FunctionCall(name='lookup', arguments='{"query":"vllm"}'),
            ),
        ],
        input_tokens=8,
        output_tokens=2,
        finish_reason='tool_calls',
    )
    response = response_model.model_dump(exclude_none=True)

    assert response_model.parallel_tool_calls is None
    assert [item['call_id'] for item in response['output']] == ['call_123', 'call_456']


def test_responses_tool_call_response_accepts_no_visible_text():
    request = ResponsesRequest(model='fake-model', input='Hi')

    response = _make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text=None,
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search', arguments='{"query":"lmdeploy"}'),
            )
        ],
        input_tokens=8,
        output_tokens=2,
        finish_reason='tool_calls',
    ).model_dump(exclude_none=True)

    assert response['output_text'] == ''
    assert response['output'][0]['type'] == 'function_call'


def test_responses_rejects_unsupported_agentic_fields_for_text_v1():
    request = ResponsesRequest(model='fake-model', input='Hi', previous_response_id='resp_123')

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'previous_response_id'


def test_responses_rejects_unsupported_conversation_for_text_v1():
    request = ResponsesRequest(model='fake-model', input='Hi', conversation={'id': 'conv_123'})

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'conversation'


def test_responses_rejects_unsupported_prompt():
    request = ResponsesRequest(model='fake-model', input='Hi', prompt={'id': 'pmpt_123'})

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'prompt'


def test_responses_rejects_missing_input_for_text_v1():
    request = ResponsesRequest(model='fake-model')

    response = _validate_text_v1_request(request)

    assert response is not None
    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'input'


def test_responses_rejects_unsupported_input_items():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'message',
            'role': 'user',
            'content': [{
                'type': 'input_image',
                'image_url': 'https://example.com/cat.png',
            }],
        }],
    )

    try:
        _messages_from_input(request)
    except ValueError as err:
        assert 'input_image' in str(err)
    else:
        raise AssertionError('input_image should be rejected by Text V1')


def test_responses_rejects_non_string_text_content_parts():
    import asyncio

    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'message',
            'role': 'user',
            'content': [{
                'type': 'input_text',
                'text': 123,
            }],
        }],
    )

    endpoint, _ = _responses_endpoint()
    response = asyncio.run(endpoint(request, _FakeRawRequest()))

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body['error']['param'] == 'input'
    assert 'Expected string' in body['error']['message']


def test_responses_rejects_reasoning_input_items():
    request = ResponsesRequest(
        model='fake-model',
        input=[{
            'type': 'reasoning',
            'summary': [],
        }],
    )

    try:
        _messages_from_input(request)
    except ValueError as err:
        assert 'reasoning' in str(err)
    else:
        raise AssertionError('reasoning input items should be rejected by Text V1')


def test_responses_forwards_repetition_penalty():
    import asyncio

    endpoint, context = _responses_endpoint()
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        repetition_penalty=1.1,
    )

    response = asyncio.run(endpoint(request, _FakeRawRequest()))

    assert response['output_text'] == 'ok'
    assert context.async_engine.generate_kwargs['gen_config'].repetition_penalty == 1.1


def test_responses_parser_request_uses_max_completion_tokens():
    import asyncio

    endpoint, _ = _responses_endpoint()
    request = ResponsesRequest(model='fake-model', input='Hi', max_output_tokens=17)

    asyncio.run(endpoint(request, _FakeRawRequest()))

    assert _PassthroughResponseParser.last_request.max_completion_tokens == 17


def test_responses_streaming_sse_shape():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

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
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_PassthroughResponseParser(request),
            )
        ]

    import asyncio

    events = asyncio.run(_collect_events())
    body = ''.join(events)
    payloads = _sse_payloads(events)

    assert 'event: response.created' in body
    assert 'event: response.in_progress' in body
    assert 'event: response.output_item.added' in body
    assert 'event: response.content_part.added' in body
    assert 'event: response.output_text.delta' in body
    assert 'event: response.completed' in body
    assert any(payload.get('delta') == 'Hello ' for payload in payloads)
    added_item = next(payload['item'] for payload in payloads if payload['type'] == 'response.output_item.added')
    done_item = next(payload['item'] for payload in payloads if payload['type'] == 'response.output_item.done')
    completed_response = payloads[-1]['response']
    assert done_item['id'] == added_item['id']
    assert payloads[-1]['type'] == 'response.completed'
    assert completed_response['output_text'] == 'Hello world!'


def test_responses_streaming_length_finish_reason_emits_incomplete_event():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='partial',
            input_token_len=8,
            generate_token_len=1,
            finish_reason='length',
        )

    async def _collect_events():
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_PassthroughResponseParser(request),
            )
        ]

    import asyncio

    payloads = _sse_payloads(asyncio.run(_collect_events()))

    assert payloads[-1]['type'] == 'response.incomplete'
    assert payloads[-1]['response']['status'] == 'incomplete'
    assert payloads[-1]['response']['incomplete_details'] == {'reason': 'max_output_tokens'}


def test_responses_streaming_error_finish_reason_emits_failed_event():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='',
            input_token_len=8,
            generate_token_len=0,
            finish_reason='error',
        )

    async def _collect_events():
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_PassthroughResponseParser(request),
            )
        ]

    import asyncio

    payloads = _sse_payloads(asyncio.run(_collect_events()))

    assert payloads[-1]['type'] == 'response.failed'
    assert payloads[-1]['response']['status'] == 'failed'
    assert payloads[-1]['response']['error']['code'] == 'server_error'


def test_responses_streaming_empty_output_announces_message_item():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    async def _result_generator():
        yield SimpleNamespace(
            response='',
            input_token_len=8,
            generate_token_len=0,
            finish_reason='stop',
        )

    async def _collect_events():
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_PassthroughResponseParser(request),
            )
        ]

    import asyncio

    payloads = _sse_payloads(asyncio.run(_collect_events()))

    assert any(payload['type'] == 'response.output_item.added'
               and payload['item']['type'] == 'message'
               for payload in payloads)
    assert any(payload['type'] == 'response.output_item.done'
               and payload['item']['type'] == 'message'
               for payload in payloads)
    assert payloads[-1]['type'] == 'response.completed'
    assert payloads[-1]['response']['output'][0]['type'] == 'message'
    assert payloads[-1]['response']['output'][0]['content'][0]['text'] == ''


def test_responses_streaming_tool_call_events():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    class _ToolParser:

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            if delta_text == 'tool-start':
                return [(
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='call_123',
                                function=DeltaFunctionCall(name='search', arguments='{"query":'),
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
                            function=DeltaFunctionCall(arguments='"lmdeploy"}'),
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
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_ToolParser(),
            )
        ]

    import asyncio

    events = asyncio.run(_collect_events())
    body = ''.join(events)
    payloads = _sse_payloads(events)

    assert 'event: response.output_item.added' in body
    assert 'event: response.function_call_arguments.delta' in body
    assert 'event: response.function_call_arguments.done' in body
    added = next(payload for payload in payloads if payload['type'] == 'response.output_item.added')
    done = next(payload for payload in payloads if payload['type'] == 'response.output_item.done')
    assert added['item']['type'] == 'function_call'
    assert added['item']['name'] == 'search'
    assert added['item']['status'] == 'in_progress'
    assert next(payload for payload in payloads
                if payload['type'] == 'response.function_call_arguments.done')['name'] == 'search'
    assert done['item']['arguments'] == '{"query":"lmdeploy"}'
    assert done['item']['status'] == 'completed'
    assert payloads[-1]['response']['output'][0]['type'] == 'function_call'
    assert payloads[-1]['response']['output'][0]['status'] == 'completed'


@pytest.mark.parametrize(('parallel_tool_calls', 'expected_call_ids'), [
    (False, ['call_123']),
    (None, ['call_123', 'call_456']),
])
def test_responses_streaming_parallel_tool_calls_filtering(parallel_tool_calls, expected_call_ids):
    request = ResponsesRequest(
        model='fake-model',
        input='Hi there',
        stream=True,
        parallel_tool_calls=parallel_tool_calls,
    )

    class _ParallelToolParser:

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            return [(
                DeltaMessage(
                    role='assistant',
                    tool_calls=[
                        DeltaToolCall(
                            index=0,
                            id='call_123',
                            function=DeltaFunctionCall(name='search', arguments='{}'),
                        ),
                        DeltaToolCall(
                            index=1,
                            id='call_456',
                            function=DeltaFunctionCall(name='lookup', arguments='{}'),
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
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_ParallelToolParser(),
            )
        ]

    import asyncio

    payloads = _sse_payloads(asyncio.run(_collect_events()))
    added_items = [
        payload['item'] for payload in payloads
        if payload['type'] == 'response.output_item.added'
    ]
    completed_output = payloads[-1]['response']['output']

    assert [item['call_id'] for item in added_items] == expected_call_ids
    assert [item['call_id'] for item in completed_output] == expected_call_ids


def test_responses_streaming_text_indices_follow_text_item_order():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    class _ToolThenTextParser:

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            if delta_text == 'tool':
                return [(
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='call_123',
                                function=DeltaFunctionCall(name='search', arguments='{}'),
                            )
                        ],
                    ),
                    True,
                )]
            return [(DeltaMessage(role='assistant', content='visible text'), False)]

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
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_ToolThenTextParser(),
            )
        ]

    import asyncio

    payloads = _sse_payloads(asyncio.run(_collect_events()))

    text_events = [
        payload for payload in payloads
        if payload['type'] in (
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


def test_responses_streaming_accepts_parser_delta_list():
    request = ResponsesRequest(model='fake-model', input='Hi there', stream=True)

    class _MultiDeltaParser:

        def stream_chunk(self, delta_text: str, delta_token_ids: list[int], **kwargs):
            return [
                (
                    DeltaMessage(
                        role='assistant',
                        tool_calls=[
                            DeltaToolCall(
                                index=0,
                                id='call_123',
                                function=DeltaFunctionCall(name='search', arguments='{}'),
                            )
                        ],
                    ),
                    True,
                ),
                (DeltaMessage(role='assistant', content='visible text'), False),
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
        return [
            event async for event in _stream_response(
                _result_generator(),
                request=request,
                model_name='fake-model',
                created_time=123,
                response_parser=_MultiDeltaParser(),
            )
        ]

    import asyncio

    completed_output = _sse_payloads(asyncio.run(_collect_events()))[-1]['response']['output']

    assert completed_output[0]['type'] == 'function_call'
    assert completed_output[1]['type'] == 'message'
    assert completed_output[1]['content'][0]['text'] == 'visible text'


def test_responses_openapi_router_is_included_with_openai_router():
    from fastapi import FastAPI

    from lmdeploy.serve.openai.api_server import router

    app = FastAPI()
    app.include_router(router)
    app.include_router(create_responses_router(None))

    assert '/v1/responses' in app.openapi()['paths']
