# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

from lmdeploy.serve.openai.protocol import FunctionCall, ToolCall
from lmdeploy.serve.openai.responses import ResponsesRequest
from lmdeploy.serve.openai.responses.response import make_response


def test_responses_non_stream_response_shape():
    request = ResponsesRequest(
        model='fake-model',
        input='Hi there',
    )

    response = make_response(
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

    response = make_response(
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

    error_response = make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        input_tokens=8,
        output_tokens=0,
        finish_reason='error',
    ).model_dump(exclude_none=True)
    abort_response = make_response(
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

    response = make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search',
                                      arguments='{"query":"lmdeploy"}'),
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
    request = ResponsesRequest(model='fake-model',
                               input='Hi',
                               parallel_tool_calls=False)

    response = make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search',
                                      arguments='{"query":"lmdeploy"}'),
            ),
            ToolCall(
                id='call_456',
                function=FunctionCall(name='lookup',
                                      arguments='{"query":"vllm"}'),
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
    request = ResponsesRequest(model='fake-model',
                               input='Hi',
                               parallel_tool_calls=None)

    response_model = make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text='',
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search',
                                      arguments='{"query":"lmdeploy"}'),
            ),
            ToolCall(
                id='call_456',
                function=FunctionCall(name='lookup',
                                      arguments='{"query":"vllm"}'),
            ),
        ],
        input_tokens=8,
        output_tokens=2,
        finish_reason='tool_calls',
    )
    response = response_model.model_dump(exclude_none=True)

    assert response_model.parallel_tool_calls is None
    assert [item['call_id']
            for item in response['output']] == ['call_123', 'call_456']


def test_responses_tool_call_response_accepts_no_visible_text():
    request = ResponsesRequest(model='fake-model', input='Hi')

    response = make_response(
        request=request,
        model_name='fake-model',
        created_time=123,
        text=None,
        tool_calls=[
            ToolCall(
                id='call_123',
                function=FunctionCall(name='search',
                                      arguments='{"query":"lmdeploy"}'),
            )
        ],
        input_tokens=8,
        output_tokens=2,
        finish_reason='tool_calls',
    ).model_dump(exclude_none=True)

    assert response['output_text'] == ''
    assert response['output'][0]['type'] == 'function_call'
