# Copyright (c) OpenMMLab. All rights reserved.

from __future__ import annotations

import asyncio
import json

import pytest

from lmdeploy.serve.openai.responses import ResponsesRequest


@pytest.mark.parametrize(
    ('field_name', 'value'),
    [
        pytest.param('temperature', -0.1, id='temperature-below-range'),
        pytest.param('temperature', 2.1, id='temperature-above-range'),
        pytest.param('top_p', -0.1, id='top-p-below-range'),
        pytest.param('top_p', 1.1, id='top-p-above-range'),
        pytest.param('top_k', -1, id='negative-top-k'),
        pytest.param('min_p', -0.1, id='min-p-below-range'),
        pytest.param('min_p', 1.1, id='min-p-above-range'),
    ],
)
def test_responses_rejects_invalid_sampling_parameter(
        responses_endpoint, fake_raw_request, field_name, value):
    endpoint, _ = responses_endpoint
    response = asyncio.run(
        endpoint(
            ResponsesRequest(model='fake-model', input='Hi', **{field_name: value}),
            fake_raw_request,
        ))

    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == field_name


def test_responses_rejects_empty_input_list(responses_endpoint,
                                            fake_raw_request):
    endpoint, _ = responses_endpoint
    response = asyncio.run(
        endpoint(
            ResponsesRequest(model='fake-model', input=[]),
            fake_raw_request,
        ))

    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'input'


def test_responses_tool_validation_uses_tools_error_param(
        responses_endpoint, fake_raw_request):
    endpoint, _ = responses_endpoint
    response = asyncio.run(
        endpoint(
            ResponsesRequest(
                model='fake-model',
                input='Hi',
                tools=[{
                    'type': 'function',
                }],
            ),
            fake_raw_request,
        ))

    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'tools'


def test_responses_tool_choice_validation_uses_tool_choice_error_param(
        responses_endpoint, fake_raw_request):
    endpoint, _ = responses_endpoint
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
            fake_raw_request,
        ))

    assert response.status_code == 400
    assert json.loads(response.body)['error']['param'] == 'tool_choice'


def test_responses_tool_choice_none_does_not_require_tool_parser(
        responses_endpoint, fake_raw_request):
    endpoint, context = responses_endpoint
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
            fake_raw_request,
        ))

    assert response['output_text'] == 'ok'
    assert context.async_engine.preprocess_kwargs['tools'] is None


def test_responses_tools_require_tool_parser(responses_endpoint,
                                             fake_raw_request):
    endpoint, _ = responses_endpoint
    response = asyncio.run(
        endpoint(
            ResponsesRequest(
                model='fake-model',
                input='Hi',
                tools=[{
                    'type': 'function',
                    'name': 'search',
                }],
            ),
            fake_raw_request,
        ))

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body['error']['param'] == 'tools'
    assert '--tool-call-parser' in body['error']['message']


def test_responses_non_streaming_cleans_up_session(
        responses_endpoint, fake_raw_request):
    endpoint, context = responses_endpoint

    response = asyncio.run(
        endpoint(ResponsesRequest(model='fake-model', input='Hi'),
                 fake_raw_request))

    assert response['output_text'] == 'ok'
    assert context.async_engine.session_mgr.removed == context.sessions


def test_responses_non_streaming_reports_reasoning_tokens(
        responses_endpoint, fake_raw_request, passthrough_response_parser_cls):
    class _ReasoningResponseParser(passthrough_response_parser_cls):
        reasoning_tokens = 3

    endpoint, context = responses_endpoint
    context.response_parser_cls = _ReasoningResponseParser

    response = asyncio.run(
        endpoint(ResponsesRequest(model='fake-model', input='Hi'),
                 fake_raw_request))

    assert response['usage']['output_tokens_details']['reasoning_tokens'] == 3


def test_responses_non_streaming_disconnect_cleans_up_session(
        responses_endpoint):
    class _DisconnectedRawRequest:

        async def is_disconnected(self):
            return True

    endpoint, context = responses_endpoint

    response = asyncio.run(
        endpoint(ResponsesRequest(model='fake-model', input='Hi'),
                 _DisconnectedRawRequest()))

    assert response.status_code == 400
    assert context.sessions[0].aborted is True
    assert context.async_engine.session_mgr.removed == context.sessions


def test_responses_streaming_cleans_up_session(responses_endpoint,
                                               fake_raw_request):
    endpoint, context = responses_endpoint

    async def _collect_stream():
        response = await endpoint(
            ResponsesRequest(model='fake-model', input='Hi', stream=True),
            fake_raw_request)
        return [event async for event in response.body_iterator]

    events = asyncio.run(_collect_stream())

    assert any('event: response.completed' in event for event in events)
    assert context.async_engine.session_mgr.removed == context.sessions


def test_responses_unconsumed_streaming_response_cleans_up_session(
        responses_endpoint, fake_raw_request):
    endpoint, context = responses_endpoint

    async def _close_without_consuming():
        response = await endpoint(
            ResponsesRequest(model='fake-model', input='Hi', stream=True),
            fake_raw_request)
        await response.close()

    asyncio.run(_close_without_consuming())

    assert context.async_engine.session_mgr.removed == context.sessions


def test_responses_uses_parser_adjusted_messages_for_generation(
        responses_endpoint, fake_raw_request, passthrough_response_parser_cls):

    class _AdjustingResponseParser(passthrough_response_parser_cls):

        adjusted_messages = [{'role': 'user', 'content': 'adjusted'}]

        def __init__(self, request):
            super().__init__(request)
            self.request.messages = self.adjusted_messages

    endpoint, context = responses_endpoint
    context.response_parser_cls = _AdjustingResponseParser

    response = asyncio.run(
        endpoint(ResponsesRequest(model='fake-model', input='original'),
                 fake_raw_request))

    assert response['output_text'] == 'ok'
    assert context.async_engine.prompt is _AdjustingResponseParser.adjusted_messages


def test_responses_rejects_non_string_text_content_parts(
        responses_endpoint, fake_raw_request):
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

    endpoint, _ = responses_endpoint
    response = asyncio.run(endpoint(request, fake_raw_request))

    assert response.status_code == 400
    body = json.loads(response.body)
    assert body['error']['param'] == 'input'
    assert 'Expected string' in body['error']['message']


def test_responses_forwards_repetition_penalty(responses_endpoint,
                                               fake_raw_request):
    endpoint, context = responses_endpoint
    request = ResponsesRequest(
        model='fake-model',
        input='Hi',
        repetition_penalty=1.1,
    )

    response = asyncio.run(endpoint(request, fake_raw_request))

    assert response['output_text'] == 'ok'
    assert context.async_engine.preprocess_kwargs[
        'gen_config'].repetition_penalty == 1.1


def test_responses_preserves_unspecified_generation_bool_defaults(
        responses_endpoint, fake_raw_request):
    endpoint, context = responses_endpoint
    context.default_gen_config = {
        'ignore_eos': True,
        'skip_special_tokens': False,
        'include_stop_str_in_output': True,
    }

    response = asyncio.run(
        endpoint(ResponsesRequest(model='fake-model', input='Hi'),
                 fake_raw_request))

    gen_config = context.async_engine.preprocess_kwargs['gen_config']
    assert response['output_text'] == 'ok'
    assert gen_config.ignore_eos is True
    assert gen_config.skip_special_tokens is False
    assert gen_config.include_stop_str_in_output is True


def test_responses_forwards_explicit_generation_bool_fields(
        responses_endpoint, fake_raw_request):
    endpoint, context = responses_endpoint
    context.default_gen_config = {
        'ignore_eos': True,
        'skip_special_tokens': False,
        'include_stop_str_in_output': True,
    }

    response = asyncio.run(
        endpoint(
            ResponsesRequest(
                model='fake-model',
                input='Hi',
                ignore_eos=False,
                skip_special_tokens=True,
                include_stop_str_in_output=False,
            ),
            fake_raw_request,
        ))

    gen_config = context.async_engine.preprocess_kwargs['gen_config']
    assert response['output_text'] == 'ok'
    assert gen_config.ignore_eos is False
    assert gen_config.skip_special_tokens is True
    assert gen_config.include_stop_str_in_output is False


def test_responses_parser_request_uses_max_completion_tokens(
        responses_endpoint, fake_raw_request, passthrough_response_parser_cls):
    endpoint, _ = responses_endpoint
    request = ResponsesRequest(model='fake-model',
                               input='Hi',
                               max_output_tokens=17)

    asyncio.run(endpoint(request, fake_raw_request))

    assert passthrough_response_parser_cls.last_request.max_completion_tokens == 17


def test_responses_engine_error_is_not_returned_as_bad_request(
        responses_endpoint, fake_raw_request):
    endpoint, context = responses_endpoint

    def _generate(_request, **kwargs):
        async def _generator():
            if False:
                yield None
            raise RuntimeError('engine exploded')

        return _generator()

    context.async_engine.generate = _generate

    with pytest.raises(RuntimeError, match='engine exploded'):
        asyncio.run(endpoint(ResponsesRequest(model='fake-model', input='Hi'), fake_raw_request))
