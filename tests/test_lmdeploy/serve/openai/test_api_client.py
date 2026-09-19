# Copyright (c) OpenMMLab. All rights reserved.
import json
from unittest.mock import Mock

import pytest

from lmdeploy.serve.openai.api_client import APIClient
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('extra_parameters', [False, True])
@pytest.mark.parametrize('chat', [False, True])
def test_completion_request_parameters(monkeypatch, stream, extra_parameters, chat):
    output = {'id': 'completion', 'choices': []}
    encoded = json.dumps(output).encode('utf-8')
    response = Mock()
    response.iter_lines.return_value = [b'data: ' + encoded, b'data: [DONE]'] if stream else [encoded]
    post = Mock(return_value=response)
    monkeypatch.setattr('lmdeploy.serve.openai.api_client.requests.post', post)

    client = APIClient('http://localhost:23333')
    parameters = {'model': 'test-model', 'stream': stream, 'temperature': 0.2}
    extra = {'seed': 7} if extra_parameters else {}
    if chat:
        parameters['messages'] = [{'role': 'user', 'content': 'Hello'}]
        if extra_parameters:
            extra['chat_template_kwargs'] = {'enable_thinking': False}
            extra['response_format'] = {'type': 'json_object'}
        results = list(client.chat_completions_v1(**parameters, **extra))
        request_type = ChatCompletionRequest
        path = '/v1/chat/completions'
    else:
        parameters['prompt'] = 'Hello'
        if extra_parameters:
            extra['min_p'] = 0.1
        results = list(client.completions_v1(**parameters, **extra))
        request_type = CompletionRequest
        path = '/v1/completions'

    assert results == [output]
    post.assert_called_once()
    assert post.call_args.args == (client.api_server_url + path,)
    assert post.call_args.kwargs['stream'] is stream
    payload = post.call_args.kwargs['json']
    assert 'kwargs' not in payload
    assert all(payload[key] == value for key, value in parameters.items())
    assert all(payload[key] == value for key, value in extra.items())
    request = request_type.model_validate(payload)
    assert request.seed == (7 if extra_parameters else None)
    if chat and extra_parameters:
        assert request.chat_template_kwargs == {'enable_thinking': False}
        assert request.response_format.type == 'json_object'
    elif extra_parameters:
        assert request.min_p == 0.1
