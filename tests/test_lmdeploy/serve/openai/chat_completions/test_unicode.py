# Copyright (c) OpenMMLab. All rights reserved.
"""Check UTF-8 on the HTTP wire as well as the decoded JSON values."""
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.core.exceptions import ErrorCode, RequestError
from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.protocol import DeltaMessage


@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('channel', ['content', 'reasoning_content', 'tool_calls'])
def test_unicode_output(chat_endpoint, n, stream, channel):
    _, context = chat_endpoint
    text = '你好，世界 🌍 café\n第二行\t"引号"\\路径\r\n分隔:\u0085\u2028\u2029 字面量: \\u4f60'
    arguments = json.dumps({'text': text}, ensure_ascii=False)
    tool_call = dict(id='call-1', type='function', function=dict(name='lookup', arguments=arguments))

    class Parser(context.response_parser_cls):

        def stream_chunk(self, delta_text, delta_token_ids, **kwargs):
            if channel == 'tool_calls':
                return [(DeltaMessage(tool_calls=[dict(index=0, **tool_call)]), False)]
            return [(DeltaMessage(**{channel: delta_text}), False)]

        def parse_complete(self, text, token_ids=None, **kwargs):
            if channel == 'tool_calls':
                return None, [tool_call], None
            if channel == 'reasoning_content':
                return None, None, text
            return text, None, None

    context.response_parser_cls = Parser
    context.async_engine.model_name = 'fake-模型'
    original_generate = context.async_engine.generate

    def generate(request, **kwargs):
        async def results():
            async for result in original_generate(request, **kwargs):
                result.response = text
                yield result
        return results()

    context.async_engine.generate = generate
    app = FastAPI()
    register(app, context)
    payload = dict(model='fake-模型', messages=[dict(role='user', content='你好')], n=n, stream=stream)
    if stream:
        payload['stream_options'] = dict(include_usage=True)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)

    assert response.status_code == 200
    wire_text = response.content.decode('utf-8')
    if stream:
        assert response.headers['content-type'] == 'text/event-stream; charset=utf-8'
        # SSE lines are delimited by LF, not Unicode line/paragraph separators.
        events = [line[6:] for line in wire_text.split('\n') if line.startswith('data: ')]
        assert events[-1] == '[DONE]'
        assert events.count('[DONE]') == 1
        chunks = [json.loads(event) for event in events[:-1]]
        messages = {choice['index']: choice['delta'] for chunk in chunks for choice in chunk['choices']
                    if channel in choice['delta']}
        usage = [chunk for chunk in chunks if chunk.get('usage') is not None]
        assert len(usage) == 1
        assert usage[0]['choices'] == []
        assert all('模型' in event for event in events[:-1])
    else:
        messages = {choice['index']: choice['message'] for choice in response.json()['choices']}

    assert set(messages) == set(range(n))
    for message in messages.values():
        if channel == 'tool_calls':
            assert message[channel][0]['function']['arguments'] == arguments
            assert json.loads(message[channel][0]['function']['arguments']) == {'text': text}
        else:
            assert message[channel] == text
    assert '你好'.encode() in response.content
    assert '🌍'.encode() in response.content
    assert context.session_manager.sessions == {}


def test_unicode_stream_error(chat_endpoint):
    _, context = chat_endpoint
    message = '请求失败：中文错误'

    async def generate(request, **kwargs):
        raise RequestError(ErrorCode.INVALID_REQUEST, message)
        yield  # pragma: no cover

    context.async_engine.generate = generate
    app = FastAPI()
    register(app, context)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=dict(
            model='fake-model', messages=[dict(role='user', content='你好')], stream=True))

    assert response.status_code == 200
    events = [line[6:] for line in response.content.decode('utf-8').split('\n') if line.startswith('data: ')]
    assert len(events) == 2
    assert json.loads(events[0])['error']['message'] == message
    assert events[1] == '[DONE]'
    assert message.encode() in response.content
    assert context.session_manager.sessions == {}
