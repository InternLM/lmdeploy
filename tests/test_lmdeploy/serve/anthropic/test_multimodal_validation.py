# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest

from lmdeploy.serve.anthropic.adapter import to_openai_messages
from lmdeploy.serve.anthropic.protocol import MessagesRequest


@pytest.mark.parametrize('block', [
    {'type': 'image'},
    {'type': 'image', 'source': {'type': 'file', 'file_id': 'file_123'}},
    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': ''}},
    {'type': 'image', 'source': {'type': 'base64', 'media_type': 'text/plain', 'data': 'AAAA'}},
    {'type': 'image', 'source': {'type': 'url', 'url': ''}},
])
@pytest.mark.parametrize('in_tool_result', [False, True])
def test_invalid_media_blocks_are_rejected_without_mutating_request(block, in_tool_result):
    content = ([{'type': 'tool_result', 'tool_use_id': 'call_1', 'content': [block]}]
               if in_tool_result else [block])
    request = MessagesRequest(model='test', max_tokens=16, messages=[{'role': 'user', 'content': content}])
    before = copy.deepcopy(request.model_dump())
    with pytest.raises(ValueError):
        to_openai_messages(request)
    assert request.model_dump() == before


def test_unknown_tool_result_blocks_are_not_silently_dropped():
    request = MessagesRequest(model='test', max_tokens=16, messages=[{
        'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1',
                                   'content': [{'type': 'unknown_block', 'text': 'preserve me'}]}]}])
    with pytest.raises(ValueError, match='Unsupported tool_result'):
        to_openai_messages(request)


@pytest.mark.parametrize('source', [
    {'type': 'url', 'url': 'https://example.com/image.png'},
    {'type': 'base64', 'media_type': 'image/png', 'data': 'AAAA'},
])
def test_supported_media_remains_in_its_tool_result(source):
    request = MessagesRequest(model='test', max_tokens=16, messages=[{
        'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'call_1',
                                   'content': [{'type': 'image', 'source': source}]}]}])
    messages = to_openai_messages(request)
    assert len(messages) == 1
    assert messages[0]['role'] == 'tool'
    assert messages[0]['tool_call_id'] == 'call_1'
    assert messages[0]['content'][0]['type'] == 'image_url'
