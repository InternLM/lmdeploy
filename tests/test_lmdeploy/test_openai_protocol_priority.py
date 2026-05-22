import pytest
from pydantic import ValidationError

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, CompletionRequest


def test_chat_completion_request_priority_extension():
    assert ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'hi'}]).priority == 0
    assert ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'hi'}],
                                 priority=255).priority == 255
    assert ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'hi'}],
                                 priority=None).priority is None

    for priority in (-1, 256, True, 1.5, '1'):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(model='m', messages=[{'role': 'user', 'content': 'hi'}], priority=priority)


def test_completion_request_priority_extension():
    assert CompletionRequest(model='m', prompt='hi').priority == 0
    assert CompletionRequest(model='m', prompt='hi', priority=5).priority == 5
    assert CompletionRequest(model='m', prompt='hi', priority=None).priority is None

    for priority in (-1, 256, True, 1.5, '1'):
        with pytest.raises(ValidationError):
            CompletionRequest(model='m', prompt='hi', priority=priority)
