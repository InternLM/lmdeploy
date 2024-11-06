import pytest
from pydantic import ValidationError

from lmdeploy.serve.openai.protocol import ChatCompletionRequestMessage


def test_single_str_input():
    messages = 'hello'
    _ = ChatCompletionRequestMessage(messages=messages)


@pytest.mark.parametrize('role', ['system', 'user', 'assistant'])
def test_list_str_input(role):
    content = 'hello'
    messages = [dict(role=role, content=content)]
    _ = ChatCompletionRequestMessage(messages=messages)


@pytest.mark.parametrize('role', ['system', 'user', 'assistant'])
def test_list_content_input(role):
    content = [dict(type='text', text='hello')]
    messages = [dict(role=role, content=content)]
    _ = ChatCompletionRequestMessage(messages=messages)


def test_user_image_input():
    content = [dict(type='image_url', image_url=dict(url='xxx'))]
    messages = [dict(role='user', content=content)]
    _ = ChatCompletionRequestMessage(messages=messages)


@pytest.mark.parametrize('role', ['system', 'assistant'])
def test_system_assistant_image_input(role):
    content = [dict(type='image_url', image_url=dict(url='xxx'))]
    messages = [dict(role=role, content=content)]
    with pytest.raises(ValidationError):
        _ = ChatCompletionRequestMessage(messages=messages)
