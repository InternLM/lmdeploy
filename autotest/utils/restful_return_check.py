import re

import requests
from openai import OpenAI
from utils.constant import BASE_URL

# Preprocess rejects oversize input with this OpenAI error substring.
CONTEXT_LENGTH_ERROR = 'context length'


def assert_openai_invalid_request_error(
    response: requests.Response | dict,
    *,
    message_substr: str | None = None,
) -> dict:
    """Assert OpenAI invalid request (HTTP 400, ``invalid_request_error``).

    Accepts a ``requests.Response`` (raw HTTP) or an error body ``dict``
    (e.g. OpenAI SDK ``BadRequestError.body``).
    """
    if isinstance(response, requests.Response):
        assert response.status_code == 400, (
            f'expected 400, got {response.status_code}: {response.text[:500]}')
        body = response.json()
    else:
        body = response

    assert body.get('object') == 'error'
    assert body.get('type') == 'invalid_request_error'
    assert body.get('code') == 400
    message = body.get('message')
    assert message
    if message_substr is not None:
        assert message_substr.lower() in message.lower()
    return body


def get_chat_message_text(choice):
    msg = choice.get('message') or {}
    texts = []
    for key in ('reasoning_content', 'content'):
        value = msg.get(key)
        if isinstance(value, str):
            texts.append(value)
    return ''.join(texts)


def get_chat_delta_text(choice):
    delta = choice.get('delta') or {}
    texts = []
    for key in ('reasoning_content', 'content'):
        value = delta.get(key)
        if isinstance(value, str):
            texts.append(value)
    return ''.join(texts)


def assert_chat_message_empty(choice):
    assert not get_chat_message_text(choice)


def assert_chat_delta_empty(choice):
    assert not get_chat_delta_text(choice)


def assert_chat_completions_batch_return(output, model_name, check_logprobs: bool = False, logprobs_num: int = 5):
    assert_usage(output.get('usage'))
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length']
        assert message.get('index') == 0
        msg = message.get('message') or {}
        content = msg.get('content')
        reasoning = msg.get('reasoning_content')
        assert (isinstance(content, str) and len(content) > 0) or (
            isinstance(reasoning, str) and len(reasoning) > 0)
        assert msg.get('role') == 'assistant'
        if check_logprobs:
            len(message.get('logprobs').get('content')) == output.get('usage').get('completion_tokens')
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_completions_batch_return(output, model_name, check_logprobs: bool = False, logprobs_num: int = 5):
    assert_usage(output.get('usage'))
    assert output.get('id') is not None
    assert output.get('object') == 'text_completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('finish_reason') in ['stop', 'length']
        assert message.get('index') == 0
        assert len(message.get('text')) > 0
        if check_logprobs:
            len(message.get('logprobs').get('content')) == output.get('usage').get('completion_tokens')
            for logprob in message.get('logprobs').get('content'):
                assert_logprobs(logprob, logprobs_num)


def assert_usage(usage):
    assert usage.get('prompt_tokens') > 0
    assert usage.get('total_tokens') > 0
    assert usage.get('completion_tokens') > 0
    assert usage.get('completion_tokens') + usage.get('prompt_tokens') == usage.get('total_tokens')


def assert_logprobs(logprobs, logprobs_num):
    assert_logprob_element(logprobs)
    assert len(logprobs.get('top_logprobs')) >= 0
    assert type(logprobs.get('top_logprobs')) is list
    assert len(logprobs.get('top_logprobs')) <= logprobs_num
    for logprob_element in logprobs.get('top_logprobs'):
        assert_logprob_element(logprob_element)


def assert_logprob_element(logprob):
    assert len(logprob.get('token')) > 0 and type(logprob.get('token')) is str
    assert len(logprob.get('bytes')) > 0 and type(logprob.get('bytes')) is list
    assert type(logprob.get('logprob')) is float


def assert_chat_completions_stream_return(output,
                                          model_name,
                                          is_last: bool = False,
                                          check_logprobs: bool = False,
                                          logprobs_num: int = 5):
    print(output)
    assert output.get('id') is not None
    assert output.get('object') == 'chat.completion.chunk'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('delta').get('role') == 'assistant'
        assert message.get('index') == 0
        delta = message.get('delta') or {}
        assert isinstance(delta.get('content'), str) or isinstance(delta.get('reasoning_content'), str)
        if not is_last:
            assert message.get('finish_reason') is None
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) >= 1)
                for content in message.get('logprobs').get('content'):
                    assert_logprobs(content, logprobs_num)
        if is_last is True:
            content = delta.get('content')
            reasoning = delta.get('reasoning_content')
            assert content is None or len(content) == 0 or 'error' in content
            assert reasoning is None or len(reasoning) == 0 or 'error' in reasoning
            assert message.get('finish_reason') in ['stop', 'length', 'error']
            if check_logprobs is True:
                assert message.get('logprobs') is None


def assert_completions_stream_return(output,
                                     model_name,
                                     is_last: bool = False,
                                     check_logprobs: bool = False,
                                     logprobs_num: int = 5):
    print(output)
    assert output.get('id') is not None
    assert output.get('object') == 'text_completion'
    assert output.get('model') == model_name
    output_message = output.get('choices')
    assert len(output_message) == 1
    for message in output_message:
        assert message.get('index') == 0
        assert len(message.get('text')) >= 0
        if is_last is False:
            assert message.get('finish_reason') is None
            if check_logprobs:
                assert (len(message.get('logprobs').get('content')) >= 1)
                for content in message.get('logprobs').get('content'):
                    assert_logprobs(content, logprobs_num)

        if is_last is True:
            assert len(message.get('text')) == 0
            assert message.get('finish_reason') in ['stop', 'length']
            if check_logprobs is True:
                assert message.get('logprobs') is None


def has_repeated_fragment(text, repeat_count=5):
    pattern = r'(.+?)\1{' + str(repeat_count - 1) + ',}'
    match = re.search(pattern, text.replace('\n', ''))
    if match:
        repeated_fragment = match.group(1)
        start_pos = match.start()
        return True, {'repeated_fragment': repeated_fragment, 'position': start_pos}
    return False, f'{text} does not contain repeated fragments'


def get_client_and_model(base_url: str | None = None) -> tuple[OpenAI, str]:
    """Return ``(OpenAI client, deployed model id)`` for a running
    api_server."""
    url = base_url or BASE_URL
    client = OpenAI(api_key='YOUR_API_KEY', base_url=f'{url.rstrip("/")}/v1')
    models = client.models.list().data
    if not models:
        raise RuntimeError(f'No model returned from GET {url}/v1/models')
    return client, models[0].id


def encode_prompt(base_url: str, text: str, *, add_bos: bool = True) -> tuple[list, int]:
    """Tokenize via ``POST /v1/encode``; returns ``(input_ids, length)``."""
    url = base_url.rstrip('/')
    response = requests.post(
        f'{url}/v1/encode',
        json={'input': text, 'do_preprocess': False, 'add_bos': add_bos},
        timeout=30,
    )
    response.raise_for_status()
    output = response.json()
    return output['input_ids'], output['length']
