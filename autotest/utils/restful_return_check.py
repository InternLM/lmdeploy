import re
from typing import Any

import requests
from openai import OpenAI
from utils.config_utils import (
    _entry_engine_config,
    get_model_path_from_config,
    iter_model_yaml_entries,
)
from utils.constant import BASE_URL
from utils.toolkit import _load_tokenizer_cached, encode_text

# Preprocess rejects oversize input with this OpenAI error substring.
CONTEXT_LENGTH_ERROR = 'context length'

# Upper bound for "large but valid" payload tests (CI scale, not full context).
CI_LARGE_PAYLOAD_TOKEN_CAP = 32_000

# Legacy anthropic large-payload test body size (128 KiB repeated filler).
CI_LARGE_PAYLOAD_CHAR_BUDGET = 128 * 1024


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


def resolve_effective_session_len(config: dict[str, Any], model_id: str) -> int:
    """Context limit aligned with ``async_engine.session_len``.

    Uses yaml ``session-len`` when set, otherwise HF ``_get_and_verify_max_len``.
    Does not apply ``tokenizer.model_max_length`` (server preprocess does not either).
    """
    model_path = get_model_path_from_config(config, model_id)
    session_len = None
    for entry in iter_model_yaml_entries(model_id):
        extra = _entry_engine_config(entry).get('extra') or {}
        if extra.get('session-len') is not None:
            session_len = int(extra['session-len'])
            break
    if session_len is None:
        from transformers import AutoConfig

        from lmdeploy.utils import _get_and_verify_max_len

        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        session_len = _get_and_verify_max_len(hf_cfg, None)
    return session_len


def build_session_sized_user_content(
    *,
    config: dict[str, Any],
    model_id: str,
    oversize: bool = False,
    max_completion_tokens: int = 0,
    reserve: int = 256,
    slack: int = 32,
    unit: str = 'x',
    token_cap: int | None = CI_LARGE_PAYLOAD_TOKEN_CAP,
) -> str:
    """Size user text relative to server ``session_len`` (local tokenizer).

    ``oversize=True``: raw user token count exceeds server ``session_len`` (400 tests).
    ``oversize=False``: as large as possible while fitting; also caps by finite
    ``tokenizer.model_max_length`` when tighter than HF context (e.g. InternVL3-38B).
    """
    session_len = resolve_effective_session_len(config, model_id)
    model_path = get_model_path_from_config(config, model_id)
    if not oversize:
        tok_mml = getattr(_load_tokenizer_cached(model_path), 'model_max_length', None)
        if tok_mml is not None and tok_mml < 1_000_000:
            session_len = min(session_len, int(tok_mml))
    text = ''
    token_len = len(encode_text(model_path, text, add_special_tokens=False))

    if oversize:
        target = session_len + slack + 1
        while token_len < target:
            deficit = target - token_len
            text += unit * max(deficit, 1)
            token_len = len(encode_text(model_path, text, add_special_tokens=False))
        return text

    input_limit = session_len - max_completion_tokens - reserve
    if token_cap is not None:
        input_limit = min(input_limit, token_cap)
    while token_len < input_limit:
        deficit = input_limit - token_len
        text += unit * max(deficit, 1)
        token_len = len(encode_text(model_path, text, add_special_tokens=False))

    session_input_limit = session_len - max_completion_tokens
    while token_len >= session_input_limit and text:
        text = text[:-(max(1, len(text) // 20))]
        token_len = len(encode_text(model_path, text, add_special_tokens=False))
    return text


def cap_completion_tokens_for_session(
    prompt_text: str,
    default_cap: int,
    *,
    config: dict[str, Any],
    model_id: str,
    reserve: int = 128,
    min_cap: int = 64,
) -> int:
    """Cap ``max_tokens`` so prompt + completion fits ``session_len``."""
    session_len = resolve_effective_session_len(config, model_id)
    model_path = get_model_path_from_config(config, model_id)
    prompt_tokens = len(encode_text(model_path, prompt_text, add_special_tokens=False))
    available = session_len - prompt_tokens - reserve
    return min(default_cap, max(min_cap, available))
