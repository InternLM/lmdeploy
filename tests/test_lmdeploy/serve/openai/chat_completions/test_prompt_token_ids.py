# Copyright (c) OpenMMLab. All rights reserved.
"""Chat prompt IDs across HTTP responses, SSE and parser buffering."""
import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.protocol import DeltaMessage


def _post(context, **options):
    app = FastAPI()
    register(app, context)
    payload = dict(model='fake-model', messages=[{'role': 'user', 'content': 'Hello'}])
    payload.update(options)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=payload)
    assert response.status_code == 200, response.text
    if not payload.get('stream'):
        return [response.json()]
    assert response.text.rstrip().endswith('data: [DONE]')
    return [json.loads(line[6:]) for line in response.text.splitlines()
            if line.startswith('data: ') and line != 'data: [DONE]']


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('n', [1, 2])
@pytest.mark.parametrize('return_prompt_text', [False, True])
@pytest.mark.parametrize('options, expected', [
    ({}, False),
    ({'return_token_ids': False}, False),
    ({'return_token_ids': None}, False),
    ({'return_token_ids': True}, True),
    ({'extra_body': {'return_token_ids': True}}, True),
    ({'return_token_ids': False, 'extra_body': {'return_token_ids': True}}, False),
    ({'return_token_ids': True, 'extra_body': {'return_token_ids': False}}, True),
])
def test_prompt_ids_http(chat_endpoint, stream, n, return_prompt_text, options, expected):
    _, context = chat_endpoint
    original_preprocess = context.async_engine.preprocess
    prompt_ids = [101, 102, 103, 104]
    rendered = '[user]Hello[assistant]'

    async def preprocess(messages, session, **kwargs):
        assert kwargs['reasoning_effort'] == 'high'
        result = await original_preprocess(messages, session, **kwargs)
        result.inputs = {'prompt': rendered, 'input_ids': prompt_ids}
        return result

    context.async_engine.preprocess = preprocess
    payloads = _post(context, stream=stream, n=n, reasoning_effort='high',
                     return_prompt_text=return_prompt_text,
                     stream_options={'include_usage': True} if stream else None, **options)
    seen = set()
    output_ids = {index: [] for index in range(n)}
    for payload in payloads:
        choices = payload['choices']
        if choices:
            first_choice = any(choice['index'] not in seen for choice in choices)
            assert ('prompt_token_ids' in payload) is (expected and first_choice)
            assert ('prompt_text' in payload) is (return_prompt_text and first_choice)
            if expected and first_choice:
                assert payload['prompt_token_ids'] == prompt_ids
            if return_prompt_text and first_choice:
                assert payload['prompt_text'] == rendered
        else:
            assert 'prompt_token_ids' not in payload
            assert 'prompt_text' not in payload
        for choice in choices:
            seen.add(choice['index'])
            if expected:
                output_ids[choice['index']].extend(choice['output_ids'])
            elif stream:
                assert 'output_ids' not in choice
            else:
                # Adding a response field must not drop existing null fields.
                assert choice['output_ids'] is None
                assert choice['logprobs'] is None
    assert seen == set(range(n))
    if expected:
        assert output_ids == {index: [index + 1] for index in range(n)}
    if stream:
        assert payloads[-1]['choices'] == []
        assert payloads[-1]['usage']['prompt_tokens'] == 4
    else:
        assert payloads[0]['service_tier'] is None
    assert len(context.session_manager.removed) == n
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('n', [1, 2])
def test_raw_input_ids_have_ids_without_rendered_prompt(chat_endpoint, stream, n):
    _, context = chat_endpoint
    input_ids = [91, 92, 93]
    original_preprocess = context.async_engine.preprocess

    async def preprocess(messages, session, **kwargs):
        assert messages is None
        assert kwargs['input_ids'] == input_ids
        assert kwargs['do_preprocess'] is False
        result = await original_preprocess(messages, session, **kwargs)
        result.inputs = {'input_ids': list(kwargs['input_ids'])}
        return result

    context.async_engine.preprocess = preprocess
    payloads = _post(context, messages=[], input_ids=input_ids, stream=stream, n=n,
                     return_token_ids=True, return_prompt_text=True)
    assert payloads
    for payload in payloads:
        assert payload['prompt_token_ids'] == input_ids
        assert payload['prompt_text'] is None
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('stream', [False, True])
def test_prompt_ids_preserve_buffered_output_ids_and_logprobs(chat_endpoint, stream, monkeypatch):
    _, context = chat_endpoint

    class BufferingParser(context.response_parser_cls):

        def stream_chunk(self, text, ids, **kwargs):
            if ids == [201]:
                return []
            if ids == [202, 203]:
                return [(DeltaMessage(content='<arg_'), False), (DeltaMessage(content='key>'), False)]
            return []

        def parse_complete(self, text, token_ids=None, **kwargs):
            assert token_ids == [201, 202, 203]
            return '<arg_key>', None, None

    context.response_parser_cls = BufferingParser
    engine = context.async_engine
    monkeypatch.setattr(engine.backend_config, 'logprobs_mode', 'raw_logprobs')
    engine.tokenizer.model.model = SimpleNamespace(
        convert_ids_to_tokens=lambda token_id: {201: '<arg_key>', 202: '<arg_', 203: 'key>'}[token_id])
    # A control ID and ordinary IDs can decode to the same text. Preserve
    # both even when the parser hides one or splits another engine result.
    rows = [('<arg_key>', [201]), ('<arg_key>', [202, 203]), ('', [])]

    async def generate(preprocessed, **kwargs):
        for text, ids in rows:
            yield SimpleNamespace(response=text, token_ids=ids, input_token_len=4,
                                  generate_token_len=3, finish_reason='stop' if not ids else None,
                                  logprobs=[{token_id: -0.1} for token_id in ids], cached_tokens=0,
                                  routed_experts=None, cache_block_ids=None)

    engine.generate = generate
    payloads = _post(context, stream=stream, return_token_ids=True, return_logprob=True,
                     logprobs=True, top_logprobs=1)
    assert payloads[0]['prompt_token_ids'] == [101, 102, 103, 104]
    assert all('prompt_token_ids' not in payload for payload in payloads[1:])
    choices = [choice for payload in payloads for choice in payload['choices']]
    assert [token_id for choice in choices for token_id in choice.get('output_ids', [])] == [201, 202, 203]
    assert [score for choice in choices for score in choice.get('output_token_logprobs', [])] == [
        [-0.1, 201], [-0.1, 202], [-0.1, 203]]
    scores = [score for choice in choices for score in (choice.get('logprobs') or {}).get('content', [])]
    assert [score['token'] for score in scores] == ['<arg_key>', '<arg_', 'key>']
    if stream:
        assert [choice.get('output_ids', []) for choice in choices] == [[201], [], [202, 203], []]
        assert choices[0]['delta'].get('content', '') == ''
        assert ''.join(choice['delta'].get('content', '') for choice in choices) == '<arg_key>'
    else:
        assert choices[0]['message']['content'] == '<arg_key>'
    assert context.session_manager.sessions == {}


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('n', [1, 2])
def test_prompt_ids_survive_empty_terminal_output(chat_endpoint, stream, n):
    _, context = chat_endpoint

    async def generate(preprocessed, **kwargs):
        # A stop token may be filtered before the first visible output.
        yield SimpleNamespace(response='', token_ids=[], input_token_len=4,
                              generate_token_len=0, finish_reason='stop',
                              logprobs=None, cached_tokens=0, routed_experts=None, cache_block_ids=None)

    context.async_engine.generate = generate
    payloads = _post(context, stream=stream, n=n, return_token_ids=True,
                     stream_options={'include_usage': True} if stream else None)
    seen = set()
    for payload in payloads:
        if not payload['choices']:
            assert 'prompt_token_ids' not in payload
            continue
        assert payload['prompt_token_ids'] == [101, 102, 103, 104]
        for choice in payload['choices']:
            assert choice['index'] not in seen
            seen.add(choice['index'])
            assert choice['output_ids'] == []
            assert choice['finish_reason'] == 'stop'
    assert seen == set(range(n))
    assert payloads[-1]['usage']['completion_tokens'] == 0
    assert len(context.session_manager.removed) == n
    assert context.session_manager.sessions == {}
