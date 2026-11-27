# Copyright (c) OpenMMLab. All rights reserved.
"""Exercise Responses reasoning with the real GLM template and parser."""

import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from lmdeploy.serve.openai.responses import create_responses_router
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.serve.parsers.tool_parser import ToolParserManager
from lmdeploy.serve.processors.multimodal import MultimodalProcessor
from lmdeploy.vl.model.base import VisionModel


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('with_image', [False, True])
@pytest.mark.parametrize('effort', [None, 'none'])
def test_low_budget_responses(responses_endpoint, hf_media_template, sse_payloads, stream, with_image, effort):
    _, context = responses_endpoint
    template = hf_media_template('glm5_3_flash')

    class Parser(BaseResponseParser):
        reasoning_parser_cls = ReasoningParserManager.get('default')
        tokenizer = SimpleNamespace(get_vocab=lambda: {'<think>': 1, '</think>': 2})

    context.response_parser_cls = Parser
    original_preprocess = context.async_engine.preprocess
    budget = 64 if with_image else 32
    answer = 'Red' if with_image else 'ok'

    async def preprocess(messages, session, **kwargs):
        result = await original_preprocess(messages, session, **kwargs)
        parsed = await MultimodalProcessor.async_parse_multimodal_item(messages)
        result.rendered_prompt = VisionModel.get_input_prompt(
            parsed, template, chat_template_kwargs=kwargs['chat_template_kwargs'])
        assert kwargs['gen_config'].max_new_tokens == budget
        if with_image:
            image = next(p['data'] for p in parsed[0]['content'] if p['type'] == 'image')
            assert image.getpixel((0, 0)) == (255, 0, 0)
            assert '<|image|>' in result.rendered_prompt
        return result

    def generate(request, **kwargs):
        # Stand in for decoding only: an open thinking prefix consumes the
        # budget, while the closed prefix generates a short direct answer.
        thinking = request.rendered_prompt.endswith('<think>')
        if not thinking:
            assert request.rendered_prompt.endswith('<think></think>')

        async def results():
            yield SimpleNamespace(
                response='analysis ' * budget if thinking else answer,
                token_ids=[100] * budget if thinking else [101],
                input_token_len=10,
                generate_token_len=budget if thinking else 1,
                finish_reason='length' if thinking else 'stop',
                cached_tokens=0,
                logprobs=None,
                routed_experts=None,
                cache_block_ids=None,
            )
        return results()

    context.async_engine.preprocess = preprocess
    context.async_engine.generate = generate
    payload = dict(model='fake-model', input='Reply with one word: ok', max_output_tokens=budget,
                   temperature=0, top_p=0.95, text={'format': {'type': 'text'}},
                   prompt_cache_key='whitelist-cache-probe', parallel_tool_calls=False, stream=stream)
    if with_image:
        buffer = BytesIO()
        Image.new('RGB', (32, 32), 'red').save(buffer, format='PNG')
        url = 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()
        payload['input'] = [dict(role='user', content=[
            dict(type='input_text', text='What color is this image?'), dict(type='input_image', image_url=url),
        ])]
    if effort is not None:
        payload['reasoning'] = {'effort': effort}
    app = FastAPI()
    app.include_router(create_responses_router(context))
    with TestClient(app) as client:
        response = client.post('/v1/responses', json=payload)
    assert response.status_code == 200
    expected_status = 'completed' if effort == 'none' else 'incomplete'
    expected_text = answer if effort == 'none' else ''
    if stream:
        events = sse_payloads([response.text])
        assert events[-1]['type'] == f'response.{expected_status}'
        assert not any(e['type'] in ('response.failed', 'error') for e in events)
        assert ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta') == expected_text
        result = events[-1]['response']
    else:
        result = response.json()
    assert result['status'] == expected_status
    assert result['output_text'] == expected_text
    assert result['output'][0]['status'] == expected_status
    if effort == 'none':
        assert [item['type'] for item in result['output']] == ['message']
    else:
        assert [item['type'] for item in result['output']] == ['reasoning']
        assert result['output'][0]['content'] == [{'type': 'reasoning_text', 'text': 'analysis ' * budget}]
    assert result['usage']['output_tokens'] == (1 if effort == 'none' else budget)
    assert result['usage']['output_tokens_details']['reasoning_tokens'] == (0 if effort == 'none' else budget)
    assert result.get('incomplete_details') == (None if effort == 'none' else {'reason': 'max_output_tokens'})
    assert len(context.session_manager.removed) == 1


@pytest.mark.parametrize('stream', [False, True])
def test_responses_reasoning_tool_round_trip(responses_endpoint, hf_media_template, sse_payloads, stream):
    _, context = responses_endpoint
    template = hf_media_template('glm5_3_flash')
    reasoning = 'Check the current status before deciding.'

    class Parser(BaseResponseParser):
        reasoning_parser_cls = ReasoningParserManager.get('default')
        tool_parser_cls = ToolParserManager.get('glm47')
        tokenizer = SimpleNamespace(get_vocab=lambda: {'<think>': 1, '</think>': 2})

    def generate(request, **kwargs):
        async def results():
            pieces = [
                ('Check the current status ', [100, 101, 102, 103], None),
                ('before deciding.', [104, 105, 106], None),
                ('</think><tool_call>get_status</tool_call>', [2, 107, 108, 109], 'stop'),
            ]
            tokens = 0
            for text, ids, finish in pieces:
                tokens += len(ids)
                yield SimpleNamespace(response=text, token_ids=ids, input_token_len=10,
                                      generate_token_len=tokens, finish_reason=finish, cached_tokens=0,
                                      logprobs=None, routed_experts=None, cache_block_ids=None)
        return results()

    context.response_parser_cls = Parser
    context.async_engine.generate = generate
    app = FastAPI()
    app.include_router(create_responses_router(context))
    user = {'role': 'user', 'content': 'Check the status.'}
    payload = dict(model='fake-model', input=[user], stream=stream, store=False,
                   reasoning={'effort': 'max', 'summary': 'auto'}, include=['reasoning.encrypted_content'],
                   tools=[{'type': 'function', 'name': 'get_status'}])
    with TestClient(app) as client:
        response = client.post('/v1/responses', json=payload)
        assert response.status_code == 200
        if stream:
            events = sse_payloads([response.text])
            assert events[-1]['type'] == 'response.completed'
            assert [event['sequence_number'] for event in events] == list(range(len(events)))
            assert ''.join(event['delta'] for event in events
                           if event['type'] == 'response.reasoning_text.delta') == reasoning
            done = next(event for event in events if event['type'] == 'response.reasoning_text.done')
            assert done['text'] == reasoning
            reasoning_end = next(i for i, event in enumerate(events)
                                 if event['type'] == 'response.output_item.done'
                                 and event['item']['type'] == 'reasoning')
            tool_start = next(i for i, event in enumerate(events)
                              if event['type'] == 'response.output_item.added'
                              and event['item']['type'] == 'function_call')
            assert reasoning_end < tool_start
            output = events[-1]['response']['output']
            assert events[reasoning_end]['item'] == output[0]
            assert done['item_id'] == output[0]['id'] and done['output_index'] == 0
        else:
            output = response.json()['output']

        assert [item['type'] for item in output] == ['reasoning', 'function_call']
        assert output[0]['content'] == [{'type': 'reasoning_text', 'text': reasoning}]
        assert output[0]['status'] == 'completed'
        call = output[1]
        assert call['name'] == 'get_status'
        tool_result = {'type': 'function_call_output', 'call_id': call['call_id'], 'output': 'READY'}
        follow_up = client.post('/v1/responses', json={**payload, 'input': [user, *output, tool_result]})
        assert follow_up.status_code == 200

    messages = context.async_engine.prompt
    assert [message['role'] for message in messages] == ['user', 'assistant', 'tool']
    assert messages[1]['reasoning_content'] == reasoning
    assert messages[1]['tool_calls'][0]['id'] == messages[2]['tool_call_id'] == call['call_id']
    prompt = template.messages2prompt(messages, reasoning_effort='max')
    assert f'<think>{reasoning}</think>' in prompt
    assert prompt.index(reasoning) < prompt.index('get_status') < prompt.index('READY')
