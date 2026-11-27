# Copyright (c) OpenMMLab. All rights reserved.
"""Replay the Sage2.1 O12 generation through the OpenAI chat endpoint on
CPU."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tokenizers import AddedToken, decoders, models, pre_tokenizers
from tokenizers import Tokenizer as FastTokenizer
from transformers import PreTrainedTokenizerFast

from lmdeploy.messages import EngineOutput, ResponseType
from lmdeploy.serve.core.async_engine import AsyncEngine, PreprocessedRequest
from lmdeploy.serve.core.chat_runner import ChatRunner, ChatRunnerOptions
from lmdeploy.serve.openai.chat_completions import register
from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import BaseResponseParser
from lmdeploy.tokenizer import DetokenizeState, HuggingFaceTokenizer, Tokenizer

# One untrimmed QA generation. Both copies of STOP_HERE and the trailing text are present.
OUTPUT_IDS = [785, 1196, 6801, 752, 311, 2550, 6896, 279, 1555, 25, 8989, 26779, 29, 45268, 80891,
              9782, 9342, 1, 323, 4302, 770, 13, 154842, 522, 26779, 29, 45268, 80891, 9782, 9342]


def _replay_tokenizer():
    # Retain the original ID -> ByteLevel piece mapping. Only STOP_HERE needs ordinary input encoding here.
    vocab = {'[UNK]': 0, '"': 1, '.': 13, ':': 25, '>': 29, 'Ġthe': 279, 'Ġto': 311, 'Ġand': 323, '</': 522,
             'Ġme': 752, 'Ġelse': 770, 'The': 785, 'Ġuser': 1196, 'Ġline': 1555, 'Ġoutput': 2550,
             'Ġnothing': 4302, 'Ġwants': 6801, 'Ġexactly': 6896, 'Ġ"</': 8989, '-text': 9342, 'Ġtail': 9782,
             'think': 26779, 'ĠSTOP': 45268, 'STOP': 50336, '_HERE': 80891}
    vocab.update({'Ċ': 198, 'X': 55, 'lookup': 20973, 'Berlin': 93615})
    vocab.update({'answer': 9214, 'ĠãĤ': 17562, 'Ħ': 226, 'tail': 14576})
    vocab.update({'\u00ef\u00bf\u00bd': 60001})  # A literal U+FFFD encoded in UTF-8.
    vocab.update({'\u00ed': 169, '\u00b4': 112})
    tags = {'<think>': 154841, '</think>': 154842, '<tool_call>': 154843, '</tool_call>': 154844,
            '<arg_key>': 154847, '</arg_key>': 154848, '<arg_value>': 154849, '</arg_value>': 154850}
    backend = FastTokenizer(models.WordLevel({**vocab, **tags}, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split('_', behavior='merged_with_next'),
        pre_tokenizers.Split('\n', behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    backend.decoder = decoders.ByteLevel()
    backend.add_tokens([AddedToken(tag, normalized=False) for tag in tags])
    hf = PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)
    tokenizer = object.__new__(Tokenizer)
    tokenizer.model = object.__new__(HuggingFaceTokenizer)
    tokenizer.model.model = hf
    tokenizer.logger = logging.getLogger('lmdeploy')
    tokenizer.model._indexes_tokens_deque = [('Berlin', [93615])]
    return tokenizer


def _replay_app(context, outputs=None):
    tokenizer = _replay_tokenizer()

    class Parser(BaseResponseParser):
        pass

    Parser.set_parsers('default', 'glm47', tokenizer.model.model)
    context.response_parser_cls = Parser
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.tokenizer = tokenizer
    engine.num_spec_token = 5
    engine.session_mgr = context.session_manager
    engine.request_logger = SimpleNamespace(log_response=lambda *args: None)
    context.async_engine.tokenizer = tokenizer
    cancel = AsyncMock()

    @asynccontextmanager
    async def request_handle():
        yield SimpleNamespace(async_cancel=cancel)

    async def preprocess(messages, session, gen_config, **kwargs):
        gen_config.convert_stop_bad_words_to_ids(tokenizer)
        session.epoch = None
        session.request_handle = request_handle
        return PreprocessedRequest(session=session, inputs={'input_ids': [785]}, input_token_len=1,
                                   gen_config=gen_config, adapter_name=None)

    @asynccontextmanager
    async def replay_backend(*args, **kwargs):
        async def batches():
            if outputs is not None:
                for output in outputs:
                    yield output
                return
            # The answer's STOP and _HERE straddle the last two batches.
            for batch in (OUTPUT_IDS[:22], OUTPUT_IDS[22:27], OUTPUT_IDS[27:]):
                yield EngineOutput(ResponseType.SUCCESS, batch)
            yield EngineOutput(ResponseType.FINISH, [])
        yield batches()

    engine.safe_run = replay_backend
    context.async_engine.preprocess = preprocess
    context.async_engine.generate = engine.generate
    context.async_engine.model_name = 'glm53-flash'
    app = FastAPI()
    register(app, context)
    return app, cancel


@pytest.mark.parametrize('options', [{}, {'return_token_ids': True}, {'stream': True, 'return_token_ids': True}],
                         ids=['qa-request', 'output-ids', 'stream-output-ids'])
@pytest.mark.parametrize('traced', [False, True])
def test_multi_token_stop_in_content_only(chat_endpoint, options, traced, monkeypatch, trace_sdk):
    _, context = chat_endpoint
    app, cancel = _replay_app(context)
    from lmdeploy.serve.observability import request_trace
    monkeypatch.setattr(request_trace, 'ENABLED', traced)
    if traced:
        context.async_engine.preprocess = request_trace.trace_preprocess(context.async_engine.preprocess)
        app = request_trace.RequestTraceMiddleware(app)
    prompt = 'Output exactly this line and nothing else: </think> STOP_HERE tail-text'
    request = dict(model='glm53-flash', messages=[dict(role='user', content=prompt)], temperature=0, max_tokens=4096,
                   reasoning_effort='high', chat_template_kwargs={'reasoning_effort': 'high'}, stop=['STOP_HERE'],
                   **options)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 200, response.text
    if options.get('stream'):
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        choices = [choice for chunk in chunks for choice in chunk['choices']]
        body = ''.join(choice['delta'].get('content') or '' for choice in choices)
        reasoning = ''.join(choice['delta'].get('reasoning_content') or '' for choice in choices)
        ids = [i for choice in choices for i in choice.get('output_ids') or []]
        finish = choices[-1]['finish_reason']
    else:
        choice = response.json()['choices'][0]
        body, reasoning = choice['message']['content'], choice['message']['reasoning_content']
        ids, finish = choice.get('output_ids'), choice['finish_reason']
    assert reasoning == ('The user wants me to output exactly the line: '
                         '"</think> STOP_HERE tail-text" and nothing else.')
    assert body == '</think> '
    if options.get('return_token_ids'):
        assert ids == OUTPUT_IDS[:-2]
    assert finish == 'stop'
    cancel.assert_awaited_once()
    if traced:
        span = next(s for s in trace_sdk.get_finished_spans() if s.name == 'inference')
        assert span.attributes['lmdeploy.token_counts_complete']
        assert span.attributes['lmdeploy.finish_reason'] == 'stop'


def test_nonstream_stop_miss_preserves_custom_parse_complete(chat_endpoint):
    _, context = chat_endpoint
    app, cancel = _replay_app(context)

    class CustomParser(context.response_parser_cls):
        def parse_complete(self, text, token_ids=None, **kwargs):
            content, tools, reasoning = super().parse_complete(text, token_ids, **kwargs)
            return f'custom: {content}', tools, reasoning

    context.response_parser_cls = CustomParser
    request = dict(model='glm53-flash', messages=[dict(role='user', content='Replay the recorded answer.')],
                   stream=False, stop=['STOP_HERE_HERE'], return_token_ids=True)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 200, response.text
    choice = response.json()['choices'][0]
    assert choice['message']['content'] == 'custom: </think> STOP_HERE tail-text'
    assert choice['output_ids'] == OUTPUT_IDS
    assert choice['finish_reason'] == 'length'
    cancel.assert_not_awaited()


def _completion_fields(response, stream):
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith('data: ') and line != 'data: [DONE]']
        choices = [choice for chunk in chunks for choice in chunk['choices']]
        usage = chunks[-1]['usage']
    else:
        choices = response.json()['choices']
        usage = response.json()['usage']
    messages = [choice['delta' if stream else 'message'] for choice in choices]
    body = ''.join(message.get('content') or '' for message in messages)
    names = [call['function']['name'] for message in messages for call in message.get('tool_calls') or []
             if call['function'].get('name')]
    ids = [token for choice in choices for token in choice.get('output_ids') or []]
    probs = [entry for choice in choices for entry in choice.get('output_token_logprobs') or []]
    return body, names, ids, probs, usage, choices[-1]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('include', [False, True])
@pytest.mark.parametrize('split', [False, True], ids=['one-batch', 'split-batches'])
@pytest.mark.parametrize('suffix, minimum, match', [
    ([198, 55, 9782], 0, 'content'),
    ([55, 9782], 6, 'length'),
    ([93615], 0, 'content'),
    ([93615], 6, 'native'),
    ([], 0, 'content'),
    ([154843, 20973, 154844, 55], 0, 'two_tools'),
])
def test_stop_after_tool_preserves_token_boundary(chat_endpoint, stream, include, split, suffix, minimum, match):
    _, context = chat_endpoint
    context.engine_config.logprobs_mode = 'raw_logprobs'
    # </think><tool_call>lookup</tool_call>, followed by two separate newline tokens.
    start = [154842, 154843, 20973, 154844, 198, 198]
    groups = [start[:5], start[5:], suffix] if split else [start + suffix]
    experts = [[[i]] for i in range(len(start + suffix))]
    context.engine_config.enable_return_routed_experts = True
    outputs = [
        EngineOutput(ResponseType.FINISH if i == len(groups) - 1 else ResponseType.SUCCESS, ids,
                     routed_experts=experts if i == len(groups) - 1 else None,
                     logprobs=[{token: -0.5} for token in ids])
        for i, ids in enumerate(groups)
    ]
    app, _ = _replay_app(context, outputs)
    request = dict(model='glm53-flash', messages=[dict(role='user', content='Look it up.')],
                   tools=[dict(type='function', function=dict(name='lookup', parameters={'type': 'object'}))],
                   stop=['\n\n', 'Berlin'], min_new_tokens=minimum, max_tokens=32,
                   include_stop_str_in_output=include, return_token_ids=True, return_logprob=True,
                   return_routed_experts=True, stream=stream,
                   stream_options={'include_usage': True} if stream else None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 200, response.text
    body, names, ids, probs, usage, last = _completion_fields(response, stream)
    expected_ids = start
    if match == 'content':
        expected_body = '\n\n' if include else ''
    elif match == 'native':
        expected_body = '\n\nBerlin' if include else '\n\n'
        expected_ids = start + suffix if include else start
    else:
        expected_body = 'X' if match == 'two_tools' else '\n\nX tail'
        expected_ids = start + suffix
    assert body == expected_body
    assert names == ['lookup'] * (2 if match == 'two_tools' else 1)
    assert ids == expected_ids
    assert probs == [[-0.5, token] for token in expected_ids]
    assert last['routed_experts'] == experts[:len(expected_ids)]
    assert usage['completion_tokens'] == len(expected_ids)
    assert last['finish_reason'] == ('stop' if match in ('native', 'content') else 'length')


@pytest.mark.parametrize('stream', [True, False])
@pytest.mark.parametrize('split', [True, False], ids=['one-id-at-a-time', 'one-batch'])
@pytest.mark.parametrize('minimum', [0, 5])
@pytest.mark.parametrize('include', [False, True])
def test_stop_before_partial_utf8(chat_endpoint, stream, split, minimum, include):
    _, context = chat_endpoint
    context.engine_config.logprobs_mode = 'raw_logprobs'
    context.engine_config.enable_return_routed_experts = True
    # 17562 contains the space ending STOP_HERE and the first two bytes of U+3084.
    ids = [154842, 9214, 45268, 80891, 17562, 226, 14576]
    groups = [[i] for i in ids] if split else [ids]
    experts = [[[i]] for i in range(len(ids))]
    outputs = [EngineOutput(ResponseType.FINISH if i == len(groups) - 1 else ResponseType.SUCCESS, group,
                            logprobs=[{token: -0.5} for token in group],
                            routed_experts=experts if i == len(groups) - 1 else None)
               for i, group in enumerate(groups)]
    app, _ = _replay_app(context, outputs)
    request = dict(model='glm53-flash', messages=[dict(role='user', content='Replay.')], stop=['STOP_HERE '],
                   min_new_tokens=minimum, max_tokens=16, include_stop_str_in_output=include,
                   return_token_ids=True, return_logprob=True, return_routed_experts=True, stream=stream,
                   stream_options={'include_usage': True} if stream else None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json=request)
    assert response.status_code == 200, response.text
    body, _, tokens, probs, usage, last = _completion_fields(response, stream)
    expected_ids = ids if minimum else ids[:5]
    expected_body = 'answer STOP_HERE \u3084tail' if minimum else ('answer STOP_HERE ' if include else 'answer ')
    assert body == expected_body
    assert tokens == expected_ids
    assert probs == [[-0.5, token] for token in expected_ids]
    assert usage['completion_tokens'] == len(expected_ids)
    assert last['routed_experts'] == experts[:len(expected_ids)]
    assert last['finish_reason'] == ('length' if minimum else 'stop')


def test_stream_delivers_newline_ids_without_unrelated_stop_lookahead(chat_endpoint):
    _, context = chat_endpoint
    consumed = []

    def outputs():
        for token in [154842, 154843, 20973, 154844, 198, 198, 198, 55]:
            consumed.append(token)
            yield EngineOutput(ResponseType.SUCCESS, [token])
        yield EngineOutput(ResponseType.FINISH, [])

    _replay_app(context, outputs())
    request = ChatCompletionRequest(model='glm53-flash', messages=[dict(role='user', content='Look it up.')],
                                    tools=[dict(type='function', function=dict(name='lookup',
                                                                               parameters={'type': 'object'}))],
                                    stop=['STOP_HERE\n'], stream=True, return_token_ids=True)

    async def run():
        runner = await ChatRunner.prepare(context, request, ChatRunnerOptions(match_glm_content_stop=True))
        received = []
        async for chunk in runner.stream():
            received.extend(chunk.token_ids)
            if chunk.token_ids == [198]:
                assert received == consumed
        assert received == consumed

    asyncio.run(run())


def test_partial_utf8_preserves_literal_replacement_and_whole_token_mode():
    tokenizer = _replay_tokenizer()
    ids = [9214, 17562, 226, 60001, 55]
    for emitted, expected in [
        (0, ['answer', ' ', '\u3084', '\ufffd', 'X']),
        (None, ['answer', '', ' \u3084', '', '\ufffdX']),
    ]:
        state = DetokenizeState(emitted_prefix_chars=emitted)
        deltas = []
        for end in range(1, len(ids) + 1):
            text, state = tokenizer.detokenize_incrementally(ids[:end], state)
            deltas.append(text)
        assert deltas == expected
        assert ''.join(deltas) == tokenizer.decode(ids)


def test_unfinished_utf8_surrogate_does_not_emit_early_replacements():
    tokenizer = _replay_tokenizer()
    ids = [169, 112, 55]  # ED B4, followed by ASCII X.
    state = DetokenizeState(emitted_prefix_chars=0)
    deltas = []
    for end in range(1, len(ids) + 1):
        text, state = tokenizer.detokenize_incrementally(ids[:end], state)
        deltas.append(text)
    assert deltas == ['', '', '\ufffd\ufffdX']
