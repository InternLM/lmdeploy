# Tests for PR #5091 review fix: reasoning counters seeded from prompt state.
# Covers: open reasoning prefix, closed reasoning prefix, plain text, mixed
# batch - in both non-streaming and streaming (include_usage) modes.
import asyncio
import json
from collections import deque
from types import SimpleNamespace

import pytest
from fastapi import APIRouter

from lmdeploy.serve.openai.endpoints.completions import register
from lmdeploy.serve.openai.protocol import CompletionRequest

OPEN_TAG_ID = 100
CLOSE_TAG_ID = 101


class _ReasoningParser:

    @staticmethod
    def get_reasoning_open_tag():
        return '<think>'

    @staticmethod
    def get_reasoning_close_tag():
        return '</think>'


class _Tokenizer:

    def get_vocab(self):
        return {'<think>': OPEN_TAG_ID, '</think>': CLOSE_TAG_ID}


class _ResponseParser:
    reasoning_parser_cls = _ReasoningParser
    tokenizer = _Tokenizer()


class _Session:

    async def async_abort(self):
        pass


class _SessionManager:

    def has(self, session_id):
        return False

    def remove(self, session):
        pass


class _AsyncEngine:
    """Serves prepared prompt ids and one chunk of generated ids per prompt."""

    model_name = 'fake-model'

    def __init__(self, prompt_input_ids, generated_ids):
        self._prompt_input_ids = deque(prompt_input_ids)
        self._generated_ids = deque(generated_ids)

    async def preprocess(self, prompt, session, **kwargs):
        input_ids = self._prompt_input_ids.popleft()
        return SimpleNamespace(inputs={'input_ids': input_ids}, input_token_len=len(input_ids))

    def generate(self, request, **kwargs):
        token_ids = self._generated_ids.popleft()
        input_token_len = len(request.inputs['input_ids'])

        async def _generator():
            yield SimpleNamespace(token_ids=token_ids,
                                  response='x' * len(token_ids),
                                  finish_reason='length',
                                  logprobs=None,
                                  cache_block_ids=[],
                                  input_token_len=input_token_len,
                                  generate_token_len=len(token_ids))

        return _generator()


class _ServerContext:

    def __init__(self, engine):
        self.async_engine = engine
        self.engine_config = SimpleNamespace(logprobs_mode=None)
        self.session_manager = _SessionManager()
        self.default_gen_config = {}
        self.response_parser_cls = _ResponseParser()

    def create_session(self, session_id=None):
        return _Session()


class _RawRequest:

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


async def _collect_reasoning_tokens(prompt_input_ids, generated_ids, stream):
    engine = _AsyncEngine(prompt_input_ids, generated_ids)
    context = _ServerContext(engine)
    router = APIRouter()
    register(router, context)
    endpoint = router.routes[0].endpoint
    request = CompletionRequest(
        model='fake-model',
        prompt=['p'] * len(prompt_input_ids),
        max_tokens=8,
        stream=stream,
        stream_options={'include_usage': True} if stream else None)
    response = await endpoint(request, _RawRequest())
    if not stream:
        return response['usage']['completion_tokens_details']['reasoning_tokens']
    usage = None
    async for chunk in response.body_iterator:
        if chunk.startswith('data: [DONE]'):
            continue
        payload = json.loads(chunk.removeprefix('data: '))
        if payload.get('usage'):
            usage = payload['usage']
    assert usage is not None
    return usage['completion_tokens_details']['reasoning_tokens']


# name, prompt_input_ids (per prompt), generated_ids (per prompt), expected
SINGLE_PROMPT_CASES = [
    # prompt ends with an open tag: generated thought tokens are counted
    ('open_reasoning_prefix', [[1, 2, OPEN_TAG_ID]], [[7, 8, CLOSE_TAG_ID, 9]], 2),
    # prompt's reasoning block already closed: nothing counted
    ('closed_reasoning_prefix', [[1, OPEN_TAG_ID, 3, CLOSE_TAG_ID, 5]], [[7, 8]], 0),
    # plain prompt: counting still starts at an explicit open tag
    ('plain_text_prompt', [[1, 2, 3]], [[7, OPEN_TAG_ID, 9, CLOSE_TAG_ID, 11]], 1),
]


@pytest.mark.parametrize('name,prompt_input_ids,generated_ids,expected', SINGLE_PROMPT_CASES)
@pytest.mark.parametrize('stream', [False, True])
def test_reasoning_tokens_are_seeded_from_prompt_state(name, prompt_input_ids, generated_ids,
                                                       expected, stream):
    reasoning = asyncio.run(
        _collect_reasoning_tokens(prompt_input_ids, generated_ids, stream))
    assert reasoning == expected


def test_mixed_prompt_batch_sums_prompt_seeded_counters():
    prompt_input_ids = [
        [1, OPEN_TAG_ID],  # open prefix -> counting immediately
        [1, 2],  # plain prompt
        [1, OPEN_TAG_ID, 3, CLOSE_TAG_ID],  # closed prefix
    ]
    generated_ids = [
        [5, CLOSE_TAG_ID, 7],  # 5 inside (1), 7 after close
        [8, OPEN_TAG_ID, 9, CLOSE_TAG_ID],  # 9 inside (1)
        [3],  # outside (0)
    ]
    for stream in [False, True]:
        reasoning = asyncio.run(
            _collect_reasoning_tokens(prompt_input_ids, generated_ids, stream))
        assert reasoning == 2
