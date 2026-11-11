# Tests for /v1/completions usage reporting of prefix-cache hits: the
# engine exposes ``cached_tokens`` on every result and the chat
# completions path already forwards it as
# ``usage.prompt_tokens_details.cached_tokens`` - completions must do the
# same in both non-streaming and streaming (include_usage) modes, and the
# details object is always reported (mirroring ``UsageInfo.build``), so a
# cache miss reads as ``cached_tokens: 0`` rather than ``null``. Batch
# prompts accumulate each prompt's cached tokens into one usage block.
import asyncio
import json
from collections import deque
from types import SimpleNamespace

from fastapi import APIRouter

from lmdeploy.serve.openai.endpoints.completions import register
from lmdeploy.serve.openai.protocol import CompletionRequest


class _ResponseParser:
    """No reasoning parser: the reasoning usage path stays out of the way."""

    reasoning_parser_cls = None
    tokenizer = None


class _Session:

    async def async_abort(self):
        pass


class _SessionManager:

    def has(self, session_id):
        return False

    def remove(self, session):
        pass


class _AsyncEngine:
    """Serves one finished generation per prompt with a chosen cache hit."""

    model_name = 'fake-model'

    def __init__(self, prompt_input_ids, cached_tokens):
        self._prompt_input_ids = deque(prompt_input_ids)
        self._cached_tokens = deque(cached_tokens)

    async def preprocess(self, prompt, session, **kwargs):
        input_ids = self._prompt_input_ids.popleft()
        return SimpleNamespace(inputs={'input_ids': input_ids}, input_token_len=len(input_ids))

    def generate(self, request, **kwargs):
        input_token_len = len(request.inputs['input_ids'])
        token_ids = [7, 8]
        cached_tokens = self._cached_tokens.popleft()

        async def _generator():
            yield SimpleNamespace(token_ids=token_ids,
                                  response='x' * len(token_ids),
                                  finish_reason='length',
                                  logprobs=None,
                                  cache_block_ids=[],
                                  input_token_len=input_token_len,
                                  generate_token_len=len(token_ids),
                                  cached_tokens=cached_tokens)

        return _generator()


class _ServerContext:

    def __init__(self, engine):
        self.async_engine = engine
        self.engine_config = SimpleNamespace(logprobs_mode=None)
        self.session_manager = _SessionManager()
        self.default_gen_config = {}
        self.response_parser_cls = _ResponseParser

    def create_session(self, session_id=None):
        return _Session()


class _RawRequest:

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


async def _collect_prompt_tokens_details(prompt_input_ids, cached_tokens, stream):
    """Run the completions endpoint; return ``prompt_tokens_details``."""
    engine = _AsyncEngine(prompt_input_ids, cached_tokens)
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
        return response['usage'].get('prompt_tokens_details')
    usage = None
    async for chunk in response.body_iterator:
        if chunk.startswith('data: [DONE]'):
            continue
        payload = json.loads(chunk.removeprefix('data: '))
        if payload.get('usage'):
            usage = payload['usage']
    assert usage is not None, 'include_usage stream must end with a usage chunk'
    return usage.get('prompt_tokens_details')


def test_cache_hit_reported_in_usage():
    details = asyncio.run(_collect_prompt_tokens_details([[1, 2, 3]], [256], False))
    assert details == {'cached_tokens': 256}


def test_cache_hit_reported_in_usage_stream():
    details = asyncio.run(_collect_prompt_tokens_details([[1, 2, 3]], [256], True))
    assert details == {'cached_tokens': 256}


def test_cache_miss_reports_zero_rather_than_null():
    # mirrors UsageInfo.build: the details object is always present, so
    # clients can distinguish "no cache support" (null, older builds)
    # from "cache enabled, nothing hit" (0)
    for stream in [False, True]:
        details = asyncio.run(_collect_prompt_tokens_details([[1, 2, 3]], [0], stream))
        assert details == {'cached_tokens': 0}


def test_batch_prompts_accumulate_cached_tokens():
    for stream in [False, True]:
        details = asyncio.run(_collect_prompt_tokens_details([[1, 2], [3, 4], [5, 6]], [128, 64, 0],
                                                             stream))
        assert details == {'cached_tokens': 192}


async def _collect_stream_choice_indexes(prompts):
    """Return the ``choices[*].index`` of every streamed chunk, in order."""
    engine = _AsyncEngine([[1]] * len(prompts), [0] * len(prompts))
    context = _ServerContext(engine)
    router = APIRouter()
    register(router, context)
    endpoint = router.routes[0].endpoint
    request = CompletionRequest(model='fake-model',
                            prompt=prompts,
                            max_tokens=8,
                            stream=True,
                            stream_options={'include_usage': True})
    response = await endpoint(request, _RawRequest())
    indexes = []
    async for chunk in response.body_iterator:
        if chunk.startswith('data: [DONE]'):
            continue
        payload = json.loads(chunk.removeprefix('data: '))
        indexes.extend(choice['index'] for choice in payload.get('choices', []))
    return indexes


def test_stream_chunks_carry_per_prompt_choice_index():
    # streamed chunks must keep the prompt index so multi-prompt streams
    # can be reassembled per prompt, like the non-streaming choices array
    assert asyncio.run(_collect_stream_choice_indexes(['a', 'b'])) == [0, 1]
