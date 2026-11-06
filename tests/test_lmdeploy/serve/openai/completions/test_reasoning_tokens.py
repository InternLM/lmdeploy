# Tests for PR #5091: reasoning token reporting in /v1/completions.
# Covers: counters seeded from the prompt state (open reasoning prefix,
# closed reasoning prefix, plain text, mixed batch; tags inside earlier
# turns are not inherited across special-token turn boundaries), tag
# resolution through convert_tokens_to_ids without copying the vocabulary
# (get_vocab is poisoned in the fake tokenizer so any regression fails
# loudly), the missing-tag behavior for both None and unk-token fallbacks,
# and skipping the accounting for streams that do not request usage - all
# in both non-streaming and streaming (include_usage) modes.
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
UNK_TOKEN_ID = 7
IM_START_ID = 5
IM_END_ID = 6
TEXT_ID = 9


class _ReasoningParser:

    @staticmethod
    def get_reasoning_open_tag():
        return '<think>'

    @staticmethod
    def get_reasoning_close_tag():
        return '</think>'


class _Tokenizer:
    """Fake HF-style tokenizer used for the reasoning tag lookups.

    ``get_vocab`` is poisoned on purpose: the endpoint must resolve the
    two tags with ``convert_tokens_to_ids`` instead of copying the whole
    vocabulary on every completion request (>100 ms per call on
    large-vocab tokenizers). Like Qwen-style tokenizers, the reasoning
    tags themselves are special tokens; the turn boundary tokens
    (im_start/im_end) model a chat template's message delimiters.

    Args:
        vocab: token-to-id mapping known to the tokenizer; tags missing
            from it exercise the missing-tag behavior.
        unk_token_id: id that unknown tokens fall back to, mirroring
            tokenizers configured with an ``unk`` token.
    """

    # tags included on purpose: the endpoint must exclude them from the
    # turn-boundary set or the tags themselves would act as boundaries
    all_special_ids = (OPEN_TAG_ID, CLOSE_TAG_ID, IM_START_ID, IM_END_ID)

    def __init__(self, vocab=None, unk_token_id=None):
        if vocab is None:
            vocab = {'<think>': OPEN_TAG_ID, '</think>': CLOSE_TAG_ID}
        self._vocab = dict(vocab)
        self.unk_token_id = unk_token_id
        self.lookup_calls = 0

    def convert_tokens_to_ids(self, tokens):
        self.lookup_calls += 1
        return [self._vocab.get(token, self.unk_token_id) for token in tokens]

    def get_vocab(self):
        raise AssertionError('the completions endpoint must not copy the tokenizer vocabulary')


class _ResponseParser:

    reasoning_parser_cls = _ReasoningParser
    tokenizer = _Tokenizer()

    def __init__(self, tokenizer=None):
        if tokenizer is not None:
            self.tokenizer = tokenizer


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

    def __init__(self, engine, tokenizer=None):
        self.async_engine = engine
        self.engine_config = SimpleNamespace(logprobs_mode=None)
        self.session_manager = _SessionManager()
        self.default_gen_config = {}
        self.response_parser_cls = _ResponseParser(tokenizer)

    def create_session(self, session_id=None):
        return _Session()


class _RawRequest:

    async def json(self):
        return {}

    async def is_disconnected(self):
        return False


async def _collect_completion_details(prompt_input_ids, generated_ids, stream,
                                      tokenizer=None):
    """Run the completions endpoint; return ``completion_tokens_details``."""
    engine = _AsyncEngine(prompt_input_ids, generated_ids)
    context = _ServerContext(engine, tokenizer)
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
        return response['usage'].get('completion_tokens_details')
    usage = None
    async for chunk in response.body_iterator:
        if chunk.startswith('data: [DONE]'):
            continue
        payload = json.loads(chunk.removeprefix('data: '))
        if payload.get('usage'):
            usage = payload['usage']
    assert usage is not None
    return usage.get('completion_tokens_details')


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
    # the fake tokenizer's get_vocab raises, so these double as the
    # regression test that tag resolution never copies the vocabulary
    details = asyncio.run(
        _collect_completion_details(prompt_input_ids, generated_ids, stream))
    assert details['reasoning_tokens'] == expected


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
        details = asyncio.run(
            _collect_completion_details(prompt_input_ids, generated_ids, stream))
        assert details['reasoning_tokens'] == 2


@pytest.mark.parametrize('unk_token_id', [None, UNK_TOKEN_ID],
                         ids=['none_fallback', 'unk_fallback'])
@pytest.mark.parametrize('missing_tag', ['<think>', '</think>'])
@pytest.mark.parametrize('stream', [False, True])
def test_unresolvable_tags_leave_reasoning_usage_unreported(missing_tag, unk_token_id, stream):
    # a tag that is not a single vocabulary token resolves to None on
    # unk-less tokenizers and to the unk fallback id otherwise; both must
    # keep the previous unreported behavior instead of counting tokens
    # against the fallback id
    vocab = {'<think>': OPEN_TAG_ID, '</think>': CLOSE_TAG_ID}
    del vocab[missing_tag]
    details = asyncio.run(
        _collect_completion_details([[1, OPEN_TAG_ID]], [[9, CLOSE_TAG_ID, UNK_TOKEN_ID]], stream,
                                    tokenizer=_Tokenizer(vocab, unk_token_id=unk_token_id)))
    assert details is None


@pytest.mark.parametrize('stream', [False, True])
def test_history_reasoning_tags_do_not_seed_counting(stream):
    # a <think> mentioned inside an earlier user turn - with the
    # assistant turn already continuing a normal answer behind a
    # special-token boundary - must not count the answer as reasoning
    # (3 ordinary tokens were previously reported as reasoning_tokens=3)
    prompt_input_ids = [[TEXT_ID, TEXT_ID, OPEN_TAG_ID, TEXT_ID, IM_END_ID, IM_START_ID, TEXT_ID,
                         TEXT_ID]]
    generated_ids = [[TEXT_ID, TEXT_ID, TEXT_ID]]
    details = asyncio.run(
        _collect_completion_details(prompt_input_ids, generated_ids, stream))
    assert details['reasoning_tokens'] == 0


@pytest.mark.parametrize('stream', [False, True])
def test_tags_after_last_boundary_seed_counting(stream):
    # an assistant turn already inside a reasoning block behind the turn
    # boundary keeps counting: boundaries exclude earlier turns, not the
    # current one
    prompt_input_ids = [[IM_START_ID, OPEN_TAG_ID, TEXT_ID]]
    generated_ids = [[TEXT_ID, TEXT_ID, CLOSE_TAG_ID, TEXT_ID]]
    details = asyncio.run(
        _collect_completion_details(prompt_input_ids, generated_ids, stream))
    assert details['reasoning_tokens'] == 2


def test_streaming_without_usage_skips_reasoning_accounting():
    # a stream that does not request usage must not run the reasoning
    # accounting at all (observable through the tokenizer lookups)
    tokenizer = _Tokenizer()
    engine = _AsyncEngine([[IM_START_ID, OPEN_TAG_ID]], [[TEXT_ID, CLOSE_TAG_ID]])
    context = _ServerContext(engine, tokenizer)
    router = APIRouter()
    register(router, context)
    endpoint = router.routes[0].endpoint
    request = CompletionRequest(model='fake-model',
                                prompt=['p'],
                                max_tokens=8,
                                stream=True,
                                stream_options=None)
    response = asyncio.run(endpoint(request, _RawRequest()))
    usage_chunks = 0
    async def _drain():
        nonlocal usage_chunks
        async for chunk in response.body_iterator:
            if chunk.startswith('data: [DONE]'):
                continue
            payload = json.loads(chunk.removeprefix('data: '))
            if payload.get('usage'):
                usage_chunks += 1
    asyncio.run(_drain())
    assert usage_chunks == 0
    assert tokenizer.lookup_calls == 0
