# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from lmdeploy.messages import EngineOutput, GenerationConfig, ResponseType
from lmdeploy.serve.core.async_engine import AsyncEngine, PreprocessedRequest
from lmdeploy.serve.core.chat_runner import ChatRunner
from lmdeploy.serve.core.content_stop import ChatContentStop
from lmdeploy.serve.core.exceptions import RequestError
from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage


class Tokenizer:
    def indexes_containing_token(self, word):
        return {'6': [6], 'END': [8]}[word]

    def decode(self, ids, **kwargs):
        return ''.join({1: 'answer ', 6: '6', 8: 'END', 9: '<eos>'}[i] for i in ids)

    def detokenize_incrementally(self, ids, state, **kwargs):
        text = self.decode(ids[state.ids_offset:])
        state.ids_offset = len(ids)
        return text, state


class Parser:
    tool_parser = None
    reasoning_tokens = 0

    def parse_complete(self, text, token_ids):
        return text, None, None

    def stream_chunk(self, text, token_ids, **kwargs):
        return [(DeltaMessage(content=text), False)]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('include_stop', [False, True])
@pytest.mark.parametrize('terminal, expected', [(6, '6'), (8, 'END'), (9, None), (1, None)])
@pytest.mark.parametrize('mixed', [False, True])
def test_engine_and_runner_preserve_actual_stop_sequence(stream, include_stop, terminal, expected, mixed):
    @asynccontextmanager
    async def request_handle():
        yield object()

    @asynccontextmanager
    async def safe_run(*args, **kwargs):
        async def generate():
            yield EngineOutput(ResponseType.SUCCESS, [1])
            yield EngineOutput(ResponseType.FINISH, [terminal])
        yield generate()

    session = SimpleNamespace(session_id=0, epoch=None, request_handle=request_handle)
    session_manager = SimpleNamespace(remove=lambda session: None)
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = session_manager
    engine.tokenizer = Tokenizer()
    engine.num_spec_token = 0
    engine.safe_run = safe_run
    engine.request_logger = SimpleNamespace(log_response=lambda *args: None)
    config = GenerationConfig(max_new_tokens=2, stop_words=['6', 'END'],
                              stop_token_ids=[6, 8, 9], include_stop_str_in_output=include_stop)
    preprocessed = PreprocessedRequest(session=session, inputs={'input_ids': [0]}, input_token_len=1,
                                       gen_config=config, adapter_name=None)
    # With include_stop enabled, the native END also completes the pending text 'answer END'.
    stop = ChatContentStop(['answer END'], include_stop) if mixed else None
    runner = ChatRunner(
        request=ChatCompletionRequest(model='fake', messages=[]), response_parser=Parser(), gen_config=config,
        session=session, result_generator=engine.generate(preprocessed, content_stop=stop),
        session_manager=session_manager, content_stop=stop)

    async def run():
        if stream:
            chunks = [chunk async for chunk in runner.stream()]
            assert all(chunk.stop_sequence is None for chunk in chunks[:-1])
            return chunks[-1], ''.join(chunk.delta_message.content or '' for chunk in chunks)
        result = await runner.collect()
        return result, result.text

    result, text = asyncio.run(run())
    assert result.stop_sequence == expected
    assert result.finish_reason == ('length' if terminal == 1 else 'stop')
    if expected is not None:
        assert text == 'answer ' + (expected if include_stop else '')
    assert runner._closed


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('include_stop', [False, True])
@pytest.mark.parametrize('batches', [
    [[2, 3, 4, 5, 6, 2, 3, 7]],
    [[2], [3, 4], [5, 6, 2], [3, 7]],
    [[2, 3, 4, 5], [6, 2], [3, 7]],  # A clean batch ends in STOP, and the next batch begins with _HERE.
    [[2, 3, 4, 5, 6, 2, 3, 8]],  # The text stop finishes before the engine stop in the same MTP batch.
])
@pytest.mark.parametrize('min_new_tokens', [None, 6])
def test_stop_in_content_after_reasoning(stream, include_stop, batches, min_new_tokens):
    # O12: the model quotes the same stop word in reasoning and in the answer.
    # IDs 4 and 5 have the same spelling, but only 4 changes the channel.
    decoded_batches = []

    class TestTokenizer(Tokenizer):
        def detokenize_incrementally(self, ids, state, **kwargs):
            decoded_batches.append(ids[state.ids_offset:])
            text, state = super().detokenize_incrementally(ids, state, **kwargs)
            state.prev_tokens = []  # This test tokenizer never retains undecoded bytes.
            return text, state

        def decode(self, ids, **kwargs):
            pieces = {2: 'STOP', 3: '_HERE', 4: '</think>', 5: '</think>', 6: ' ', 7: ' tail-text'}
            return ''.join(pieces[i] for i in ids)

    class TestParser(Parser):
        reasoning = True

        def stream_chunk(self, text, ids, **kwargs):
            if 4 in ids:
                self.reasoning = False
                return []
            if self.reasoning:
                return [(DeltaMessage(reasoning_content=text), False)]
            return [(DeltaMessage(content=text), False)]

    async def cancel(session_id):
        cancelled.append(session_id)

    @asynccontextmanager
    async def request_handle():
        yield SimpleNamespace(async_cancel=cancel)

    @asynccontextmanager
    async def safe_run(*args, **kwargs):
        async def generate():
            for ids in batches:
                status = ResponseType.FINISH if ids[-1] == 8 else ResponseType.SUCCESS
                yield EngineOutput(status, ids, logprobs=[{i: -0.1} for i in ids])
            pytest.fail('the engine read past the completed stop string')
        yield generate()

    cancelled = []
    session = SimpleNamespace(session_id=1, epoch=None, request_handle=request_handle)
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SimpleNamespace(remove=lambda session: None)
    engine.tokenizer = TestTokenizer()
    engine.safe_run = safe_run
    engine.num_spec_token = 5
    engine.request_logger = SimpleNamespace(log_response=lambda *args: None)
    config = GenerationConfig(stop_words=['END'], stop_token_ids=[8], min_new_tokens=min_new_tokens,
                              include_stop_str_in_output=include_stop)
    def preview(ids):
        return None if 4 in ids else engine.tokenizer.decode(ids)

    stop = ChatContentStop(['STOP_HERE'], include_stop, preview)
    prep = PreprocessedRequest(session=session, inputs={'input_ids': [0]}, input_token_len=1,
                               gen_config=config, adapter_name=None)
    request = ChatCompletionRequest(model='test', messages=[], stop='STOP_HERE', return_token_ids=True)
    runner = ChatRunner(request, TestParser(), config, session, engine.generate(prep, content_stop=stop),
                        engine.session_mgr, content_stop=stop)

    async def run():
        if stream:
            chunks = [chunk async for chunk in runner.stream()]
            assert all(chunk.finish_reason is None for chunk in chunks[:-1])
            fields = [chunk.delta_message for chunk in chunks]
            return (''.join(x.content or '' for x in fields), ''.join(x.reasoning_content or '' for x in fields),
                    [i for c in chunks for i in c.token_ids], [p for c in chunks for p in c.logprobs or []], chunks[-1])
        result = await runner.collect()
        return result.text, result.reasoning_content, result.token_ids, result.logprobs, result

    content, reasoning, ids, logprobs, last = asyncio.run(run())
    assert content == ('</think> STOP_HERE' if include_stop else '</think> ')
    assert reasoning == 'STOP_HERE'
    assert ids == [2, 3, 4, 5, 6, 2, 3]
    assert logprobs == [{i: -0.1} for i in ids]
    assert last.generate_token_len == len(ids)
    assert (last.finish_reason, last.stop_sequence) == ('stop', 'STOP_HERE')
    assert cancelled == ([] if batches[-1][-1] == 8 else [1])
    if [6, 2] in batches:
        assert [6, 2] in decoded_batches
        assert decoded_batches[-1] == [3]  # No decode or metadata for the trailing ID 7.
    assert runner._closed


def test_content_stop_keeps_partial_and_matches_inside_token():
    stop = ChatContentStop(['STOP_HERE', 'HERE'], False)

    def feed(text, final=False):
        return ''.join(delta.content or '' for delta, _ in stop.filter_deltas([(DeltaMessage(content=text), False)],
                                                                               final))

    assert feed('print ST') == 'print '
    assert feed('OP_x') == 'STOP_x'
    assert feed(' prefix STOP') == ' prefix '
    assert feed('_HERE extra') == ''
    assert stop.matched == 'STOP_HERE'

    stop = ChatContentStop(['STOP_HERE'], False)
    assert feed('tail ST') == 'tail '
    assert feed('', final=True) == 'ST'


def test_min_new_tokens_keeps_prefix_without_replaying_an_earlier_match():
    stop = ChatContentStop(['AB', 'ABCD'], include_stop=False)
    # AB completed below the minimum. The next token can complete ABCD,
    # or disprove it and release AB as ordinary content.
    assert stop.filter_deltas([(DeltaMessage(content='AB'), False)], final=False, can_stop=False) == []
    result = stop.filter_deltas([(DeltaMessage(content='X'), False)], final=False)
    assert [delta.content for delta, _ in result] == ['ABX']
    assert stop.matched is None
    assert stop.filter_deltas([(DeltaMessage(content='AB'), False)], final=False, can_stop=False) == []
    assert stop.filter_deltas([(DeltaMessage(content='CD'), False)], final=True) == []
    assert stop.matched == 'ABCD'


@pytest.mark.parametrize('terminal', [ResponseType.CANCEL, ResponseType.FINISH, 'same_batch', 'error', 'missing'])
@pytest.mark.parametrize('use_ref', [False, True])
@pytest.mark.parametrize('start', [0, 2])
def test_content_stop_collects_only_accepted_routed_experts(terminal, use_ref, start, monkeypatch):
    from lmdeploy.serve.core.async_engine import metrics_processor

    updates = []
    monkeypatch.setattr(metrics_processor, 'queue_update', lambda item: updates.append(item[0]))
    experts = 'batch-key' if use_ref else list(range(start, 7))
    cancelled = []
    trimmed = []

    async def cancel(session_id):
        cancelled.append(session_id)

    async def trim(key, num_rows):
        trimmed.append((key, num_rows))
        return 'prefix-key'

    @asynccontextmanager
    async def request_handle():
        yield SimpleNamespace(async_cancel=cancel, trim_routed_experts=trim)

    @asynccontextmanager
    async def safe_run(*args, **kwargs):
        async def generate():
            # The third ID in this MTP batch is past the text stop.
            ids = [1, 6, 1]
            probs = [{i: -0.1} for i in ids]
            if terminal == 'same_batch':
                yield EngineOutput(ResponseType.FINISH, ids, logprobs=probs, routed_experts=experts)
                return
            yield EngineOutput(ResponseType.SUCCESS, ids, logprobs=probs)
            assert cancelled == [7]
            yield EngineOutput(ResponseType.SUCCESS, [1], logprobs=[{1: -0.2}])
            if terminal == 'error':
                yield EngineOutput(ResponseType.INTERNAL_ENGINE_ERROR, [])
            elif terminal != 'missing':
                yield EngineOutput(terminal, [], routed_experts=experts)
        yield generate()

    session = SimpleNamespace(session_id=7, epoch=None, request_handle=request_handle)
    engine = AsyncEngine.__new__(AsyncEngine)
    engine.session_mgr = SimpleNamespace(remove=lambda _: None)
    engine.tokenizer = Tokenizer()
    engine.num_spec_token = 5
    engine.safe_run = safe_run
    engine.request_logger = SimpleNamespace(log_response=lambda *args: None)
    config = GenerationConfig(return_routed_experts=True, routed_experts_start_len=start)
    stop = ChatContentStop(['answer 6'], include_stop=False)
    prep = PreprocessedRequest(session=session, inputs={'input_ids': [0, 0]}, input_token_len=2,
                               gen_config=config, adapter_name=None)
    request = ChatCompletionRequest(model='test', messages=[], return_token_ids=True, return_routed_experts=True)
    runner = ChatRunner(request, Parser(), config, session, engine.generate(prep, content_stop=stop),
                        engine.session_mgr, content_stop=stop)
    if terminal in ('error', 'missing'):
        with pytest.raises(RequestError):
            asyncio.run(runner.collect())
        assert runner._closed
        return

    result = asyncio.run(runner.collect())
    assert (result.finish_reason, result.stop_sequence, result.text) == ('stop', 'answer 6', None)
    assert result.token_ids == [1, 6]
    assert result.generate_token_len == 2
    assert [(o.status, o.token_ids) for o in updates] == [(ResponseType.FINISH, [1, 6])]
    assert result.logprobs == [{1: -0.1}, {6: -0.1}]
    assert result.routed_experts == ('prefix-key' if use_ref else list(range(start, 3)))
    assert trimmed == ([('batch-key', 3 - start)] if use_ref else [])
    assert cancelled == ([] if terminal == 'same_batch' else [7])
    assert runner._closed


def test_stop_in_tool_arguments_preserves_literal_and_breaks_content_prefix():
    stop = ChatContentStop(['STOP_HERE'], False)
    tool = DeltaMessage(tool_calls=[{'index': 0, 'function': {'arguments': 'STOP_HERE'}}])
    deltas = [(DeltaMessage(content='answer ST'), False), (tool, True), (DeltaMessage(content='OP_HERE'), False)]
    output = stop.filter_deltas(deltas, final=True)
    assert [msg.content for msg, _ in output] == ['answer ', 'ST', None, 'OP_HERE']
    assert output[2] == (tool, True)
    assert stop.matched is None
