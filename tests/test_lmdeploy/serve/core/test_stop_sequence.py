# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from lmdeploy.messages import EngineOutput, GenerationConfig, ResponseType
from lmdeploy.serve.core.async_engine import AsyncEngine, PreprocessedRequest
from lmdeploy.serve.core.chat_runner import ChatRunner
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
def test_engine_and_runner_preserve_actual_stop_sequence(stream, include_stop, terminal, expected):
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
    runner = ChatRunner(
        request=ChatCompletionRequest(model='fake', messages=[]), response_parser=Parser(), gen_config=config,
        session=session, result_generator=engine.generate(preprocessed), session_manager=session_manager)

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
        assert (expected in text) is include_stop
    assert runner._closed
