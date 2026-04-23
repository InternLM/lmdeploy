# Copyright (c) OpenMMLab. All rights reserved.
import base64
import struct
from http import HTTPStatus
from unittest.mock import MagicMock, patch

import pytest
import torch

from lmdeploy.messages import Response
from lmdeploy.serve.openai.api_server import VariableInterface, create_embeddings
from lmdeploy.serve.openai.protocol import EmbeddingsRequest


def _async_gen(items):
    """Helper to create an async generator from a list of items."""
    async def gen():
        for item in items:
            yield item
    return gen()


def _mock_response(text='', finish_reason='stop', input_token_len=5,
                   last_hidden_state=None, token_ids=None):
    return Response(
        text=text,
        generate_token_len=0 if finish_reason != 'length' else 1,
        input_token_len=input_token_len,
        finish_reason=finish_reason,
        token_ids=token_ids or [],
        last_hidden_state=last_hidden_state,
        index=0,
    )


@pytest.mark.asyncio
async def test_embeddings_single_string():
    hidden = torch.tensor([0.1, 0.2, 0.3])
    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = MagicMock(return_value=_async_gen([
        _mock_response(finish_reason=None, last_hidden_state=None),
        _mock_response(finish_reason='stop', last_hidden_state=hidden, input_token_len=5),
    ]))

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input='hello world', model='test-model'))

    assert resp.model == 'test-model'
    assert resp.usage.prompt_tokens == 5
    assert resp.usage.total_tokens == 5
    assert resp.usage.completion_tokens == 0
    assert len(resp.data) == 1
    assert resp.data[0]['index'] == 0
    assert resp.data[0]['object'] == 'embedding'
    assert len(resp.data[0]['embedding']) == 3


@pytest.mark.asyncio
async def test_embeddings_list_input():
    hidden1 = torch.tensor([0.1, 0.2])
    hidden2 = torch.tensor([0.3, 0.4])
    call_count = 0

    def mock_generate(**kwargs):
        nonlocal call_count
        call_count += 1
        hidden = hidden1 if call_count == 1 else hidden2
        return _async_gen([
            _mock_response(finish_reason='stop', last_hidden_state=hidden, input_token_len=3),
        ])

    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = mock_generate

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input=['first text', 'second text']))

    assert len(resp.data) == 2
    assert resp.data[0]['embedding'] == pytest.approx([0.1, 0.2])
    assert resp.data[1]['embedding'] == pytest.approx([0.3, 0.4])
    assert resp.usage.prompt_tokens == 6  # 3 + 3


@pytest.mark.asyncio
async def test_embeddings_base64_format():
    hidden = torch.tensor([0.1, 0.2, 0.3])
    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = MagicMock(return_value=_async_gen([
        _mock_response(finish_reason='stop', last_hidden_state=hidden, input_token_len=5),
    ]))

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(
            EmbeddingsRequest(input='hello', encoding_format='base64'))

    embedding = resp.data[0]['embedding']
    assert isinstance(embedding, str)
    decoded = struct.unpack('<3f', base64.b64decode(embedding.encode('utf-8'), validate=True))
    assert decoded == pytest.approx((0.1, 0.2, 0.3))


@pytest.mark.asyncio
async def test_embeddings_empty_input():
    engine = MagicMock()
    engine.model_name = 'test-model'
    with patch.object(VariableInterface, 'async_engine', engine):
        resp = await create_embeddings(EmbeddingsRequest(input=''))
    assert resp.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_embeddings_empty_list():
    engine = MagicMock()
    engine.model_name = 'test-model'
    with patch.object(VariableInterface, 'async_engine', engine):
        resp = await create_embeddings(EmbeddingsRequest(input=[]))
    assert resp.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_embeddings_empty_text_in_list():
    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = MagicMock(return_value=_async_gen([
        _mock_response(finish_reason='stop', last_hidden_state=torch.tensor([0.1]), input_token_len=5),
    ]))

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input=['valid', '']))
    assert resp.status_code == HTTPStatus.BAD_REQUEST


@pytest.mark.asyncio
async def test_embeddings_no_hidden_states():
    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = MagicMock(return_value=_async_gen([
        _mock_response(finish_reason='stop', last_hidden_state=None, input_token_len=5),
    ]))

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input='hello'))
    assert resp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_embeddings_error_finish_reason():
    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = MagicMock(return_value=_async_gen([
        _mock_response(text='prefix caching conflict',
                       finish_reason='error', input_token_len=0),
    ]))

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input='hello'))
    assert resp.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


@pytest.mark.asyncio
async def test_embeddings_model_default_when_none():
    hidden = torch.tensor([0.1, 0.2, 0.3])
    engine = MagicMock()
    engine.model_name = 'default-model-name'
    engine.generate = MagicMock(return_value=_async_gen([
        _mock_response(finish_reason='stop', last_hidden_state=hidden, input_token_len=5),
    ]))

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input='hello'))
    assert resp.model == 'default-model-name'


@pytest.mark.asyncio
async def test_embeddings_prompt_tokens_summed():
    hidden = torch.tensor([0.1])
    call_count = 0

    def mock_generate(**kwargs):
        nonlocal call_count
        call_count += 1
        return _async_gen([
            _mock_response(finish_reason='stop', last_hidden_state=hidden, input_token_len=call_count * 10),
        ])

    engine = MagicMock()
    engine.model_name = 'test-model'
    engine.generate = mock_generate

    with patch.object(VariableInterface, 'async_engine', engine), \
         patch.object(VariableInterface, 'create_session', return_value=MagicMock(session_id=0)):
        resp = await create_embeddings(EmbeddingsRequest(input=['a', 'b', 'c']))

    # prompt_tokens = 10 + 20 + 30 = 60
    assert resp.usage.prompt_tokens == 60
    assert resp.usage.total_tokens == 60
