# Copyright (c) OpenMMLab. All rights reserved.
"""Reject untrusted DistServe fields on /v1/chat/completions."""
from __future__ import annotations

import asyncio

from fastapi.responses import JSONResponse

from lmdeploy.serve.openai.protocol import ChatCompletionRequest

_POISON_MIGRATION = {
    'protocol': 3,
    'remote_engine_id': 'http://127.0.0.1:19001',
    'remote_session_id': 123456789,
    'remote_token_id': 1,
    'remote_block_ids': [],
    'is_dummy_prefill': False,
}


def test_chat_completions_rejects_untrusted_migration_request(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    fake_raw_request._payload = {'migration_request': _POISON_MIGRATION}
    request = ChatCompletionRequest(
        model='fake-model',
        messages=[{
            'role': 'user',
            'content': 'hi'
        }],
    )
    response = asyncio.run(endpoint(request, fake_raw_request))
    assert isinstance(response, JSONResponse)
    assert response.status_code == 400
    assert 'DistServe-only' in response.body.decode()
    assert context.async_engine.call_count == 0


def test_chat_completions_still_serves_after_untrusted_migration_request(
        chat_endpoint, fake_raw_request):
    endpoint, context = chat_endpoint
    fake_raw_request._payload = {'migration_request': _POISON_MIGRATION}
    rejected = asyncio.run(
        endpoint(
            ChatCompletionRequest(
                model='fake-model',
                messages=[{
                    'role': 'user',
                    'content': 'hi'
                }],
            ),
            fake_raw_request,
        ))
    fake_raw_request._payload = {}
    ok = asyncio.run(
        endpoint(
            ChatCompletionRequest(
                model='fake-model',
                messages=[{
                    'role': 'user',
                    'content': 'hi'
                }],
            ),
            fake_raw_request,
        ))
    assert isinstance(rejected, JSONResponse)
    assert rejected.status_code == 400
    assert ok['object'] == 'chat.completion'
    assert context.async_engine.call_count == 1
