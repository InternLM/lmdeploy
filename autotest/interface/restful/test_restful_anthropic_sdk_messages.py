from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip('anthropic')

from utils.constant import BACKEND_LIST, DEFAULT_SERVER, RESTFUL_MODEL_LIST
from utils.tool_reasoning_definitions import get_async_anthropic_client_and_model

BASE_HTTP_URL = f'http://{DEFAULT_SERVER}'
DEFAULT_PORT = 23333


def _text_from_message(msg) -> str:
    parts: list[str] = []
    for block in getattr(msg, 'content', []) or []:
        if getattr(block, 'type', None) == 'text':
            parts.append(getattr(block, 'text', '') or '')
    return ''.join(parts)


def _first_message_start_usage(events: list) -> tuple[int, int] | None:
    for ev in events:
        if getattr(ev, 'type', None) != 'message_start':
            continue
        msg = getattr(ev, 'message', None)
        if msg is None:
            continue
        u = getattr(msg, 'usage', None)
        if u is None:
            return None
        return getattr(u, 'input_tokens', 0), getattr(u, 'output_tokens', 0)
    return None


async def _sdk_simple_non_stream() -> object:
    client, model_name = get_async_anthropic_client_and_model(
        base_url=f'{BASE_HTTP_URL}:{DEFAULT_PORT}',
    )
    return await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0.01,
        messages=[{'role': 'user', 'content': 'how are you!'}],
    )


async def _sdk_system_non_stream() -> object:
    client, model_name = get_async_anthropic_client_and_model(
        base_url=f'{BASE_HTTP_URL}:{DEFAULT_PORT}',
    )
    return await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0.01,
        system='you are a helpful assistant',
        messages=[{'role': 'user', 'content': 'how are you!'}],
    )


async def _sdk_stream_events_and_final() -> tuple[list, object | None]:
    client, model_name = get_async_anthropic_client_and_model(
        base_url=f'{BASE_HTTP_URL}:{DEFAULT_PORT}',
    )
    stream = await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0.01,
        messages=[{'role': 'user', 'content': 'how are you!'}],
        stream=True,
    )
    events: list = []
    async for event in stream:
        events.append(event)
    final_msg = None
    getter = getattr(stream, 'get_final_message', None)
    if callable(getter):
        try:
            final_msg = await getter()
        except Exception:
            final_msg = None
    return events, final_msg


@pytest.mark.order(8)
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize('backend', BACKEND_LIST)
@pytest.mark.parametrize('model_case', RESTFUL_MODEL_LIST)
class TestRestfulAnthropicSdkMessages:
    """Covers simple / system / streaming Messages (LMDeploy streams zero usage on ``message_start``)."""

    def test_sdk_simple_messages_non_stream(self, backend, model_case):
        msg = asyncio.run(_sdk_simple_non_stream())
        assert getattr(msg, 'role', None) == 'assistant'
        assert getattr(msg, 'stop_reason', None) in ('end_turn', 'max_tokens')
        text = _text_from_message(msg)
        assert len(text) > 0
        usage = getattr(msg, 'usage', None)
        assert usage is not None
        assert getattr(usage, 'input_tokens', 0) > 0
        assert getattr(usage, 'output_tokens', 0) > 0

    def test_sdk_system_message_non_stream(self, backend, model_case):
        msg = asyncio.run(_sdk_system_non_stream())
        assert getattr(msg, 'role', None) == 'assistant'
        assert getattr(msg, 'stop_reason', None) in ('end_turn', 'max_tokens')
        text = _text_from_message(msg)
        assert len(text) > 0

    def test_sdk_streaming(self, backend, model_case):
        events, final_msg = asyncio.run(_sdk_stream_events_and_final())
        assert len(events) > 0

        usage0 = _first_message_start_usage(events)
        assert usage0 is not None, 'message_start with usage not found in stream'
        in0, out0 = usage0
        assert out0 == 0, 'LMDeploy streams output_tokens=0 until message_delta'
        assert in0 == 0, 'LMDeploy streams input_tokens=0 on message_start (final usage appears in message_delta)'

        if final_msg is not None:
            assert getattr(final_msg, 'role', None) == 'assistant'
            u = getattr(final_msg, 'usage', None)
            assert u is not None
            assert getattr(u, 'input_tokens', 0) > 5
            assert getattr(u, 'output_tokens', 0) > 0
            text = _text_from_message(final_msg)
            assert len(text) > 0
            return

        serialised = []
        for e in events:
            if hasattr(e, 'model_dump'):
                serialised.append(e.model_dump())
            else:
                serialised.append({'repr': repr(e)})
        blob = json.dumps(serialised, default=str)
        assert 'message_delta' in blob or 'output_tokens' in blob
