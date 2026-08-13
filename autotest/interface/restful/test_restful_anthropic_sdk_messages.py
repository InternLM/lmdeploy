from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip('anthropic')

from utils.anthropic_messages import get_async_anthropic_client_and_model
from utils.constant import BACKEND_LIST, RESTFUL_MODEL_LIST


def _text_from_message(msg) -> str:
    parts: list[str] = []
    for block in msg.content:
        if block.type == 'text':
            parts.append(block.text)
    return ''.join(parts)


def _first_message_start_usage(events: list) -> tuple[int, int]:
    for ev in events:
        if ev.type != 'message_start':
            continue
        usage = ev.message.usage
        return usage.input_tokens, usage.output_tokens
    raise AssertionError('message_start with usage not found in stream')


async def _sdk_simple_non_stream() -> object:
    client, model_name = get_async_anthropic_client_and_model()
    return await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0.01,
        messages=[{'role': 'user', 'content': 'how are you!'}],
    )


async def _sdk_system_non_stream() -> object:
    client, model_name = get_async_anthropic_client_and_model()
    return await client.messages.create(
        model=model_name,
        max_tokens=1024,
        temperature=0.01,
        system=[{'type': 'text', 'text': 'you are a helpful assistant'}],
        messages=[{'role': 'user', 'content': 'how are you!'}],
    )


async def _sdk_stream_events_and_final() -> tuple[list, object | None]:
    client, model_name = get_async_anthropic_client_and_model()
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
    """SDK smoke: simple / system / streaming Messages API."""

    def test_sdk_simple_messages_non_stream(self, backend, model_case):
        msg = asyncio.run(_sdk_simple_non_stream())
        assert msg.role == 'assistant'
        assert msg.stop_reason in ('end_turn', 'max_tokens')
        text = _text_from_message(msg)
        assert len(text) > 0
        usage = msg.usage
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0

    def test_sdk_system_message_non_stream(self, backend, model_case):
        msg = asyncio.run(_sdk_system_non_stream())
        assert msg.role == 'assistant'
        assert msg.stop_reason in ('end_turn', 'max_tokens')
        text = _text_from_message(msg)
        assert len(text) > 0

    def test_sdk_streaming(self, backend, model_case):
        events, final_msg = asyncio.run(_sdk_stream_events_and_final())
        assert len(events) > 0

        in0, out0 = _first_message_start_usage(events)
        assert in0 >= 0
        assert out0 >= 0

        if final_msg is not None:
            assert final_msg.role == 'assistant'
            u = final_msg.usage
            assert u.input_tokens > 0
            assert u.output_tokens > 0
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
