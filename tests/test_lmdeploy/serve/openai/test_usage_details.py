# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for usage ``completion_tokens_details`` and response ``service_tier``.

Covers Task 5: ``UsageInfo`` gains an optional ``completion_tokens_details``
field backed by a new ``CompletionTokensDetails`` model, and
``ChatCompletionResponse`` gains a ``service_tier`` placeholder field.
"""
from lmdeploy.serve.openai.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionTokensDetails,
    UsageInfo,
)


def test_usage_has_completion_tokens_details():
    u = UsageInfo(prompt_tokens=5,
                  completion_tokens=3,
                  total_tokens=8,
                  completion_tokens_details=CompletionTokensDetails(
                      reasoning_tokens=2))
    dumped = u.model_dump()
    assert dumped['completion_tokens_details']['reasoning_tokens'] == 2


def test_usage_completion_tokens_details_defaults_none():
    assert UsageInfo(prompt_tokens=1,
                     completion_tokens=1,
                     total_tokens=2).completion_tokens_details is None


def test_completion_tokens_details_reasoning_tokens_defaults_zero():
    d = CompletionTokensDetails()
    assert d.reasoning_tokens == 0
    # reserved OpenAI fields are present and default to None
    assert d.accepted_prediction_tokens is None
    assert d.rejected_prediction_tokens is None
    assert d.audio_tokens is None


def test_chat_completion_response_has_service_tier_default_none():
    choice = ChatCompletionResponseChoice(index=0,
                                          message=ChatMessage(role='assistant',
                                                              content='hi'),
                                          finish_reason='stop')
    resp = ChatCompletionResponse(model='m', choices=[choice],
                                  usage=UsageInfo(prompt_tokens=1,
                                                  completion_tokens=1,
                                                  total_tokens=2))
    assert resp.service_tier is None
    assert 'service_tier' in resp.model_dump()
