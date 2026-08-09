# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the ``reasoning_effort`` request field widening and forwarding.

Covers Task 3: the ``reasoning_effort`` Literal on ``ChatCompletionRequest``
is widened to ``none/minimal/low/medium/high/xhigh/max`` and *all* non-None
values are forwarded into ``chat_template_kwargs`` (previously only
``high``/``max`` were forwarded).
"""
import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import BaseResponseParser


@pytest.mark.parametrize('effort',
                         ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'])
def test_reasoning_effort_accepts_all_values(effort):
    req = ChatCompletionRequest(model='m',
                                messages=[{
                                    'role': 'user',
                                    'content': 'hi'
                                }],
                                reasoning_effort=effort)
    assert req.reasoning_effort == effort


def test_reasoning_effort_invalid_rejected():
    with pytest.raises(Exception):
        ChatCompletionRequest(model='m',
                              messages=[{
                                  'role': 'user',
                                  'content': 'hi'
                              }],
                              reasoning_effort='ultra')


def test_reasoning_effort_forwarded_to_template_kwargs():
    """response_parser should forward ALL non-None values, not just
    high/max."""
    req = ChatCompletionRequest(model='m',
                                messages=[{
                                    'role': 'user',
                                    'content': 'hi'
                                }],
                                reasoning_effort='minimal')
    kwargs = BaseResponseParser.chat_template_kwargs_from_request(req)
    assert kwargs.get('reasoning_effort') == 'minimal'


def test_reasoning_effort_none_not_forwarded():
    """When reasoning_effort is None it must not be injected into kwargs."""
    req = ChatCompletionRequest(model='m',
                                messages=[{
                                    'role': 'user',
                                    'content': 'hi'
                                }],
                                reasoning_effort=None)
    kwargs = BaseResponseParser.chat_template_kwargs_from_request(req)
    assert 'reasoning_effort' not in kwargs
