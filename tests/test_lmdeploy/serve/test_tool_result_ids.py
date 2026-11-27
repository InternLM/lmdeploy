# Copyright (c) OpenMMLab. All rights reserved.
import copy

import pytest

from lmdeploy.serve.openai.protocol import ChatCompletionRequest
from lmdeploy.serve.parsers.response_parser import normalize_chat_request


def history():
    return [dict(role='assistant', content=None, tool_calls=[
        dict(id=cid, type='function', function=dict(name='image', arguments='{}'))
        for cid in ('call_a', 'call_b')]),
        dict(role='tool', tool_call_id='call_b', content='second'),
        dict(role='tool', tool_call_id='call_a', content='first')]


@pytest.mark.parametrize('kind', ['missing', 'unknown', 'duplicate'])
def test_reject_invalid_tool_result_associations(kind):
    messages = history()
    if kind == 'missing':
        messages[1].pop('tool_call_id')
    elif kind == 'unknown':
        messages[1]['tool_call_id'] = 'call_absent'
    else:
        messages.append(copy.deepcopy(messages[1]))
    with pytest.raises(ValueError, match='tool_call_id'):
        normalize_chat_request(ChatCompletionRequest(model='test', messages=messages))


def test_reversed_parallel_results_preserve_order_and_input():
    messages = history()
    original = copy.deepcopy(messages)
    result = normalize_chat_request(ChatCompletionRequest(model='test', messages=messages))
    assert [m['tool_call_id'] for m in result.messages[1:]] == ['call_b', 'call_a']
    assert messages == original


def test_partial_history_keeps_existing_contract():
    messages = [dict(role='tool', tool_call_id='external_call', content='result')]
    assert normalize_chat_request(ChatCompletionRequest(model='test', messages=messages)).messages == messages


def test_result_ids_are_scoped_to_each_assistant_call_group():
    # The checkpoint template resolves IDs within the preceding assistant group.
    # Do not impose a new global uniqueness rule on historical client messages.
    messages = history() + [dict(role='user', content='Again.')] + history()
    result = normalize_chat_request(ChatCompletionRequest(model='test', messages=messages))
    assert len(result.messages) == len(messages)


def test_result_cannot_match_a_call_from_an_older_group():
    messages = history()[:1] + [
        dict(role='assistant', content=None, tool_calls=[
            dict(id='call_new', type='function', function=dict(name='image', arguments='{}'))]),
        dict(role='tool', tool_call_id='call_a', content='stale result')]
    with pytest.raises(ValueError, match='tool_call_id'):
        normalize_chat_request(ChatCompletionRequest(model='test', messages=messages))
