# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.serve.anthropic.adapter import normalize_tool_choice
from lmdeploy.serve.anthropic.protocol import MessagesRequest


def test_messages_accepts_official_none_tool_choice():
    request = MessagesRequest(model='test', max_tokens=16,
                              messages=[dict(role='user', content='Answer directly.')],
                              tool_choice={'type': 'none'},
                              tools=[dict(name='lookup', input_schema={'type': 'object'})])
    assert normalize_tool_choice(request.tool_choice) == 'none'


def test_auto_tool_choice_remains_auto():
    request = MessagesRequest(model='test', max_tokens=16,
                              messages=[dict(role='user', content='Hello.')], tool_choice={'type': 'auto'})
    assert normalize_tool_choice(request.tool_choice) == 'auto'
