# Copyright (c) OpenMMLab. All rights reserved.
import pytest

from lmdeploy.serve.openai.protocol import (
    AllowedToolChoice,
    AllowedTools,
    ChatCompletionRequest,
    Function,
    Tool,
)
from lmdeploy.serve.parsers.response_parser import BaseResponseParser


def test_dump_tools_filters_by_allowed_tool_choice():
    """dump_tools filters tools to only those in AllowedToolChoice."""
    tools = [
        Tool(function=Function(name='get_weather')),
        Tool(function=Function(name='get_time')),
        Tool(function=Function(name='search')),
    ]
    allowed = AllowedTools(mode='auto', tools=[
        {'type': 'function', 'function': {'name': 'get_weather'}},
        {'type': 'function', 'function': {'name': 'search'}},
    ])
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=tools,
        tool_choice=AllowedToolChoice(allowed_tools=allowed),
    )

    dumped = BaseResponseParser.dump_tools(request)

    dumped_names = [t['name'] for t in dumped.tools]
    assert dumped_names == ['get_weather', 'search']


def test_dump_tools_populates_from_allowed_tools_when_request_tools_missing():
    """dump_tools uses allowed_tools.tools when top-level tools is omitted."""
    allowed = AllowedTools(mode='auto', tools=[
        {'type': 'function', 'function': {'name': 'get_weather', 'description': 'weather'}},
        {'type': 'function', 'function': {'name': 'search', 'description': 'search'}},
    ])
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=None,
        tool_choice=AllowedToolChoice(allowed_tools=allowed),
    )

    dumped = BaseResponseParser.dump_tools(request)

    dumped_names = [t['name'] for t in dumped.tools]
    assert dumped_names == ['get_weather', 'search']
    assert dumped.tools[0]['description'] == 'weather'


def test_dump_tools_raises_when_allowed_tool_missing_from_request_tools():
    """dump_tools rejects allowed tool names that are not in request.tools."""
    tools = [Tool(function=Function(name='get_weather'))]
    allowed = AllowedTools(mode='auto', tools=[
        {'type': 'function', 'function': {'name': 'get_weather'}},
        {'type': 'function', 'function': {'name': 'search'}},
    ])
    request = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=tools,
        tool_choice=AllowedToolChoice(allowed_tools=allowed),
    )

    with pytest.raises(ValueError, match="Allowed tool\\(s\\) not found in request.tools: \\['search'\\]"):
        BaseResponseParser.dump_tools(request)
