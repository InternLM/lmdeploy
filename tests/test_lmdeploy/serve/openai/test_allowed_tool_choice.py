from lmdeploy.serve.openai.protocol import (
    AllowedToolChoice,
    AllowedTools,
    ChatCompletionRequest,
    Function,
    Tool,
)
from lmdeploy.serve.parsers.response_parser import BaseResponseParser


def test_allowed_tools_model():
    at = AllowedTools(mode='auto', tools=[{'type': 'function', 'function': {'name': 'get_weather'}}])
    assert at.mode == 'auto'
    assert len(at.tools) == 1


def test_allowed_tool_choice_model():
    at = AllowedTools(mode='required', tools=[{'type': 'function', 'function': {'name': 'get_weather'}}])
    atc = AllowedToolChoice(allowed_tools=at)
    assert atc.type == 'allowed_tools'
    assert atc.allowed_tools.mode == 'required'


def test_chat_completion_request_accepts_allowed_tool_choice():
    at = AllowedTools(mode='auto', tools=[{'type': 'function', 'function': {'name': 'get_weather'}}])
    req = ChatCompletionRequest(model='test', messages=[], tool_choice=AllowedToolChoice(allowed_tools=at))
    assert isinstance(req.tool_choice, AllowedToolChoice)


def test_dump_tools_filters_by_allowed_tool_choice():
    tools = [
        Tool(function=Function(name='get_weather')),
        Tool(function=Function(name='get_time')),
        Tool(function=Function(name='search')),
    ]
    at = AllowedTools(mode='auto', tools=[
        {'type': 'function', 'function': {'name': 'get_weather'}},
        {'type': 'function', 'function': {'name': 'search'}},
    ])
    req = ChatCompletionRequest(
        model='test',
        messages=[],
        tools=tools,
        tool_choice=AllowedToolChoice(allowed_tools=at),
    )
    dumped = BaseResponseParser.dump_tools(req)
    dumped_names = [t['name'] for t in dumped.tools]
    assert 'get_weather' in dumped_names
    assert 'search' in dumped_names
    assert 'get_time' not in dumped_names


def test_chat_completion_request_still_accepts_string_tool_choice():
    for choice in ('auto', 'required', 'none'):
        req = ChatCompletionRequest(model='test', messages=[], tool_choice=choice)
        assert req.tool_choice == choice
