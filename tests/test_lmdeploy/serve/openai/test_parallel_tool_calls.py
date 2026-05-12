from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    FunctionCall,
    ToolCall,
)


def test_parallel_tool_calls_default_is_true():
    req = ChatCompletionRequest(model='test', messages=[])
    assert req.parallel_tool_calls is True


def test_parallel_tool_calls_can_be_set_false():
    req = ChatCompletionRequest(model='test', messages=[], parallel_tool_calls=False)
    assert req.parallel_tool_calls is False


def test_truncate_tool_calls_returns_all_when_true():
    from lmdeploy.serve.parsers.response_parser import BaseResponseParser
    calls = [
        ToolCall(function=FunctionCall(name='get_weather', arguments='{"city":"NYC"}')),
        ToolCall(function=FunctionCall(name='get_time', arguments='{"tz":"EST"}')),
    ]
    result = BaseResponseParser.truncate_tool_calls(calls, True)
    assert len(result) == 2


def test_truncate_tool_calls_truncates_when_false():
    from lmdeploy.serve.parsers.response_parser import BaseResponseParser
    calls = [
        ToolCall(function=FunctionCall(name='get_weather', arguments='{"city":"NYC"}')),
        ToolCall(function=FunctionCall(name='get_time', arguments='{"tz":"EST"}')),
    ]
    result = BaseResponseParser.truncate_tool_calls(calls, False)
    assert len(result) == 1
    assert result[0].function.name == 'get_weather'


def test_truncate_tool_calls_returns_none_when_none():
    from lmdeploy.serve.parsers.response_parser import BaseResponseParser
    result = BaseResponseParser.truncate_tool_calls(None, False)
    assert result is None


def test_truncate_tool_calls_returns_single_call_unchanged():
    from lmdeploy.serve.parsers.response_parser import BaseResponseParser
    calls = [ToolCall(function=FunctionCall(name='get_weather', arguments='{"city":"NYC"}'))]
    result = BaseResponseParser.truncate_tool_calls(calls, False)
    assert len(result) == 1


def test_truncate_tool_calls_default_none_does_not_truncate():
    from lmdeploy.serve.parsers.response_parser import BaseResponseParser
    calls = [
        ToolCall(function=FunctionCall(name='get_weather', arguments='{"city":"NYC"}')),
        ToolCall(function=FunctionCall(name='get_time', arguments='{"tz":"EST"}')),
    ]
    result = BaseResponseParser.truncate_tool_calls(calls, None)
    assert len(result) == 2
