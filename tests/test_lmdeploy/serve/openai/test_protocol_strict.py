from lmdeploy.serve.openai.protocol import ChatCompletionRequest, Function, Tool


def test_function_strict_field_default_is_none():
    fn = Function(name='get_weather')
    assert fn.strict is None


def test_function_strict_field_can_be_set():
    fn = Function(name='get_weather', strict=True)
    assert fn.strict is True
    fn2 = Function(name='get_weather', strict=False)
    assert fn2.strict is False


def test_tool_with_strict_passes_through():
    params = {'type': 'object', 'properties': {'city': {'type': 'string'}}}
    tool = Tool(function=Function(name='get_weather', strict=True, parameters=params))
    req = ChatCompletionRequest(model='test', messages=[], tools=[tool])
    assert req.tools[0].function.strict is True


def test_tool_without_strict_passes_through():
    tool = Tool(function=Function(name='get_weather'))
    req = ChatCompletionRequest(model='test', messages=[], tools=[tool])
    assert req.tools[0].function.strict is None


def test_function_model_dump_includes_strict():
    fn = Function(name='get_weather', strict=True, parameters={'type': 'object'})
    dumped = fn.model_dump()
    assert dumped['strict'] is True


def test_dump_tools_includes_strict():
    from lmdeploy.serve.parsers.response_parser import BaseResponseParser
    tool = Tool(function=Function(name='get_weather', strict=True, parameters={'type': 'object'}))
    req = ChatCompletionRequest(model='test', messages=[], tools=[tool])
    dumped_req = BaseResponseParser.dump_tools(req)
    assert dumped_req.tools[0]['strict'] is True
