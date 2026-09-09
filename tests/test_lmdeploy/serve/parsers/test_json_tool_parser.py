from lmdeploy.serve.parsers.tool_parser import JsonToolParser

from .helpers import final_tool_call


class _FunctionFieldJsonParser(JsonToolParser):
    name_field = 'function'

    @classmethod
    def get_tool_open_tag(cls) -> str:
        return '<tool_call>'

    @classmethod
    def get_tool_close_tag(cls) -> str:
        return '</tool_call>'


def test_function_name_field_is_configurable():
    tool_call = final_tool_call(
        _FunctionFieldJsonParser(),
        '{"function":"f","arguments":{"x":1}}',
    )

    assert tool_call.function.name == 'f'
    assert tool_call.function.arguments == '{"x":1}'
