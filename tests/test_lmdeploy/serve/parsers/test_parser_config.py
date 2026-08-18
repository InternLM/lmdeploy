import pytest

from lmdeploy.serve.parsers import ResponseParserManager, validate_parser_names
from lmdeploy.serve.parsers.reasoning_parser import ReasoningParserManager
from lmdeploy.serve.parsers.tool_parser import ToolParserManager


@pytest.fixture
def default_response_parser_cls():
    cls = ResponseParserManager.get('default')
    reasoning_parser_cls = cls.reasoning_parser_cls
    tool_parser_cls = cls.tool_parser_cls
    try:
        cls.reasoning_parser_cls = None
        cls.tool_parser_cls = None
        yield cls
    finally:
        cls.reasoning_parser_cls = reasoning_parser_cls
        cls.tool_parser_cls = tool_parser_cls


def test_validate_parser_names_rejects_unknown_tool_parser_before_tokenizer():
    with pytest.raises(ValueError, match='The tool parser default is not in the parser list'):
        validate_parser_names(reasoning_parser_name='qwen-qwq', tool_parser_name='default')


def test_validate_parser_names_maps_legacy_reasoning_parser():
    reasoning_parser_name, tool_parser_name = validate_parser_names(
        reasoning_parser_name='qwen-qwq',
        tool_parser_name='interns2-preview',
        warn_legacy=False,
    )

    assert reasoning_parser_name == 'default'
    assert tool_parser_name == 'interns2-preview'


def test_response_parser_set_parsers_rejects_unknown_tool_parser(default_response_parser_cls):
    with pytest.raises(ValueError, match='The tool parser default is not in the parser list'):
        default_response_parser_cls.set_parsers(reasoning_parser_name='default', tool_parser_name='default')

    assert default_response_parser_cls.reasoning_parser_cls is None
    assert default_response_parser_cls.tool_parser_cls is None


def test_response_parser_set_parsers_accepts_registered_names(default_response_parser_cls):
    default_response_parser_cls.set_parsers(reasoning_parser_name='default', tool_parser_name='interns2-preview')

    assert default_response_parser_cls.reasoning_parser_cls is ReasoningParserManager.get('default')
    assert default_response_parser_cls.tool_parser_cls is ToolParserManager.get('interns2-preview')
