# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for ``tool_choice='required'`` forced decoding (Task 4b).

The brief placed these under ``tests/test_openai_api/`` which does not exist in
this repo; the canonical chat-completions test location is
``tests/test_lmdeploy/serve/openai/chat_completions/``.
"""
import pytest

from lmdeploy.serve.openai.protocol import (
    AllowedToolChoice,
    AllowedTools,
    ChatCompletionRequest,
    Function,
    Tool,
)
from lmdeploy.serve.parsers.response_parser import (
    BaseResponseParser,
    build_required_response_format,
)


class _FakeToolParser:
    """A tool parser that exposes known begin/end tags (as classmethods, the
    same interface ``_parser_tags`` / ``_parser_tags_or_none`` introspects)."""

    OPEN = '<tool>'
    CLOSE = '</tool>'

    @classmethod
    def get_tool_open_tag(cls):
        return cls.OPEN

    @classmethod
    def get_tool_close_tag(cls):
        return cls.CLOSE


@pytest.fixture
def fake_parser_cls():
    """Configure ``BaseResponseParser`` with a fake tool parser for the test,
    restoring the original class attribute afterwards."""
    orig = BaseResponseParser.tool_parser_cls
    BaseResponseParser.tool_parser_cls = _FakeToolParser
    try:
        yield _FakeToolParser
    finally:
        BaseResponseParser.tool_parser_cls = orig


@pytest.fixture
def no_parser_cls():
    """Ensure ``BaseResponseParser`` has no tool parser configured, restoring
    the original class attribute afterwards."""
    orig = BaseResponseParser.tool_parser_cls
    BaseResponseParser.tool_parser_cls = None
    try:
        yield
    finally:
        BaseResponseParser.tool_parser_cls = orig


def _req(tool_choice='required', tools=None):
    return ChatCompletionRequest(
        model='m',
        messages=[{'role': 'user', 'content': 'hi'}],
        tool_choice=tool_choice,
        tools=tools,
    )


def test_required_produces_structural_tag_for_all_tools():
    """tool_choice='required' must force a structural_tag over ALL tools, not
    just pass them to the template like 'auto'."""
    tools = [{'type': 'function', 'function': {
        'name': 'get_weather',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    rf = build_required_response_format(tools, open_tag='[TOOL]', close_tag='[/TOOL]')
    assert rf is not None
    assert rf['type'] == 'structural_tag'
    assert '[TOOL]' in str(rf) and '[/TOOL]' in str(rf)


def test_required_without_tools_raises():
    with pytest.raises(ValueError, match='required.*tools'):
        build_required_response_format([], open_tag='[TOOL]', close_tag='[/TOOL]')


def test_required_uses_allowed_tools_subset():
    """With AllowedToolChoice, required should constrain only the allowed
    subset."""
    tools = [{'type': 'function', 'function': {'name': 'a', 'parameters': {'type': 'object'}}},
             {'type': 'function', 'function': {'name': 'b', 'parameters': {'type': 'object'}}}]
    rf = build_required_response_format(tools, open_tag='[TOOL]', close_tag='[/TOOL]',
                                        allowed_names={'b'})
    assert rf is not None
    # tool 'a' is filtered out; only 'b' remains in the union
    assert "'a'" not in str(rf)


def test_required_forces_at_least_one_tool_call():
    """Required must set at_least_one=True so the model cannot emit only free
    text (unlike 'auto'+strict which is optional)."""
    tools = [{'type': 'function', 'function': {
        'name': 'get_weather',
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    rf = build_required_response_format(tools, open_tag='[TOOL]', close_tag='[/TOOL]')
    payload = rf['structural_tag']
    # the forced payload uses the native triggered_tags format with at_least_one
    assert payload['format']['type'] == 'triggered_tags'
    assert payload['format']['at_least_one'] is True


def test_required_not_treated_as_auto_in_dump_tools(fake_parser_cls):
    """dump_tools must set a forced response_format for required (NOT silently
    treat it as 'auto' and inject nothing).

    A configured tool parser with known tags is required to produce a non-None response_format.
    """
    req = _req(tools=[Tool(function=Function(name='f', parameters={'type': 'object'}))])
    dumped = BaseResponseParser.dump_tools(req)
    assert dumped.response_format is not None
    assert dumped.response_format.get('type') == 'structural_tag'
    # the structural_tag must carry the parser's real begin/end tags
    payload = dumped.response_format['structural_tag']
    fmt = payload['format']
    assert fmt['triggers'] == [fake_parser_cls.OPEN]
    assert fmt['tags'][0]['begin'] == fake_parser_cls.OPEN
    assert fmt['tags'][0]['end'] == fake_parser_cls.CLOSE
    assert fmt['at_least_one'] is True


def test_required_dump_tools_without_parser_skips_response_format(no_parser_cls):
    """Graceful-skip: when no tool parser is configured, ``dump_tools`` with
    ``required`` must NOT fabricate default tags / inject a response_format.
    Fabricating tags the model never emits would be worse than no constraint;
    ``required`` then degrades to 'auto'-like behavior (tools in the template,
    no forced grammar) ONLY in the no-parser case."""
    req = _req(tools=[Tool(function=Function(name='f', parameters={'type': 'object'}))])
    dumped = BaseResponseParser.dump_tools(req)
    # tools are still dumped for the template, but no forced response_format
    assert dumped.tools is not None and dumped.tools[0]['name'] == 'f'
    assert dumped.response_format is None


def test_required_allowed_tool_choice_mode_required_forces_subset(fake_parser_cls):
    """AllowedToolChoice with mode='required' must force a structural_tag over
    the allowed subset only, using the parser's real tags."""
    allowed = AllowedTools(mode='required', tools=[
        {'type': 'function', 'function': {'name': 'a'}},
        {'type': 'function', 'function': {'name': 'b'}},
    ])
    req = ChatCompletionRequest(
        model='m', messages=[],
        tools=[Tool(function=Function(name='a', parameters={'type': 'object'})),
               Tool(function=Function(name='b', parameters={'type': 'object'})),
               Tool(function=Function(name='c', parameters={'type': 'object'}))],
        tool_choice=AllowedToolChoice(allowed_tools=allowed),
    )
    dumped = BaseResponseParser.dump_tools(req)
    # only the allowed subset is dumped
    assert [t['name'] for t in dumped.tools] == ['a', 'b']
    assert dumped.response_format is not None
    assert dumped.response_format.get('type') == 'structural_tag'
    s = str(dumped.response_format)
    assert "'c'" not in s
    assert "'a'" in s and "'b'" in s


def test_required_allowed_tool_choice_without_parser_skips(no_parser_cls):
    """AllowedToolChoice mode='required' with no parser configured must also
    gracefully skip (no fabricated response_format)."""
    allowed = AllowedTools(mode='required', tools=[
        {'type': 'function', 'function': {'name': 'a'}},
    ])
    req = ChatCompletionRequest(
        model='m', messages=[],
        tools=[Tool(function=Function(name='a', parameters={'type': 'object'}))],
        tool_choice=AllowedToolChoice(allowed_tools=allowed),
    )
    dumped = BaseResponseParser.dump_tools(req)
    assert dumped.tools is not None and [t['name'] for t in dumped.tools] == ['a']
    assert dumped.response_format is None


def test_validation_required_without_tools_returns_error():
    """check_request must reject tool_choice='required' with no tools."""
    from lmdeploy.serve.openai.chat_completions.validation import check_request

    class _Ctx:
        class engine_config:
            logprobs_mode = None

        class session_manager:
            @staticmethod
            def has(_):
                return False

        response_parser_cls = None

    req = _req(tools=None)
    err = check_request(req, _Ctx())
    assert err
    assert 'required' in err and 'tools' in err
