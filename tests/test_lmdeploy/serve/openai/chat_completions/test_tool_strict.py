# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for ``tools[].function.strict`` -> structural_tag decoding (Task 4).

The brief placed these under ``tests/test_openai_api/`` which does not exist in
this repo; the canonical chat-completions test location is
``tests/test_lmdeploy/serve/openai/chat_completions/``.
"""
from lmdeploy.serve.parsers.response_parser import (
    build_strict_tool_response_format,
    _tool_schema_to_structural_tag,
)


def test_strict_tool_produces_structural_tag_response_format():
    """A strict tool with a known parser should translate response_format
    into a structural_tag grammar so arguments conform to the schema."""
    tools = [{'type': 'function', 'function': {
        'name': 'get_weather', 'strict': True,
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    rf = build_strict_tool_response_format(tools, open_tag='<tool>', close_tag='</tool>')
    assert rf is not None
    assert rf['type'] == 'structural_tag'
    # structural_tag payload contains begin/end tags
    assert '<tool>' in str(rf) and '</tool>' in str(rf)


def test_non_strict_tool_returns_none():
    """Without strict=True the auto path must not constrain generation."""
    tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {}}}]  # no strict
    rf = build_strict_tool_response_format(tools, open_tag='<tool>', close_tag='</tool>')
    assert rf is None


def test_strict_uses_triggered_tags_so_auto_text_allowed():
    """Under tool_choice='auto' a strict tool must be OPTIONAL: free text is
    allowed and the tool call is triggered only when the model emits the
    open tag. The structural_tag must therefore be a triggered_tags payload
    (at_least_one=False), not a forced single tag."""
    tools = [{'type': 'function', 'function': {
        'name': 'get_weather', 'strict': True,
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}]
    rf = build_strict_tool_response_format(tools, open_tag='<tool>', close_tag='</tool>')
    payload = rf['structural_tag']
    # triggered_tags shape => 'tags' list present (multi-tag form), free text allowed
    assert 'tags' in payload


def test_tool_schema_to_structural_tag_includes_tool_name():
    """The wrapped schema must carry the tool name (as a const) so the model
    emits the right function, plus the arguments schema."""
    tool = {'type': 'function', 'function': {
        'name': 'get_weather', 'strict': True,
        'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                       'required': ['city']}}}
    payload = _tool_schema_to_structural_tag(tool, '<tool>', '</tool>')
    schema = payload['schema']
    assert schema['properties']['name'] == {'const': 'get_weather'}
    assert 'arguments' in schema['properties']


def test_multiple_strict_tools_build_union_schema():
    """Multiple strict tools under auto share the same open/close tags, so they
    are combined into a single tag with a oneOf union schema."""
    tools = [
        {'type': 'function', 'function': {'name': 'a', 'strict': True,
          'parameters': {'type': 'object'}}},
        {'type': 'function', 'function': {'name': 'b', 'strict': True,
          'parameters': {'type': 'object'}}},
    ]
    rf = build_strict_tool_response_format(tools, open_tag='<tool>', close_tag='</tool>')
    assert rf is not None
    payload = rf['structural_tag']
    # triggered_tags shape: single tag carrying a oneOf union schema
    schema = payload['tags'][0]['schema']
    assert 'oneOf' in schema
    names = {item['properties']['name']['const'] for item in schema['oneOf']}
    assert names == {'a', 'b'}


def test_strict_without_close_tag_returns_none():
    """Some tool parsers (e.g. llama3) have no close tag; we cannot wrap a
    JSON span without an end delimiter, so strict is a no-op there."""
    tools = [{'type': 'function', 'function': {
        'name': 'f', 'strict': True, 'parameters': {'type': 'object'}}}]
    rf = build_strict_tool_response_format(tools, open_tag='<tool>', close_tag=None)
    assert rf is None
