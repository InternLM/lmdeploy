# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

import pytest

from lmdeploy.serve.core.chat_runner import should_validate_complete
from lmdeploy.serve.openai.chat_completions.validation import check_request
from lmdeploy.serve.openai.protocol import ChatCompletionRequest


def _tools():
    return [{
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'parameters': {
                'type': 'object',
                'properties': {
                    'city': {
                        'type': 'string',
                    },
                },
                'required': ['city'],
            },
        },
    }]


def _request(**kwargs):
    defaults = {
        'model': 'test-model',
        'messages': [{
            'role': 'user',
            'content': 'weather?',
        }],
        'tools': _tools(),
        'tool_choice': 'required',
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


class _SessionManager:

    @staticmethod
    def has(_session_id):
        return False


class _SupportedParser:
    tool_parser_cls = object()

    @classmethod
    def supports_required_tool_choice(cls):
        return True


class _UnsupportedParser:
    tool_parser_cls = object()

    @classmethod
    def supports_required_tool_choice(cls):
        return False


def _server_context(response_parser_cls):
    return SimpleNamespace(
        engine_config=SimpleNamespace(
            logprobs_mode=None,
            enable_return_routed_experts=False,
        ),
        session_manager=_SessionManager(),
        response_parser_cls=response_parser_cls,
    )


@pytest.mark.parametrize('tools', [None, []])
def test_required_rejects_missing_tools(tools):
    error = check_request(_request(tools=tools), _server_context(_SupportedParser))

    assert error == '`tool_choice="required"` requires at least one tool.'


@pytest.mark.parametrize(
    'parser_cls',
    [
        None,
        SimpleNamespace(tool_parser_cls=None),
    ],
)
def test_required_rejects_missing_tool_parser(parser_cls):
    error = check_request(_request(), _server_context(parser_cls))

    assert '--tool-call-parser' in error
    assert 'if you want to use tools' in error


def test_required_rejects_unsupported_tool_parser():
    error = check_request(_request(), _server_context(_UnsupportedParser))

    assert 'does not support' in error
    assert '`tool_choice="required"`' in error


def test_required_accepts_supported_tool_parser():
    assert check_request(_request(), _server_context(_SupportedParser)) == ''


def test_nested_allowed_tools_required_keeps_existing_behavior():
    request = _request(
        tool_choice={
            'type': 'allowed_tools',
            'allowed_tools': {
                'mode': 'required',
                'tools': [{
                    'type': 'function',
                    'function': {
                        'name': 'get_weather',
                    },
                }],
            },
        },
    )

    # The nested form is not routed through literal "required" support.
    assert check_request(request, _server_context(_UnsupportedParser)) == ''


@pytest.mark.parametrize(
    ('finish_reason', 'expected'),
    [
        ('stop', True),
        ('length', False),
        ('abort', False),
        (None, False),
    ],
)
def test_required_terminal_validation_preserves_length(finish_reason, expected):
    assert should_validate_complete(_request(), finish_reason) is expected
