# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for threading ``disable_any_whitespace`` to xgrammar's
``compile_json_schema`` (Task 9).

The brief's path ``tests/test_openai_api/`` does not exist in this repo; the real
test root is ``tests/test_lmdeploy/serve/openai/``.
"""
from unittest.mock import patch

import xgrammar as xgr

from lmdeploy.pytorch.engine import guided_process


def _make_compiler() -> xgr.GrammarCompiler:
    tokenizer_info = xgr.TokenizerInfo([], xgr.VocabType.RAW)
    return xgr.GrammarCompiler(tokenizer_info)


def test_compile_json_schema_passes_any_whitespace_false():
    """When disable_any_whitespace is set, compile_json_schema must be called
    with any_whitespace=False (no whitespace allowed)."""
    c = _make_compiler()
    with patch.object(c, 'compile_json_schema', wraps=c.compile_json_schema) as spy:
        guided_process._compile_json_schema_opts(c, '{"type":"object"}', any_whitespace=False)
        assert spy.call_args.kwargs.get('any_whitespace') is False


def test_compile_json_schema_passes_any_whitespace_true_default():
    """By default any_whitespace=True is forwarded (xgrammar default)."""
    c = _make_compiler()
    with patch.object(c, 'compile_json_schema', wraps=c.compile_json_schema) as spy:
        guided_process._compile_json_schema_opts(c, '{"type":"object"}')
        assert spy.call_args.kwargs.get('any_whitespace') is True


def test_structured_outputs_disable_any_whitespace_field():
    from lmdeploy.serve.openai.protocol import StructuredOutputs

    so = StructuredOutputs(choice=['yes', 'no'], disable_any_whitespace=True)
    assert so.disable_any_whitespace is True
    so2 = StructuredOutputs(choice=['yes', 'no'])
    assert so2.disable_any_whitespace is False


def test_serving_disable_any_whitespace_added_to_dict():
    """When disable_any_whitespace is set, the merged response_format dict
    carries any_whitespace=False for the JSON-schema compile path."""
    from lmdeploy.serve.openai.chat_completions import serving
    from lmdeploy.serve.openai.protocol import (
        ChatCompletionRequest,
        StructuredOutputs,
    )

    req = ChatCompletionRequest(
        model='m',
        messages=[{'role': 'user', 'content': 'hi'}],
        structured_outputs=StructuredOutputs(
            grammar='root ::= "foo"', disable_any_whitespace=True),
    )
    rf = serving._structured_outputs_to_response_format(req)
    assert rf['any_whitespace'] is False


def test_pytorch_compile_json_schema_branch_threads_any_whitespace():
    """The pytorch _compile json_schema branch forwards any_whitespace."""
    c = _make_compiler()

    class _Stub:
        pass

    stub = _Stub()
    stub.compiler = c
    with patch.object(c, 'compile_json_schema', wraps=c.compile_json_schema) as spy:
        guided_process.GuidedDecodingManager._compile.__get__(stub)(
            '{"type":"object"}', 'json_object', any_whitespace=False)
        assert spy.call_args.kwargs.get('any_whitespace') is False
