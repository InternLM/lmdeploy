# Copyright (c) OpenMMLab. All rights reserved.
"""Tests for the ``structured_outputs`` request field (choice/grammar).

Covers Task 8: the new ``StructuredOutputs`` model on ``ChatCompletionRequest``,
the serving-layer merge of ``structured_outputs`` into ``gen_config.response_format``
(taking precedence over ``response_format``), and the engine compile branches for
``choice``/``grammar`` on the pytorch ``GuidedDecodingManager``.

The brief's path ``tests/test_openai_api/`` does not exist in this repo; the real
test root is ``tests/test_lmdeploy/serve/openai/``.
"""
import xgrammar as xgr

from lmdeploy.serve.openai.chat_completions.guided import compile_choice
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    StructuredOutputs,
)


def _make_compiler() -> xgr.GrammarCompiler:
    tokenizer_info = xgr.TokenizerInfo([], xgr.VocabType.RAW)
    return xgr.GrammarCompiler(tokenizer_info)


def test_structured_outputs_choice_accepted():
    req = ChatCompletionRequest(
        model='m',
        messages=[{'role': 'user', 'content': 'hi'}],
        structured_outputs=StructuredOutputs(choice=['yes', 'no']),
    )
    assert req.structured_outputs.choice == ['yes', 'no']


def test_structured_outputs_grammar_accepted():
    so = StructuredOutputs(grammar='root ::= "foo" | "bar"')
    assert so.grammar == 'root ::= "foo" | "bar"'


def test_choice_compiles_to_alternation_grammar():
    g = compile_choice(['yes', 'no'])
    assert g is not None  # xgrammar Grammar object


def test_pytorch_compiles_grammar_branch():
    """The pytorch ``_compile`` grammar branch compiles an EBNF string."""
    from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager
    # Avoid building a full manager; call _compile via the compile entrypoint
    # using a stand-in compiler.
    compiler = _make_compiler()

    class _Stub:
        pass

    stub = _Stub()
    stub.compiler = compiler
    compiled = GuidedDecodingManager._compile.__get__(stub)(  # type: ignore[attr-defined]
        'root ::= "foo" | "bar"', 'grammar')
    assert compiled is not None


def test_pytorch_compiles_choice_branch():
    """The pytorch ``_compile`` choice branch compiles a choice list."""
    from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager
    compiler = _make_compiler()

    class _Stub:
        pass

    stub = _Stub()
    stub.compiler = compiler
    compiled = GuidedDecodingManager._compile.__get__(stub)(  # type: ignore[attr-defined]
        ['yes', 'no'], 'choice')
    assert compiled is not None


def test_serving_merges_structured_outputs_choice_into_response_format():
    """serving.py merges structured_outputs into gen_config.response_format,
    taking precedence over response_format."""
    from lmdeploy.serve.openai.chat_completions import serving

    req = ChatCompletionRequest(
        model='m',
        messages=[{'role': 'user', 'content': 'hi'}],
        response_format={'type': 'json_object'},
        structured_outputs=StructuredOutputs(choice=['yes', 'no']),
    )
    gen_config_rf = serving._structured_outputs_to_response_format(req)
    assert gen_config_rf == {'type': 'choice', 'choice': ['yes', 'no']}


def test_serving_merges_structured_outputs_grammar_into_response_format():
    from lmdeploy.serve.openai.chat_completions import serving

    req = ChatCompletionRequest(
        model='m',
        messages=[{'role': 'user', 'content': 'hi'}],
        structured_outputs=StructuredOutputs(grammar='root ::= "foo" | "bar"'),
    )
    gen_config_rf = serving._structured_outputs_to_response_format(req)
    assert gen_config_rf == {'type': 'grammar', 'grammar': 'root ::= "foo" | "bar"'}


def test_serving_structured_outputs_none_returns_none():
    from lmdeploy.serve.openai.chat_completions import serving

    req = ChatCompletionRequest(
        model='m',
        messages=[{'role': 'user', 'content': 'hi'}],
    )
    assert serving._structured_outputs_to_response_format(req) is None
