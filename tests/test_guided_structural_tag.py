# Copyright (c) OpenMMLab. All rights reserved.
"""Pure compile-time unit tests for the structural_tag guided-decoding path.

These tests do NOT require a GPU or a model. They exercise:
  * ``lmdeploy.serve.openai.endpoints.chat_completions.guided`` compile helpers
  * the pytorch ``GuidedDecodingManager`` structural_tag compile branch
  * the xgrammar capability leveraged by the turbomind compile branch

The turbomind end-to-end path needs a model instance and is covered elsewhere;
here we only smoke-test that ``xgr.GrammarCompiler.compile_structural_tag`` is
callable with the structural-tag objects produced by ``guided.py``.
"""
import xgrammar as xgr

from lmdeploy.serve.openai.endpoints.chat_completions.guided import (
    compile_choice,
    compile_structural_tag_payload,
    _to_xgr_structural_tag,
)

# A representative lmdeploy-internal structural_tag response_format dict as the
# engines will receive it (single tag wrapping a JSON schema).
STRUCTURAL_TAG_RF = {
    'type': 'structural_tag',
    'structural_tag': {
        'begin': '<a>',
        'end': '</a>',
        'schema': {
            'type': 'object',
            'properties': {
                'x': {'type': 'string'}
            },
        },
    },
}

STRUCTURAL_TAG_PAYLOAD = STRUCTURAL_TAG_RF['structural_tag']


def _make_compiler() -> xgr.GrammarCompiler:
    """Build a GrammarCompiler with a minimal empty tokenizer."""
    tokenizer_info = xgr.TokenizerInfo([], xgr.VocabType.RAW)
    return xgr.GrammarCompiler(tokenizer_info)


def test_to_xgr_structural_tag_single():
    st = _to_xgr_structural_tag(STRUCTURAL_TAG_PAYLOAD)
    assert isinstance(st, xgr.StructuralTag)


def test_to_xgr_structural_tag_multiple():
    payload = {
        'tags': [
            {'begin': '<a>', 'end': '</a>', 'schema': {'type': 'object'}},
            {'begin': '<b>', 'end': '</b>', 'schema': {'type': 'object'}},
        ]
    }
    st = _to_xgr_structural_tag(payload)
    assert isinstance(st, xgr.StructuralTag)


def test_compile_structural_tag_payload_returns_grammar():
    grammar = compile_structural_tag_payload(STRUCTURAL_TAG_PAYLOAD)
    assert isinstance(grammar, xgr.Grammar)
    # The grammar must be compilable by a real compiler.
    compiler = _make_compiler()
    compiled = compiler.compile_grammar(grammar)
    assert compiled is not None


def test_compile_choice_returns_grammar():
    grammar = compile_choice(['yes', 'no'])
    assert isinstance(grammar, xgr.Grammar)
    compiler = _make_compiler()
    compiled = compiler.compile_grammar(grammar)
    assert compiled is not None


def test_pytorch_guided_compiles_structural_tag():
    from lmdeploy.pytorch.engine.guided_process import GuidedDecodingManager
    mgr = GuidedDecodingManager.__new__(GuidedDecodingManager)  # light construct
    mgr.compiler = _make_compiler()
    compiled = mgr._compile_response_format(STRUCTURAL_TAG_RF)
    assert compiled is not None  # non-None => compiled successfully


def test_turbomind_compiles_structural_tag():
    # turbomind side: verify xgrammar can compile a structural tag built by
    # ``_to_xgr_structural_tag`` via the same call the turbomind branch uses.
    compiler = _make_compiler()
    st = _to_xgr_structural_tag(STRUCTURAL_TAG_PAYLOAD)
    g = compiler.compile_structural_tag(st)
    assert g is not None
