# Copyright (c) OpenMMLab. All rights reserved.
"""Pure compile-time unit tests for the structural_tag guided-decoding path.

These tests do NOT require a GPU or a model. They exercise:
  * ``lmdeploy.serve.openai.chat_completions.guided`` compile helpers
  * the pytorch ``GuidedDecodingManager`` structural_tag compile branch
  * the xgrammar capability leveraged by the turbomind compile branch

The turbomind end-to-end path needs a model instance and is covered elsewhere;
here we only smoke-test that ``xgr.GrammarCompiler.compile_structural_tag`` is
callable with the structural-tag objects produced by ``guided.py``.
"""
import xgrammar as xgr

from lmdeploy.serve.openai.chat_completions.guided import (
    _to_xgr_structural_tag,
    compile_choice,
    compile_structural_tag_payload,
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


def test_compile_choice_escapes_quote_and_backslash_without_crash():
    """Option strings containing ``"`` or ``\\`` must not crash the EBNF lexer
    (previously a ``RuntimeError``) and must compile successfully."""
    grammar = compile_choice(['a"b', 'c\\d'])
    assert isinstance(grammar, xgr.Grammar)
    compiler = _make_compiler()
    compiled = compiler.compile_grammar(grammar)  # must not raise RuntimeError
    assert compiled is not None


def test_compile_choice_no_ebnf_injection():
    """An option containing ``" | "`` must be treated as a single literal — it
    must NOT inject an extra alternation that makes ``evil`` a valid match.

    We assert via the serialized grammar that the full injection string's byte
    sequence appears as one contiguous const-string literal (proving it was not
    split into separate ``"x"`` / ``"evil"`` alternations), and that the grammar
    compiles without a lexer crash.
    """
    injection = 'x" | "evil'
    grammar = compile_choice([injection])
    serialized = grammar.serialize_json()
    # The full literal's bytes must appear contiguously in the serialized
    # grammar (as one const-string), proving no alternation was injected.
    contiguous = ','.join(str(b) for b in injection.encode())
    assert contiguous in serialized, (
        f'injection string not found as one contiguous literal; serialized '
        f'grammar: {serialized}')
    # And it must compile (no RuntimeError / lexer crash).
    compiler = _make_compiler()
    compiled = compiler.compile_grammar(grammar)
    assert compiled is not None


def test_to_xgr_structural_tag_native_extra_keys_tolerated():
    """A native-format payload with extra keys must NOT raise pydantic's
    ``ValidationError`` (which would propagate uncaught past the engines'
    ``except ValueError``).

    Extra keys are stripped and conversion succeeds.
    """
    payload = {
        'type': 'structural_tag',
        'format': {'type': 'const_string', 'value': 'x'},
        'unexpected_extra_key': 'oops',
    }
    st = _to_xgr_structural_tag(payload)  # must not raise
    assert isinstance(st, xgr.StructuralTag)


def test_to_xgr_structural_tag_native_bad_format_raises_valueerror():
    """A native-format payload with an invalid format must raise ``ValueError``
    (caught by the engines' ``except ValueError``), not pydantic's
    ``ValidationError``."""
    import pytest
    with pytest.raises(ValueError):
        _to_xgr_structural_tag({'type': 'structural_tag',
                                'format': {'type': 'not_a_real_format'}})


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
