# Copyright (c) OpenMMLab. All rights reserved.
"""Compile helpers for guided decoding.

This module is the **single source of truth** for converting lmdeploy-internal
guided-decoding payloads (``structural_tag``, ``choice``) into xgrammar grammar
objects.  It is consumed by both engines (pytorch ``GuidedDecodingManager`` and
the turbomind ``response_format`` compile path) and by downstream
``chat_completions`` tasks (strict / required / structured_outputs).

``structural_tag`` is a vLLM/lmdeploy-internal extension and is **NOT** part of
the OpenAI ``response_format`` standard (OpenAI only defines ``text``,
``json_object`` and ``json_schema``).  It is never exposed to clients; later
tasks inject it internally to constrain JSON emitted inside tool-call tags,
reusing the existing tag-based tool-parser mechanism.

xgrammar 0.2.3 API notes (verified):
  * ``xgr.GrammarCompiler(tokenizer_info, ...)`` — requires a ``TokenizerInfo``.
  * ``compiler.compile_structural_tag(structural_tag) -> CompiledGrammar``
    where ``structural_tag`` may be an ``xgr.StructuralTag``, a JSON string, or
    a dict in xgrammar's native ``{'type': 'structural_tag', 'format': ...}``
    shape.
  * ``xgr.StructuralTag(format=...)`` — pydantic model; the ``format`` field is
    a discriminated union (``TagFormat`` for a single tag,
    ``TriggeredTagsFormat`` for multiple tags with triggers).  The legacy
    ``StructuralTag(tags=[StructuralTagItem(...)])`` constructor is **not**
    supported in 0.2.3 (``tags`` was replaced by ``format``).
  * ``xgr.StructuralTagItem(begin, end, schema)`` still exists but is
    deprecated; ``TagFormat(begin, end, content)`` is the current shape, where
    ``content`` is a format such as ``JSONSchemaFormat(json_schema=...)``.
  * ``xgr.Grammar.from_structural_tag(structural_tag) -> Grammar`` — produces an
    uncompiled ``Grammar`` from a ``StructuralTag``/dict.
  * ``xgr.Grammar.from_ebnf(ebnf) -> Grammar`` — produces an uncompiled
    ``Grammar`` from an EBNF string.
  * ``compiler.compile_grammar(grammar_or_str, *, root_rule_name='root') ->
    CompiledGrammar`` — compiles a ``Grammar`` or EBNF string.
  * ``xgr.Grammar.union(*grammars) -> Grammar`` — alternation of grammars.
"""
from typing import Any

import xgrammar as xgr
from pydantic import ValidationError


def _to_xgr_structural_tag(payload: Any) -> xgr.StructuralTag:
    """Convert a structural_tag payload dict into an ``xgr.StructuralTag``.

    Accepted payload shapes (the engines receive shape 1 or 2; shape 3 is
    accepted for forward compatibility):

    1. lmdeploy single-tag (most common):
       ``{'begin': '<a>', 'end': '</a>', 'schema': {json_schema}}``
    2. lmdeploy multi-tag:
       ``{'tags': [{'begin', 'end', 'schema'}, ...]}``
    3. already-xgrammar native dict:
       ``{'type': 'structural_tag', 'format': {...TagFormat/TriggeredTagsFormat}}``
    4. an existing ``xgr.StructuralTag`` (passed through unchanged).

    Returns an ``xgr.StructuralTag`` that can be handed to
    ``GrammarCompiler.compile_structural_tag``.
    """
    if isinstance(payload, xgr.StructuralTag):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            f'structural_tag payload must be a dict or xgr.StructuralTag, '
            f'got {type(payload).__name__}')

    # Already in xgrammar's native format.  Forward only the ``type``/``format``
    # keys xgrammar expects so extra payload keys don't raise pydantic's
    # ``ValidationError`` (which would propagate uncaught past the engines'
    # ``except ValueError``).  Convert any validation error into ``ValueError``
    # for the same reason.
    if 'format' in payload:
        native = {'type': payload.get('type', 'structural_tag'),
                  'format': payload['format']}
        try:
            return xgr.StructuralTag(**native)
        except ValidationError as e:
            raise ValueError(f'Invalid structural_tag payload: {e}') from e

    # lmdeploy single-tag shape: {begin, end, schema}.
    if 'begin' in payload and 'end' in payload:
        schema = payload.get('schema', {})
        try:
            return xgr.StructuralTag(format={
                'type': 'tag',
                'begin': payload['begin'],
                'end': payload['end'],
                'content': {'type': 'json_schema', 'json_schema': schema},
            })
        except ValidationError as e:
            raise ValueError(f'Invalid structural_tag payload: {e}') from e

    # lmdeploy multi-tag shape: {tags: [{begin, end, schema}, ...]}.
    if 'tags' in payload:
        tag_formats = []
        for item in payload['tags']:
            tag_formats.append({
                'type': 'tag',
                'begin': item['begin'],
                'end': item['end'],
                'content': {
                    'type': 'json_schema',
                    'json_schema': item.get('schema', {}),
                },
            })
        try:
            return xgr.StructuralTag(format={
                'type': 'triggered_tags',
                'triggers': [t['begin'] for t in tag_formats],
                'tags': tag_formats,
            })
        except ValidationError as e:
            raise ValueError(f'Invalid structural_tag payload: {e}') from e

    raise ValueError(f'Unrecognized structural_tag payload: {payload!r}')


def compile_structural_tag_payload(payload: Any) -> xgr.Grammar:
    """Build an uncompiled ``xgr.Grammar`` from a structural_tag payload.

    The returned ``Grammar`` is built via ``xgr.Grammar.from_structural_tag``.
    Engines compile it with their own ``GrammarCompiler``, e.g.::

        grammar = compile_structural_tag_payload(payload)
        compiled = compiler.compile_grammar(grammar)

    or equivalently the engines may call
    ``compiler.compile_structural_tag(_to_xgr_structural_tag(payload))``
    to skip the intermediate ``Grammar`` object.
    """
    return xgr.Grammar.from_structural_tag(_to_xgr_structural_tag(payload))


def _const_string_grammar(value: str) -> xgr.Grammar:
    """Build an uncompiled ``xgr.Grammar`` matching the literal string
    ``value``.

    Uses xgrammar's ``ConstStringFormat`` (via ``Grammar.from_structural_tag``)
    so the value is treated as opaque bytes — no EBNF escaping is needed and
    no character in ``value`` (``"`` , ``\\``, ``|`` …) can break out of the
    literal or inject grammar.
    """
    return xgr.Grammar.from_structural_tag(xgr.StructuralTag(
        format={'type': 'const_string', 'value': value}))


def compile_choice(options: list[str]) -> xgr.Grammar:
    """Build an alternation grammar matching exactly one of ``options``.

    ``options`` is a list of literal strings, e.g. ``["yes", "no"]``.  Each
    option is compiled to a const-string grammar and the results are combined
    with ``xgr.Grammar.union``, so option strings are treated as opaque
    literals — a ``"`` or ``\\`` in an option cannot break out of the literal
    (no EBNF injection / lexer crash).  The returned ``Grammar`` is uncompiled;
    engines compile it with their own ``GrammarCompiler``::

        compiled = compiler.compile_grammar(compile_choice(["yes", "no"]))
    """
    if not options:
        raise ValueError('compile_choice requires at least one option')
    return xgr.Grammar.union(*[_const_string_grammar(opt) for opt in options])
