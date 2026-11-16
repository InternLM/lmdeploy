# Copyright (c) OpenMMLab. All rights reserved.
"""Adapt XGrammar's GLM XML schema to the parser's token-ID boundaries."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import Tool, ToolChoice


# XGrammar emits each XML parameter as one sequence. Its value is a separate
# expression/rule: do not rewrite that expression, since const/enum/pattern
# values may themselves contain the same tag spellings.
_PARAMETER = re.compile(
    r'("<arg_key>(?:\\.|[^"\\])*")([^|\n]*?)"</arg_key>"'
    r'([^|\n]*?)"<arg_value>"([^|\n]*?)"</arg_value>"')
_RULE = re.compile(r'^([\w]+) ::= ', re.MULTILINE)
_LEXEME = re.compile(r'"(?:\\.|[^"\\])*"|\[(?:\\.|[^\]\\])*\]|([A-Za-z_]\w*)')


def _tokenize_xml_schema(schema: dict | bool, ids: dict[str, int]):
    """Keep XGrammar's schema rules, replacing only XML framing terminals.

    JSONSchemaFormat's glm_xml style currently emits byte-string delimiters. Lower it through the public EBNF interface
    and walk the object productions, stopping at parameter values so their strings and nested JSON stay data.
    """
    import xgrammar as xgr
    from xgrammar.structural_tag import GrammarFormat, JSONSchemaFormat, StructuralTag

    try:
        ebnf = str(xgr.Grammar.from_structural_tag(
            StructuralTag(format=JSONSchemaFormat(json_schema=schema, style='glm_xml'))))
    except RuntimeError as err:
        raise ValueError(f'Unsupported named tool parameter schema: {err}') from err
    parts = _RULE.split(ebnf)
    rules = dict(zip(parts[1::2], parts[2::2]))
    pending, visited = ['root'], set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        body = rules[name]
        # Values are intentionally omitted from this traversal.
        object_body = _PARAMETER.sub(lambda m: m[1] + m[2] + m[3], body)
        pending.extend(m[1] for m in _LEXEME.finditer(object_body) if m[1] in rules)

        def replace_parameter(match):
            key = match[1][len('"<arg_key>'):]
            return (f'Token({ids["<arg_key>"]}) "{key}{match[2]}'
                    f'Token({ids["</arg_key>"]}){match[3]}'
                    f'Token({ids["<arg_value>"]}){match[4]}'
                    f'Token({ids["</arg_value>"]})')

        rules[name] = _PARAMETER.sub(replace_parameter, body)

    # Free XML string values end at the dedicated ID, not at an ordinary-token
    # spelling of </arg_value>. Other reserved IDs are never argument text.
    excluded = ', '.join(str(token_id) for token_id in ids.values())
    if 'xml_string' in rules:
        rules['xml_string'] = f'(ExcludeToken({excluded}))*\n'
    return GrammarFormat(grammar=''.join(f'{name} ::= {body}' for name, body in rules.items()))


def build_named_tool_format(tools: list[Tool], choice: ToolChoice, *, reasoning: bool,
                            tokenizer: PreTrainedTokenizerBase) -> dict:
    """Require one named GLM call with token-delimited, schema-guided XML."""
    from .glm47_tool_parser import Glm47ToolParser

    functions = [tool.function for tool in tools if tool.function.name == choice.function.name]
    if len(functions) != 1:
        raise ValueError('Named tool_choice must match exactly one declared tool.')
    spellings = (*Glm47ToolParser.token_id_terminals, '<think>', '</think>')
    vocab = tokenizer.get_added_vocab()
    needed = spellings if reasoning else Glm47ToolParser.token_id_terminals
    if any(tag not in vocab for tag in needed):
        raise ValueError('GLM named tools require dedicated control token IDs')
    token_ids = tuple((tag, vocab[tag]) for tag in spellings if tag in vocab)
    function = functions[0]
    schema = function.parameters if function.parameters is not None else True
    # Property order is significant to XGrammar's default XML schema mode.
    # Snapshot the schema instead of retaining mutable request objects.
    schema_json = json.dumps(schema, ensure_ascii=False, separators=(',', ':'))
    serialized = _cached_named_tool_format(function.name, schema_json, reasoning, token_ids)
    # Each request owns its format, including nested lists and dictionaries.
    return json.loads(serialized)


@lru_cache(maxsize=128)
def _cached_named_tool_format(name: str, schema_json: str, reasoning: bool,
                              token_ids: tuple[tuple[str, int], ...]) -> str:
    """Cache schema lowering and format construction as immutable JSON."""
    from xgrammar.structural_tag import (
        AnyTokensFormat,
        ConstStringFormat,
        SequenceFormat,
        StructuralTag,
        TokenFormat,
    )

    ids = dict(token_ids)
    schema = json.loads(schema_json)
    elements = []
    if reasoning:
        elements.extend([AnyTokensFormat(exclude_tokens=list(ids.values())), TokenFormat(token=ids['</think>'])])
    elements.extend([
        TokenFormat(token=ids['<tool_call>']), ConstStringFormat(value=name),
        _tokenize_xml_schema(schema, ids), TokenFormat(token=ids['</tool_call>']),
    ])
    result = StructuralTag(format=SequenceFormat(elements=elements)).model_dump(mode='json')
    # Reserve these IDs in the compiler too: constrained value rules use text
    # matching, which must never consume the same IDs as XML boundaries.
    result['_token_id_only'] = list(ids.values())
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))
