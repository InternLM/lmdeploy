# Copyright (c) OpenMMLab. All rights reserved.
"""Token-aware argument-name constraints for GLM automatic tool calls.

Constrain declared top-level names for plain object schemas. As a tool-calling
policy, omitted ``additionalProperties`` is treated as false here, unlike its
JSON Schema default. Explicitly open or composed schemas stay open. This is
not full JSON Schema validation: required fields, duplicates and values retain
their existing behavior. Auto choice still permits reasoning, ordinary text,
zero calls and multiple calls.

Dispatch on control IDs, not their spellings: an ordinary-token literal such
as ``</arg_value>`` in a shell command must remain ordinary argument text.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from lmdeploy.serve.openai.protocol import Tool


def _closed_properties(schema: dict[str, Any] | None) -> tuple[str, ...] | None:
    if not isinstance(schema, dict):
        return None
    properties = schema.get('properties')
    if (not isinstance(properties, dict) or not properties
            or schema.get('additionalProperties', False) is not False
            or any(key in schema for key in ('$ref', 'allOf', 'anyOf', 'oneOf', 'patternProperties'))):
        return None
    return tuple(properties)


def build_auto_tool_format(
    tools: list[Tool], tokenizer: PreTrainedTokenizerBase | None,
) -> dict[str, Any] | None:
    """Build a token-ID grammar from each tool's declared top-level
    properties."""
    if tokenizer is None or not tools:
        return None
    functions = [tool.function for tool in tools]
    properties = [_closed_properties(function.parameters) for function in functions]
    if not any(keys is not None for keys in properties):
        return None

    from xgrammar.structural_tag import (
        AnyTokensFormat,
        ConstStringFormat,
        Format,
        OrFormat,
        RegexFormat,
        SequenceFormat,
        StarFormat,
        StructuralTag,
        TokenDispatchFormat,
        TokenFormat,
    )

    spellings = ('<tool_call>', '</tool_call>', '<arg_key>', '</arg_key>',
                 '<arg_value>', '</arg_value>')
    vocab = tokenizer.get_added_vocab()
    if not all(spelling in vocab for spelling in spellings):
        raise ValueError('GLM auto tool constraints require dedicated control token IDs')
    ids = {spelling: vocab[spelling] for spelling in spellings}
    boundaries = list(ids.values())

    def token(spelling: str) -> TokenFormat:
        return TokenFormat(token=ids[spelling])

    def either(elements: list[Format]) -> Format:
        return elements[0] if len(elements) == 1 else OrFormat(elements=elements)

    bodies = []
    for function, keys in zip(functions, properties):
        key = (either([ConstStringFormat(value=name) for name in keys])
               if keys is not None else AnyTokensFormat(exclude_tokens=boundaries))
        pair = SequenceFormat(elements=[
            token('<arg_key>'), RegexFormat(pattern=r'\s*'), key,
            RegexFormat(pattern=r'\s*'), token('</arg_key>'),
            RegexFormat(pattern=r'\s*'), token('<arg_value>'),
            AnyTokensFormat(exclude_tokens=boundaries), token('</arg_value>'),
            RegexFormat(pattern=r'\s*'),
        ])
        bodies.append(SequenceFormat(elements=[
            RegexFormat(pattern=r'\s*'), ConstStringFormat(value=function.name),
            RegexFormat(pattern=r'\s*'), StarFormat(content=pair), token('</tool_call>'),
        ]))
    return StructuralTag(format=TokenDispatchFormat(
        rules=[(ids['<tool_call>'], either(bodies))], loop=True,
    )).model_dump(mode='json')
