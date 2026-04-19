# Copyright (c) OpenMMLab. All rights reserved.
"""Adapters between Anthropic requests and LMDeploy internals."""

from __future__ import annotations

from typing import Any

from lmdeploy.messages import GenerationConfig

from .protocol import CountTokensRequest, MessagesRequest, TextContentBlockParam


def get_model_list(server_context) -> list[str]:
    """Return available model names from the server context."""

    model_names = [server_context.async_engine.model_name]
    cfg = server_context.async_engine.backend_config
    model_names += getattr(cfg, 'adapters', None) or []
    return model_names


def ensure_tools_not_requested(request: MessagesRequest | CountTokensRequest) -> None:
    """Reject tool-related fields while parser refactor is in progress."""

    if getattr(request, 'tools', None):
        raise NotImplementedError('Anthropic tool fields are temporarily unsupported.')
    if getattr(request, 'tool_choice', None) is not None:
        raise NotImplementedError('Anthropic tool_choice is temporarily unsupported.')


def _text_from_blocks(blocks: list[TextContentBlockParam | dict[str, Any]], field_name: str) -> str:
    out: list[str] = []
    for idx, block in enumerate(blocks):
        if isinstance(block, dict):
            block_type = block.get('type')
            text = block.get('text')
        else:
            block_type = block.type
            text = block.text
        if block_type != 'text':
            raise ValueError(
                f'Only text content blocks are supported in `{field_name}`. '
                f'Got: {block_type!r} at index {idx}.')
        if text is None:
            raise ValueError(f'Missing `text` in `{field_name}` content block at index {idx}.')
        out.append(text)
    return ''.join(out)


def text_from_content(content: str | list[TextContentBlockParam], field_name: str) -> str:
    """Normalize Anthropic content field to plain text."""

    if isinstance(content, str):
        return content
    return _text_from_blocks(content, field_name=field_name)


def to_lmdeploy_messages(request: MessagesRequest | CountTokensRequest) -> list[dict[str, str]]:
    """Convert Anthropic request messages into LMDeploy chat messages."""

    lm_messages: list[dict[str, str]] = []
    if request.system is not None:
        lm_messages.append(
            dict(role='system', content=text_from_content(request.system, field_name='system')))
    for idx, message in enumerate(request.messages):
        content = text_from_content(message.content, field_name=f'messages[{idx}].content')
        lm_messages.append(dict(role=message.role, content=content))
    return lm_messages


def to_generation_config(request: MessagesRequest) -> GenerationConfig:
    """Map Anthropic messages request to LMDeploy generation config."""

    return GenerationConfig(
        max_new_tokens=request.max_tokens,
        do_sample=True,
        top_k=40 if request.top_k is None else request.top_k,
        top_p=1.0 if request.top_p is None else request.top_p,
        temperature=1.0 if request.temperature is None else request.temperature,
        stop_words=request.stop_sequences,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )


def count_input_tokens(async_engine, messages: list[dict[str, str]]) -> int:
    """Approximate Anthropic token counting using LMDeploy
    tokenizer/template."""

    prompt = async_engine.chat_template.messages2prompt(messages, sequence_start=True)
    token_ids = async_engine.tokenizer.encode(prompt, add_bos=True)
    return len(token_ids)


def map_finish_reason(reason: str | None) -> str:
    """Map LMDeploy/OpenAI-like finish reason to Anthropic stop reason."""

    mapping = {
        'stop': 'end_turn',
        'length': 'max_tokens',
        'tool_calls': 'stop_sequence',
        'abort': 'stop_sequence',
        'error': 'stop_sequence',
    }
    return mapping.get(reason, 'end_turn')
