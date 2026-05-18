# Copyright (c) OpenMMLab. All rights reserved.
"""Adapters between Anthropic requests and LMDeploy internals."""

from __future__ import annotations

import json
from typing import Any

import shortuuid

from lmdeploy.messages import GenerationConfig
from lmdeploy.serve.openai.protocol import Tool, ToolChoice, ToolChoiceFuncName

from .protocol import (
    ContentBlockParam,
    CountTokensRequest,
    MessagesRequest,
    MessageTextBlock,
    MessageThinkingBlock,
    MessageToolUseBlock,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolParam,
)


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


def to_openai_tools(tools: list[ToolParam] | None) -> list[Tool] | None:
    """Convert Anthropic tools into OpenAI protocol tool entries."""

    if not tools:
        return None
    return [
        Tool(
            type='function',
            function=dict(
                name=tool.name,
                description=tool.description,
                parameters=tool.input_schema,
            ),
        ) for tool in tools
    ]


def normalize_tool_choice(tool_choice: ToolChoiceParam | str | None) -> ToolChoice | str:
    """Map Anthropic tool choice values to OpenAI-compatible values."""

    if tool_choice is None:
        return 'auto'
    if isinstance(tool_choice, str):
        if tool_choice == 'any':
            return 'required'
        return tool_choice
    if isinstance(tool_choice, ToolChoiceToolParam):
        return ToolChoice(function=ToolChoiceFuncName(name=tool_choice.name))
    if tool_choice.type == 'any':
        return 'required'
    return 'auto'


def _safe_parse_tool_input(arguments: str | None) -> dict[str, Any]:
    if not arguments:
        return {}
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return dict(raw_arguments=arguments)
    if isinstance(parsed, dict):
        return parsed
    return dict(value=parsed)


def build_message_content_blocks(
    text: str | None,
    tool_calls: list[Any] | None,
    reasoning_content: str | None,
) -> list[MessageTextBlock | MessageThinkingBlock | MessageToolUseBlock]:
    """Build Anthropic message content blocks from parser outputs."""

    blocks: list[MessageTextBlock | MessageThinkingBlock | MessageToolUseBlock] = []
    if reasoning_content:
        blocks.append(MessageThinkingBlock(thinking=reasoning_content))
    if text:
        blocks.append(MessageTextBlock(text=text))
    if tool_calls:
        for tool_call in tool_calls:
            if getattr(tool_call, 'type', None) != 'function' or not getattr(tool_call, 'function', None):
                continue
            blocks.append(
                MessageToolUseBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=_safe_parse_tool_input(tool_call.function.arguments),
                ))
    return blocks


def _stringify_block_value(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if hasattr(value, 'model_dump'):
        value = value.model_dump()
    return json.dumps(value, ensure_ascii=False)


def _text_from_block_content(content: Any, field_name: str) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _text_from_blocks(content, field_name=field_name)
    return _stringify_block_value(content)


def _text_from_blocks(blocks: list[ContentBlockParam | dict[str, Any]], field_name: str) -> str:
    out: list[str] = []
    for idx, block in enumerate(blocks):
        if isinstance(block, dict):
            block_type = block.get('type')
            text = block.get('text')
            content = block.get('content')
            tool_use_id = block.get('tool_use_id')
            tool_name = block.get('name')
            tool_input = block.get('input')
            thinking = block.get('thinking')
        else:
            block_type = block.type
            text = block.text
            content = block.content
            tool_use_id = block.tool_use_id
            tool_name = block.name
            tool_input = block.input
            thinking = block.thinking
        if block_type == 'text':
            if text is None:
                raise ValueError(f'Missing `text` in `{field_name}` content block at index {idx}.')
            out.append(text)
        elif block_type == 'tool_result':
            result_text = _text_from_block_content(content, field_name=f'{field_name}[{idx}].content')
            out.append(f'\n[tool_result id={tool_use_id or ""}]\n{result_text}\n[/tool_result]\n')
        elif block_type == 'tool_use':
            tool_payload = _stringify_block_value(tool_input)
            out.append(f'\n[tool_use name={tool_name or ""}]\n{tool_payload}\n[/tool_use]\n')
        elif block_type in ('thinking', 'redacted_thinking'):
            if thinking:
                out.append(f'\n[thinking]\n{thinking}\n[/thinking]\n')
        else:
            out.append(f'\n[{block_type}]\n{_stringify_block_value(block)}\n[/{block_type}]\n')
    return ''.join(out)


def text_from_content(content: str | list[ContentBlockParam], field_name: str) -> str:
    """Normalize Anthropic content field to plain text."""

    if isinstance(content, str):
        return content
    return _text_from_blocks(content, field_name=field_name)


def _block_get(block: ContentBlockParam | dict[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _convert_image_source_to_url(source: Any) -> str:
    source_type = _block_get(source, 'type')
    if source_type == 'url':
        return _block_get(source, 'url', '')
    if source_type == 'base64':
        media_type = _block_get(source, 'media_type', 'image/jpeg')
        data = _block_get(source, 'data', '')
        if data:
            return f'data:{media_type};base64,{data}'
    return ''


def _convert_system_blocks_to_text(system: list[ContentBlockParam]) -> str:
    system_prompt = ''
    for block in system:
        if _block_get(block, 'type') != 'text':
            continue
        text = _block_get(block, 'text')
        if not text or text.startswith('x-anthropic-billing-header'):
            continue
        system_prompt += text
    return system_prompt


def _convert_user_tool_result(block: ContentBlockParam | dict[str, Any]) -> list[dict[str, Any]]:
    tool_text = ''
    tool_image_urls: list[str] = []
    content = _block_get(block, 'content')

    if isinstance(content, str):
        tool_text = content
    elif isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if _block_get(item, 'type') == 'text':
                text_parts.append(_block_get(item, 'text', ''))
            elif _block_get(item, 'type') == 'image':
                url = _convert_image_source_to_url(_block_get(item, 'source', {}))
                if url:
                    tool_image_urls.append(url)
        tool_text = '\n'.join(text_parts)

    messages = [
        dict(
            role='tool',
            tool_call_id=_block_get(block, 'tool_use_id') or _block_get(block, 'id') or '',
            content=tool_text or '',
        )
    ]
    if tool_image_urls:
        messages.append(
            dict(
                role='user',
                content=[dict(type='image_url', image_url=dict(url=url)) for url in tool_image_urls],
            ))
    return messages


def to_openai_messages(request: MessagesRequest | CountTokensRequest) -> list[dict[str, Any]]:
    """Convert Anthropic request messages into OpenAI-compatible message
    dicts."""

    openai_messages: list[dict[str, Any]] = []
    if request.system is not None:
        if isinstance(request.system, str):
            openai_messages.append(dict(role='system', content=request.system))
        else:
            openai_messages.append(dict(role='system', content=_convert_system_blocks_to_text(request.system)))

    for idx, message in enumerate(request.messages):
        if isinstance(message.content, str):
            openai_messages.append(dict(role=message.role, content=message.content))
            continue

        openai_message: dict[str, Any] = dict(role=message.role)
        content_parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        for block_idx, block in enumerate(message.content):
            block_type = _block_get(block, 'type')
            if block_type == 'text':
                text = _block_get(block, 'text')
                if text:
                    content_parts.append(dict(type='text', text=text))
                continue

            if block_type == 'image':
                source = _block_get(block, 'source')
                if source:
                    url = _convert_image_source_to_url(source)
                    if url:
                        content_parts.append(
                            dict(type='image_url', image_url=dict(url=url)))
                continue

            if block_type == 'tool_use':
                tool_calls.append(
                    dict(
                        id=_block_get(block, 'id') or f'call_{shortuuid.random()}',
                        type='function',
                        function=dict(
                            name=_block_get(block, 'name') or '',
                            arguments=json.dumps(_block_get(block, 'input') or {}),
                        ),
                    ))
                continue

            if block_type == 'tool_result':
                if message.role == 'user':
                    openai_messages.extend(_convert_user_tool_result(block))
                else:
                    result_text = _text_from_block_content(
                        _block_get(block, 'content'),
                        field_name=f'messages[{idx}].content[{block_idx}].content',
                    )
                    content_parts.append(dict(type='text', text=f'Tool result: {result_text}'))
                continue

            if block_type == 'thinking':
                thinking = _block_get(block, 'thinking')
                if thinking is not None:
                    reasoning_parts.append(thinking)
                continue

            if block_type == 'redacted_thinking':
                continue

            content_parts.append(dict(type='text', text=_stringify_block_value(block)))

        if reasoning_parts:
            openai_message['reasoning_content'] = ''.join(reasoning_parts)
        if tool_calls:
            openai_message['tool_calls'] = tool_calls
        if content_parts:
            if len(content_parts) == 1 and content_parts[0]['type'] == 'text':
                openai_message['content'] = content_parts[0]['text']
            else:
                openai_message['content'] = content_parts

        if ('content' in openai_message or 'tool_calls' in openai_message
                or 'reasoning_content' in openai_message):
            openai_messages.append(openai_message)
    return openai_messages


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
        'tool_calls': 'tool_use',
        'abort': 'stop_sequence',
        'error': 'stop_sequence',
    }
    return mapping.get(reason, 'end_turn')
