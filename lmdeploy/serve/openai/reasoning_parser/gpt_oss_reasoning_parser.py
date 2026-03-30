# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/vllm-project/vllm/blob/v0.10.2rc1/vllm/entrypoints/harmony_utils.py
from __future__ import annotations

import shortuuid
from openai_harmony import HarmonyEncodingName, Role, StreamableParser, load_harmony_encoding

from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    FunctionCall,
    ToolCall,
)
from lmdeploy.serve.openai.response_parser import StreamBuffer

from .reasoning_parser import ReasoningParser, ReasoningParserManager

_harmony_encoding = None


def get_encoding():
    global _harmony_encoding
    if _harmony_encoding is None:
        _harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _harmony_encoding


def get_streamable_parser_for_assistant() -> StreamableParser:
    return StreamableParser(get_encoding(), role=Role.ASSISTANT)


class GptOssChatParser:
    """Harmony stream parser for GPT-OSS (assistant role): content, reasoning,
    tool calls."""

    def __init__(self):
        self.parser = get_streamable_parser_for_assistant()

    def parse_streaming(self, tokens: list[int]) -> DeltaMessage:
        parser = self.parser
        delta_message = DeltaMessage(role='assistant')
        content = ''
        reasoning_content = ''
        tool_calls = []
        delta_tool_call = None
        for token in tokens:
            prev_recipient = parser.current_recipient
            parser.process(token)
            cur_channel = parser.current_channel
            cur_recipient = parser.current_recipient
            delta_text = parser.last_content_delta or ''
            if cur_channel == 'final':
                content += delta_text
            elif cur_channel == 'analysis':
                reasoning_content += delta_text
            elif cur_channel == 'commentary' and cur_recipient and cur_recipient.startswith('functions.'):
                base_index = 0
                for msg in parser.messages:
                    if msg.channel == 'commentary' and msg.recipient and msg.recipient.startswith('functions.'):
                        base_index += 1
                if prev_recipient != cur_recipient:
                    if delta_tool_call is not None:
                        tool_calls.append(delta_tool_call)
                    tool_name = cur_recipient.split('functions.', 1)[1]
                    delta_tool_call = DeltaToolCall(id=f'chatcmpl-tool-{shortuuid.random()}',
                                                    type='function',
                                                    index=base_index,
                                                    function=DeltaFunctionCall(name=tool_name, arguments=''))
                elif delta_text:
                    # Continuing the same tool call. Ensure we don't duplicate the
                    # very first delta string in this chunk. Previously we initialized
                    # with arguments=delta_text and then appended again, causing
                    # duplicated content like "locationlocation".
                    if delta_tool_call is None:
                        # We are in the middle of a tool call carried over from the
                        # previous chunk. Initialize an empty arguments buffer.
                        delta_tool_call = DeltaToolCall(index=base_index, function=DeltaFunctionCall(arguments=''))
                    delta_tool_call.function.arguments += delta_text

        if delta_tool_call:
            tool_calls.append(delta_tool_call)

        delta_message.content = content if content else None
        delta_message.reasoning_content = reasoning_content if reasoning_content else None
        delta_message.tool_calls = tool_calls
        return delta_message

    def parse_full(self, tokens: list[int]) -> ChatMessage:
        delta_message = self.parse_streaming(tokens)
        tool_calls = []
        for delta_tool_call in delta_message.tool_calls:
            function = FunctionCall(**delta_tool_call.function.model_dump())
            tool_calls.append(ToolCall(id=delta_tool_call.id, type=delta_tool_call.type, function=function))
        chat_message = ChatMessage(role='assistant',
                                   content=delta_message.content,
                                   tool_calls=tool_calls,
                                   reasoning_content=delta_message.reasoning_content)
        return chat_message


@ReasoningParserManager.register_module('gpt-oss')
class GptOssReasoningParser(ReasoningParser):
    """Reasoning / channel parser for OpenAI Harmony GPT-OSS wire format (token
    stream).

    Use ``--reasoning-parser gpt-oss`` when serving GPT-OSS models. When the engine
    architecture is ``GptOssForCausalLM``, the API server also enables this parser
    automatically even if the flag is omitted.
    """

    def __init__(self, tokenizer: object, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self._chat = GptOssChatParser()

    def parse_streaming(self, tokens: list[int]) -> DeltaMessage:
        """Parse one engine chunk of token ids into a
        :class:`~lmdeploy.serve.openai.protocol.DeltaMessage`."""
        return self._chat.parse_streaming(tokens)

    def parse_full(self, tokens: list[int]) -> ChatMessage:
        """Parse the full completion token sequence into a
        :class:`~lmdeploy.serve.openai.protocol.ChatMessage`."""
        return self._chat.parse_full(tokens)

    def extract_reasoning_streaming(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: object,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ):
        """Not used; GPT-OSS uses :meth:`parse_streaming` on token ids in the
        API server."""
        return None

    def extract_reasoning(self, model_output: str, request:
        ChatCompletionRequest, **kwargs) -> tuple[str | None, str | None]:
        """Not used for Harmony decoding; non-streaming path uses
        :meth:`parse_full` on token ids."""
        return None, model_output
