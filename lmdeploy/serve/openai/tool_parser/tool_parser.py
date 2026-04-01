# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/tool_parsers
from collections.abc import Sequence
from functools import cached_property

from mmengine import Registry

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage, ExtractedToolCallInformation
from lmdeploy.serve.openai.response_parser import StreamBuffer
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')
ToolParserManager = Registry('tool_parser', locations=['lmdeploy.serve.openai.tool_parser'])


class ToolParser:
    """Abstract ToolParser class that should not be used directly.

    Provided properties and methods should be used in derived classes.
    """

    def __init__(self, tokenizer: object):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def adjust_request(self, request: ChatCompletionRequest) -> ChatCompletionRequest:
        """Static method that used to adjust the request parameters."""
        if request.tools is not None and request.tool_choice != 'none':
            if not isinstance(request.tool_choice, str):
                request.tools = [
                    item.function.model_dump() for item in request.tools
                    if item.function.name == request.tool_choice.function.name
                ]
            else:
                request.tools = [item.function.model_dump() for item in request.tools]
        return request

    def extract_tool_calls(self, model_output: str, request: ChatCompletionRequest) -> ExtractedToolCallInformation:
        """Static method that should be implemented for extracting tool calls
        from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response available before sending to the client.
        Static because it's stateless.
        """
        raise NotImplementedError('AbstractToolParser.extract_tool_calls has not been implemented!')

    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        """Instance method that should be implemented for extracting tool calls
        from an incomplete response; for use when handling tool calls and
        streaming.

        Args:
            delta_text: The new text chunk for this iteration.
            delta_token_ids: The new token ids for this chunk.
            request: The chat completion request.
            stream_buffer: Cumulative decoding state (``ResponseParser`` or a test
                double); use ``stream_buffer.current_text`` for the full partial output.
                Tool-specific
                fields live on the parser instance (one instance per request).

        Instance method because streaming uses the shared buffer plus parser-local state.
        """
        raise NotImplementedError('AbstractToolParser.extract_tool_calls_streaming has not been '
                                  'implemented!')

    def detect_tool_start_tag(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        *,
        stream_buffer: StreamBuffer,
        request: ChatCompletionRequest,
    ) -> int | None:
        """Optional hint for where tool-call protocol starts in *delta_text*.

        Default implementation returns None, meaning "no tool start detected in this chunk". Concrete parsers can
        override this to let ResponseParser know where to split reasoning vs tool content without hard-coding any
        protocol details here.
        """
        return None

    def get_tool_open_tag(self) -> str | None:
        """Return tool opening tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_open_tag has not been implemented!')

    def get_tool_close_tag(self) -> str | None:
        """Return tool closing tag string, or None if unsupported."""
        raise NotImplementedError('ToolParser.get_tool_close_tag has not been implemented!')

    def get_tool_payload_format(self) -> str:
        """Return payload format for tool call body."""
        raise NotImplementedError('ToolParser.get_tool_payload_format has not been implemented!')
