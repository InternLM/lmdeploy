# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/reasoning_parsers
from functools import cached_property

from mmengine import Registry

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage
from lmdeploy.serve.openai.response_parser import StreamBuffer

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.openai.reasoning_parser'])

StreamingParserState = StreamBuffer


class ReasoningParser:
    """Abstract base class for reasoning content parsers."""

    def __init__(self, tokenizer: object, **kwargs):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def extract_reasoning_streaming(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: object,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        """Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming.

        Args:
            delta_text: The new text chunk (may have been modified by the tool
                parser before being passed here).
            delta_token_ids: The new token ids for this chunk.
            request: The request object.
            stream_buffer: Cumulative decoding state (``ResponseParser.stream``);
                Token ids from prior chunks are in ``stream_buffer.previous_token_ids``
                at the time this method runs (after ``stream_buffer.update`` for this chunk).

        Returns a DeltaMessage with reasoning_content and/or content fields,
        or None if the delta should be skipped.
        """
        raise NotImplementedError('ReasoningParser.extract_reasoning_streaming '
                                  'has not been implemented!')

    def extract_reasoning(self, model_output: str, request: 'ChatCompletionRequest',
                                  **kwargs) -> tuple[str | None, str | None]:
        """Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Args:
            model_output: The model-generated string to extract reasoning content from.
            request: The request object that was used to generate the model_output.

        Returns:
            A tuple of (reasoning_content, final_output). Either may be None.
        """
        raise NotImplementedError('ReasoningParser.extract_reasoning '
                                  'has not been implemented!')


class ThinkingReasoningParser(ReasoningParser):
    """Base class for reasoning parsers that use <think>...</think> style tags.

    Subclasses only need to set `start_token`, `end_token`.

    This parser uses a two-step detection strategy (inspired by vllm):
      1. First check token_ids (fast integer comparison) to determine whether
         the start/end tags are present.
      2. Only when confirmed, use str.find() to locate exact positions for
         slicing.
    If the tokenizer does not have single-token representations for the tags,
    it falls back to string-based detection automatically.
    """

    start_token: str = '<think>'
    end_token: str = '</think>'


    def __init__(self, tokenizer: object, **kwargs):
        super().__init__(tokenizer, **kwargs)

        # Try to resolve single token ids for fast detection.
        # If the tokenizer doesn't have them as single tokens, fall back to
        # string-based detection (token ids will be None).
        self.start_token_id: int = self.vocab.get(self.start_token)
        self.end_token_id: int = self.vocab.get(self.end_token)

    def extract_reasoning_streaming(
        self,
        delta_text: str,
        delta_token_ids: list[int],
        request: object,
        *,
        stream_buffer: StreamBuffer,
        **kwargs,
    ) -> DeltaMessage | None:
        """Extract reasoning content from a streaming model-generated string.

        Args:
            delta_text: The new text chunk (may have been modified by the tool
                parser before being passed here).
            delta_token_ids: The new token ids for this chunk.
            request: The request object.
            stream_buffer: Cumulative decoding state (see base class).

        Returns a DeltaMessage with reasoning_content and/or content fields,
        or None if the delta should be skipped.
        """
        previous_token_ids = stream_buffer.previous_token_ids

        # Handle single special tokens
        if len(delta_token_ids) == 1 and (delta_token_ids[0] in [self.start_token_id, self.end_token_id]):
            return None

        # Check if start tag is in previous tokens
        if self.start_token_id in previous_token_ids:
            if self.end_token_id in delta_token_ids:
                # Both start and end in delta -> extract between them
                end_idx = delta_text.find(self.end_token)
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.end_token):]
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            elif self.end_token_id in previous_token_ids:
                # end in previous, no start -> reasoning is done
                return DeltaMessage(content=delta_text)
            else:
                # start in previous, no end -> reasoning continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.start_token_id in delta_token_ids:
            start_index = delta_text.find(self.start_token)
            if self.end_token_id in delta_token_ids:
                # Both start and end in delta -> extract between them
                end_index = delta_text.find(self.end_token)
                reasoning_content = delta_text[start_index + len(self.start_token) : end_index]
                content = delta_text[end_index + len(self.end_token) :]
                return DeltaMessage(
                    reasoning_content=reasoning_content, content=content if content else None
                )
            else:
                # start token in delta, no end token in delta, reasoning content continues
                return DeltaMessage(reasoning_content=delta_text[start_index + len(self.start_token):])
        else:
            # not find thinking start token
            return DeltaMessage(content=delta_text)

    def extract_reasoning(self, model_output: str, request: 'ChatCompletionRequest', **kwargs) -> tuple[str, str]:
        """Extract reasoning content from a complete model-generated string.

        Args:
            model_output: The model-generated string to extract reasoning content from.
            request: The request object that was used to generate the model_output.

        Returns:
            A tuple of (reasoning_content, final_output). Either may be None.
        """

        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output

        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        # For models that may not generate start token,
        # assume the reasoning content is always at the start.
        if self.end_token not in model_output:
            return model_output, None
        else:
            reasoning, _, content = model_output.partition(self.end_token)
            # If generation stops right after end-of-think, return None content
            final_content = content or None
            # If the model_output is like "<think></think>...", return None reasoning
            reasoning = reasoning or None
            return reasoning, final_content
