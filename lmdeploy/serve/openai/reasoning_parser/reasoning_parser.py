# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/reasoning_parsers
from dataclasses import dataclass, field
from functools import cached_property

from mmengine import Registry

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.openai.reasoning_parser'])


@dataclass
class StreamingParserState:
    """Shared state for streaming parsing, attached to a request object.

    Both reasoning parsers and tool parsers read/write the same state so that text accumulated by the streaming loop is
    available to all parsers without duplication.
    """
    previous_text: str = ''
    current_text: str = ''
    previous_token_ids: list[int] = field(default_factory=list)
    current_token_ids: list[int] = field(default_factory=list)

    def update(self, delta_text: str, delta_token_ids: list[int]) -> None:
        """Accumulate new delta into current_text / current_token_ids."""
        self.current_text += delta_text
        self.current_token_ids.extend(delta_token_ids)

    def step(self) -> None:
        """Advance: copy current -> previous (call at end of each iteration)."""
        self.previous_text = self.current_text
        self.previous_token_ids = self.current_token_ids


def get_streaming_state(request: object) -> StreamingParserState:
    """Get or create a StreamingParserState on the request object."""
    state = getattr(request, '_streaming_parser_state', None)
    if state is None:
        state = StreamingParserState()
        setattr(request, '_streaming_parser_state', state)
    return state


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
        **kwargs,
    ) -> DeltaMessage | None:
        """Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming.

        Args:
            delta_text: The new text chunk (may have been modified by the tool
                parser before being passed here).
            delta_token_ids: The new token ids for this chunk.
            request: The request object; a ``StreamingParserState`` is attached
                to it via ``get_streaming_state(request)`` so that previous /
                current text and token ids are available.

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
        **kwargs,
    ) -> DeltaMessage | None:
        """Extract reasoning content from a streaming model-generated string.

        Args:
            delta_text: The new text chunk (may have been modified by the tool
                parser before being passed here).
            delta_token_ids: The new token ids for this chunk.
            request: The request object; a ``StreamingParserState`` is attached
                to it via ``get_streaming_state(request)`` so that previous /
                current text and token ids are available.

        Returns a DeltaMessage with reasoning_content and/or content fields,
        or None if the delta should be skipped.
        """
        state = get_streaming_state(request)
        previous_token_ids = state.previous_token_ids

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
        # Check if the start token is present in the model output, remove it
        # if it is present.
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
            # If generation stops right after end-of-think, return null content
            final_content = content or None
            return reasoning, final_content
