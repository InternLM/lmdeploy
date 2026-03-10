# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/reasoning_parsers
from dataclasses import dataclass, field
from functools import cached_property
from typing import Sequence

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

    def update(self, delta_text: str, delta_token_ids: Sequence[int]) -> None:
        """Accumulate new delta into current_text / current_token_ids."""
        self.current_text += delta_text
        self.current_token_ids = self.current_token_ids + list(delta_token_ids)

    def step(self) -> None:
        """Advance: copy current -> previous (call at end of each iteration)."""
        self.previous_text = self.current_text
        self.previous_token_ids = list(self.current_token_ids)


def get_streaming_state(request: object) -> StreamingParserState:
    """Get or create a StreamingParserState on the request object."""
    state = getattr(request, '_streaming_parser_state', None)
    if state is None:
        state = StreamingParserState()
        setattr(request, '_streaming_parser_state', state)
    return state


class ReasoningParser:
    """Abstract base class for reasoning content parsers."""

    def __init__(self, tokenizer: object):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

    def extract_reasoning_content_streaming(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        request: object,
        **kwargs,
    ) -> DeltaMessage | None:
        """Extract reasoning content from an incomplete (streaming) response.

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
        raise NotImplementedError('ReasoningParser.extract_reasoning_content_streaming '
                                  'has not been implemented!')

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest, **kwargs) -> tuple[str, str]:
        """Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Args:
            model_output: The model-generated string to extract reasoning content from.
            request: The request object that was used to generate the model_output.

        Returns:
            A tuple of (reasoning_content, final_output). Either may be None.
        """
        raise NotImplementedError('ReasoningParser.extract_reasoning_content '
                                  'has not been implemented!')


class ThinkingReasoningParser(ReasoningParser):
    """Base class for reasoning parsers that use <think>...</think> style tags.

    Subclasses only need to set `start_token`, `end_token`, and optionally
    override `strip_newlines` and `on_missing_start_tag` to customize behavior.

    This parser uses a two-step detection strategy (inspired by vllm):
      1. First check token_ids (fast integer comparison) to determine whether
         the start/end tags are present.
      2. Only when confirmed, use str.find() to locate exact positions for
         slicing.
    If the tokenizer does not have single-token representations for the tags,
    it falls back to string-based detection automatically.
    """

    # Subclasses should set these
    start_token: str = '<think>'
    end_token: str = '</think>'

    # Whether to strip leading/trailing newlines from reasoning content
    # in non-streaming extraction.
    strip_newlines: bool = False

    # Behavior when end_token is not found in non-streaming extraction:
    #   'reasoning' -> treat entire output as reasoning (DeepSeek R1 behavior)
    #   'content'   -> treat entire output as content (QwQ/Qwen3 behavior)
    on_missing_end_tag: str = 'content'

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)

        if not self.model_tokenizer:
            raise ValueError('The model tokenizer must be passed to the '
                             'ReasoningParser constructor during construction.')

        # Try to resolve single token ids for fast detection.
        # If the tokenizer doesn't have them as single tokens, fall back to
        # string-based detection (token ids will be None).
        self.start_token_id: int = self.vocab.get(self.start_token)
        self.end_token_id: int = self.vocab.get(self.end_token)

    # ---- internal helpers for tag detection ----

    def _has_start(self, token_ids: Sequence[int], text: str) -> bool:
        """Check whether the start tag is present."""
        if self.start_token_id is not None:
            return self.start_token_id in token_ids
        return self.start_token in text

    def _has_end(self, token_ids: Sequence[int], text: str) -> bool:
        """Check whether the end tag is present."""
        if self.end_token_id is not None:
            return self.end_token_id in token_ids
        return self.end_token in text

    def _is_single_start_token(self, delta_token_ids: Sequence[int], delta_text: str) -> bool:
        """Check if the delta is exactly the start tag (single token)."""
        if self.start_token_id is not None:
            return len(delta_token_ids) == 1 and delta_token_ids[0] == self.start_token_id
        return delta_text == self.start_token

    def _is_single_end_token(self, delta_token_ids: Sequence[int], delta_text: str) -> bool:
        """Check if the delta is exactly the end tag (single token)."""
        if self.end_token_id is not None:
            return len(delta_token_ids) == 1 and delta_token_ids[0] == self.end_token_id
        return delta_text == self.end_token

    def _split_at_end_token(self, text: str) -> tuple[str, str]:
        """Split text at the end token, returning (before, after)."""
        idx = text.find(self.end_token)
        return text[:idx], text[idx + len(self.end_token):]

    # ---- public API ----

    def extract_reasoning_content_streaming(
        self,
        delta_text: str,
        delta_token_ids: Sequence[int],
        request: object,
        **kwargs,
    ) -> DeltaMessage | None:
        state = get_streaming_state(request)
        previous_text = state.previous_text
        previous_token_ids = state.previous_token_ids

        # Handle single special tokens
        if self._is_single_end_token(delta_token_ids, delta_text):
            return DeltaMessage(content='')
        if self._is_single_start_token(delta_token_ids, delta_text):
            return DeltaMessage(content='')

        # Check if start tag is in previous tokens
        if self._has_start(previous_token_ids, previous_text):
            if self._has_end(delta_token_ids, delta_text):
                # start in previous, end in delta -> split at end tag
                reasoning_content, content = self._split_at_end_token(delta_text)
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            elif self._has_end(previous_token_ids, previous_text):
                # start in previous, end in previous -> reasoning is done
                return DeltaMessage(content=delta_text)
            else:
                # start in previous, no end yet -> still reasoning
                return DeltaMessage(reasoning_content=delta_text)

        # Check if start tag is in delta
        if self._has_start(delta_token_ids, delta_text):
            if self._has_end(delta_token_ids, delta_text):
                # Both start and end in delta -> extract between them
                start_idx = delta_text.find(self.start_token)
                end_idx = delta_text.find(self.end_token)
                reasoning_content = delta_text[start_idx + len(self.start_token):end_idx]
                content = delta_text[end_idx + len(self.end_token):]
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            else:
                # start in delta, no end -> reasoning begins
                return DeltaMessage(reasoning_content=delta_text)

        # No start tag in previous or delta.
        # Still need to check for end tag (model may omit start tag).
        # Ref: https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self._has_end(delta_token_ids, delta_text):
            reasoning_content, content = self._split_at_end_token(delta_text)
            return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
        elif self._has_end(previous_token_ids, previous_text):
            # end in previous -> reasoning finished earlier
            return DeltaMessage(content=delta_text)
        else:
            # no end anywhere -> still in reasoning
            return DeltaMessage(reasoning_content=delta_text)

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest, **kwargs) -> tuple[str, str]:
        # If end tag is not present, behavior depends on on_missing_end_tag
        if self.end_token not in model_output:
            if self.on_missing_end_tag == 'reasoning':
                return model_output, None
            else:
                return None, model_output

        # Add start tag if missing (compatibility with models that omit it)
        if self.start_token not in model_output:
            model_output = f'{self.start_token}{model_output}'

        # Extract reasoning content using str.find() + slicing
        start_idx = model_output.find(self.start_token)
        end_idx = model_output.find(self.end_token)
        reasoning_content = model_output[start_idx + len(self.start_token):end_idx]
        final_output = model_output[end_idx + len(self.end_token):]

        if self.strip_newlines:
            reasoning_content = reasoning_content.strip('\n')

        return (
            reasoning_content if reasoning_content else None,
            final_output if final_output else None,
        )
