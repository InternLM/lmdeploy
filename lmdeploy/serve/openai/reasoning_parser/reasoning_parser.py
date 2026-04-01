# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/reasoning_parsers
from functools import cached_property

from mmengine import Registry

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

    def get_reasoning_open_tag(self) -> str | None:
        """Return reasoning opening tag string, or None if no opening tag."""
        raise NotImplementedError('ReasoningParser.get_reasoning_open_tag has not been implemented!')

    def get_reasoning_close_tag(self) -> str | None:
        """Return reasoning closing tag string, or None if no closing tag."""
        raise NotImplementedError('ReasoningParser.get_reasoning_close_tag has not been implemented!')

    def starts_in_reasoning_mode(self) -> bool:
        """Whether streaming should begin in reasoning mode."""
        raise NotImplementedError('ReasoningParser.starts_in_reasoning_mode has not been implemented!')


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

    def get_reasoning_open_tag(self) -> str | None:
        return self.start_token

    def get_reasoning_close_tag(self) -> str | None:
        return self.end_token

    def starts_in_reasoning_mode(self) -> bool:
        return True
