# Copyright (c) OpenMMLab. All rights reserved.
import re
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name=['qwen-qwq', 'intern-s1'])
class QwenQwQReasoningParser(ReasoningParser):
    """Reasoning parser for Qwen QwQ model.

    The Qwen QwQ model uses <think>...</think> tokens to denote reasoning text. This parser extracts the reasoning
    content from the model output.
    """

    def __init__(self, tokenizer: object):
        super().__init__(tokenizer)
        self.think_start_token = '<think>'
        self.think_end_token = '</think>'

        self.reasoning_regex = re.compile(rf'{self.think_start_token}(.*?){self.think_end_token}', re.DOTALL)

        if not self.model_tokenizer:
            raise ValueError('The model tokenizer must be passed to the ReasoningParser '
                             'constructor during construction.')

    def extract_reasoning_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        **kwargs,
    ) -> Union[DeltaMessage, None]:
        """Instance method that should be implemented for extracting reasoning
        from an incomplete response; for use when handling reasoning calls and
        streaming.

        Has to be an instance method because  it requires state - the current tokens/diffs, but also the information
        about what has previously been parsed and extracted (see constructor)
        """
        # Skip single special tokens
        if delta_text == self.think_end_token or delta_text == self.think_start_token:
            return DeltaMessage(content='')

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.think_start_token in previous_text:
            if self.think_end_token in delta_text:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            elif self.think_end_token in previous_text:
                # <think> in previous, </think> in previous,
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.think_start_token in delta_text:
            if self.think_end_token in delta_text:
                # <think> in delta, </think> in delta, extract reasoning content
                start_index = delta_text.find(self.think_start_token)
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[start_index + len(self.think_start_token):end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            else:
                # <think> in delta, no </think> in delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        else:
            # no <think> in previous or delta, all content
            return DeltaMessage(content=delta_text)

    def extract_reasoning_content(self, model_output: str, request: ChatCompletionRequest,
                                  **kwargs) -> Tuple[Optional[str], Optional[str]]:
        """Extract reasoning content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Args:
            model_output (str): The model-generated string to extract reasoning content from.
            request (ChatCompletionRequest): he request object that was used to generate the model_output.

        Returns:
            reasoning_content (str | None): The reasoning content.
            final_output (str | None): The content.
        """
        start_index = model_output.find(self.think_start_token)
        end_index = model_output.find(self.think_end_token)
        # Thus we assume the reasoning content is always at the start.
        if end_index < 0:
            # for qwen3 model, the reasoning content is wrapped by <think> </think> xml tags
            if start_index < 0:
                return None, model_output
            reasoning_content = model_output[start_index + len(self.think_start_token):]
            reasoning_content = self._trim_newlines(reasoning_content)
            return reasoning_content, None

        if start_index >= 0 and start_index < end_index:
            reasoning_content = model_output[start_index + len(self.think_start_token):end_index]
        else:
            reasoning_content = model_output[:end_index]
        reasoning_content = self._trim_newlines(reasoning_content)

        final_output = model_output[end_index + len(self.think_end_token):]
        final_output = self._trim_newlines(final_output)

        if len(final_output) == 0:
            return reasoning_content, None
        return reasoning_content, final_output

    @classmethod
    def _trim_newlines(cls, text: str):
        """Trim newlines from the start and end of a string."""
        while text.startswith('\n'):
            text = text[1:]
        while text.endswith('\n'):
            text = text[:-1]
        return text
