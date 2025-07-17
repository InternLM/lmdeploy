# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/reasoning_parsers
import re
from typing import Optional, Sequence, Tuple, Union

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

from .reasoning_parser import ReasoningParser, ReasoningParserManager


@ReasoningParserManager.register_module(name='deepseek-r1')
class DeepSeekR1ReasoningParser(ReasoningParser):
    """Reasoning parser for DeepSeek R1 model.

    The DeepSeek R1 model uses <think>...</think> tokens to denote reasoning text. This parser extracts the reasoning
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

        self.think_start_token_id = self.vocab.get(self.think_start_token)
        self.think_end_token_id = self.vocab.get(self.think_end_token)
        if (self.think_start_token_id is None or self.think_end_token_id is None):
            raise RuntimeError('DeepSeek R1 reasoning parser could not locate think start/end '
                               'tokens in the tokenizer!')

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
        if len(delta_token_ids) == 1:
            if delta_token_ids[0] == self.think_end_token_id:
                return DeltaMessage(content='')
            elif delta_token_ids[0] == self.think_start_token_id:
                return None

        # Check if <think> is present in previous or delta.
        # Keep compatibility with models that don't generate <think> tokens.
        if self.think_start_token_id in previous_token_ids:
            if self.think_end_token_id in delta_token_ids:
                # <think> in previous, </think> in delta,
                # extract reasoning content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            elif self.think_end_token_id in previous_token_ids:
                # <think> in previous, </think> in previous,
                return DeltaMessage(content=delta_text)
            else:
                # <think> in previous, no </think> in previous or delta,
                # reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)
        elif self.think_start_token_id in delta_token_ids:
            if self.think_end_token_id in delta_token_ids:
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
            # No <think> in previous or delta, also need to check for </think>.
            # Because the model may have generated </think> without <think>
            # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
            if self.think_end_token_id in delta_token_ids:
                # </think> in delta with more tokens,
                # extract reasoning content and content
                end_index = delta_text.find(self.think_end_token)
                reasoning_content = delta_text[:end_index]
                content = delta_text[end_index + len(self.think_end_token):]
                return DeltaMessage(reasoning_content=reasoning_content, content=content if content else None)
            elif self.think_end_token_id in previous_token_ids:
                # </think> in previous, thinking content ends
                return DeltaMessage(content=delta_text)
            else:
                # no </think> in previous or delta, reasoning content continues
                return DeltaMessage(reasoning_content=delta_text)

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
        # DeepSeek R1 doesn't generate <think> now.
        # Thus we assume the reasoning content is always at the start.
        # Ref https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f
        if self.think_end_token not in model_output:
            return model_output, None
        else:
            # Add a start token if it's missing to keep compatibility.
            if self.think_start_token not in model_output:
                model_output = f'{self.think_start_token}{model_output}'
            # Use a regex to find the reasoning content
            reasoning_content = self.reasoning_regex.findall(model_output)[0]

            end_index = len(f'{self.think_start_token}{reasoning_content}{self.think_end_token}')
            final_output = model_output[end_index:]

            if len(final_output) == 0:
                return reasoning_content, None

            return reasoning_content, final_output
