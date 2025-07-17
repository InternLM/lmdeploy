# Copyright (c) OpenMMLab. All rights reserved.
# modified from https://github.com/vllm-project/vllm/tree/v0.7.3/vllm/entrypoints/openai/reasoning_parsers
from functools import cached_property
from typing import Dict, Optional, Sequence, Tuple, Union

from mmengine import Registry

from lmdeploy.serve.openai.protocol import ChatCompletionRequest, DeltaMessage

ReasoningParserManager = Registry('reasoning_parser', locations=['lmdeploy.serve.openai.reasoning_parser'])


class ReasoningParser:

    def __init__(self, tokenizer: object):
        self.model_tokenizer = tokenizer

    @cached_property
    def vocab(self) -> Dict[str, int]:
        # NOTE: Only PreTrainedTokenizerFast is guaranteed to have .vocab
        # whereas all tokenizers have .get_vocab()
        return self.model_tokenizer.get_vocab()

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
        raise NotImplementedError('ReasoningParser.extract_reasoning_content_streaming '
                                  'has not been implemented!')

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
        raise NotImplementedError('ReasoningParser.extract_reasoning_content '
                                  'has not been implemented!')
