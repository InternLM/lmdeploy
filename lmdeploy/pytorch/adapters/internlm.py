# Copyright (c) OpenMMLab. All rights reserved.
import logging
import re

from transformers import (PreTrainedTokenizerFast, StoppingCriteria,
                          StoppingCriteriaList)
import torch

logger = logging.getLogger(__name__)


class InternLMStoppingCriteria(StoppingCriteria):
    """Stopping criteria for HF version of InternLM."""

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        return input_ids[0, -1] in [2, 103028]


class InternLMAdapter:
    hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')
    # id of '<|User|>:'
    B_USER_ID = torch.tensor([[1, 333, 352, 1621, 352, 27232]])
    E_USER_ID = torch.tensor([[103027, 13, 333, 352, 23845, 352, 27232]])

    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def encode_and_decorate(self, prompt):
        r"""Encode prompt and decorate with template.

        InternLM use the following template:

        <bos> (no actual newline here, just for better readability)
        <|User|>:{prompt}<eoh>\n
        <|Bot|>:{model_output}<eoa>\n
        <|User|>:{prompt}<eoh>\n
        <|Bot|>:{model_output}<eoa>\n
        ...
        <eos>

        Note: we leave <bos> or chat history to session manager to add,
        so we will decorate input_ids with <|User|>:{prompt}<eoh>\n<|Bot|>:
        """
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            return_tensors='pt',
        )
        # This is f'<|User|>:{prompt}<eoh>\n<|Bot|>:'
        # but force \n to 13 instead of 364
        input_ids = torch.cat([self.B_USER_ID, input_ids, self.E_USER_ID])
        return input_ids

    def decode(self, value):
        """Decode generated tokens for InternLM."""

        tok = self.tokenizer.decode(value)
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '<eoa>' or tok == '\r':
            tok = '\n'

        logger.debug(f'Decode {value} to {repr(tok)}')

        return tok

    @property
    def stopping_criteria(self):
        return StoppingCriteriaList([InternLMStoppingCriteria()])
