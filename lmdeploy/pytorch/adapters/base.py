# Copyright (c) OpenMMLab. All rights reserved.
"""Basic adapter suitable for general HuggingFace models."""

import logging
import re

from transformers import (PreTrainedTokenizer, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)


class BaseAdapter:
    """Base class for all adapters.

    Note:
        Adapters coordinate with the session manager to prepare input_ids.
        The full sequence fed to the model is as follows:

            ```
            adapter.start_ids
            adapter.encode_and_decorate(user_input_1)
            output_1_generated_by_model
            adapter.sep_ids
            adapter.encode_and_decorate(user_input_2)
            output_2_generated_by_model
            adapter.sep_ids
            adapter.encode_and_decorate(user_input_3)
            ```

        Thus adapter is responsible for providing model specific
        ``start_ids``, ``sep_ids``, and method to encode single prompt.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def encode_and_decorate(self, prompt, add_special_tokens=False):
        """Model specific method to encode and decorate prompt."""
        raise NotImplementedError

    def decode(self, value):
        """Model specific method to decode single value to string."""
        raise NotImplementedError

    @property
    def stopping_criteria(self):
        """Model specific stopping criteria for generation."""
        return None

    @property
    def start_ids(self):
        """Model specific start ids."""
        return [self.tokenizer.bos_token_id]

    @property
    def sep_ids(self):
        """Model specific separation ids."""
        return [self.tokenizer.bos_token_id]


class BasicAdapter(BaseAdapter):
    """Basic adapter for slow tokenizers."""

    def encode_and_decorate(self, prompt, add_special_tokens=False):
        """Encode prompt.

        Note:
            we leave <bos> to session manager to add.
        """
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors='pt',
        )
        logger.debug(f'Encode {prompt} to {input_ids}')
        return input_ids

    def decode(self, value):
        """Fallback when tokenizer is not fast."""

        self.tokenizer: PreTrainedTokenizer

        tok = self.tokenizer.decode(value)
        return tok + ' '


class BasicAdapterFast(BaseAdapter):
    """Basic adapter for slow tokenizers."""

    hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def encode_and_decorate(self, prompt, add_special_tokens=False):
        """Encode prompt.

        Note:
            we leave <bos> to session manager to add.
        """
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors='pt',
        )
        logger.debug(f'Encode {prompt} to {input_ids}')
        return input_ids

    def decode(self, value):
        """Decode with fast tokenizers."""

        self.tokenizer: PreTrainedTokenizerFast

        tok = self.tokenizer._convert_id_to_token(value)
        if tok.startswith('▁'):  # sentencepiece
            space = ' '
            tok = tok[1:]
        else:
            space = ''
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '\r':
            tok = '\n'

        tok = space + tok

        logger.debug(f'Decode {value} to {repr(tok)}')

        return tok
