# Copyright (c) OpenMMLab. All rights reserved.
"""Basic adapter suitable for general HuggingFace models."""

import logging
import re

from transformers import (PreTrainedTokenizer, PreTrainedTokenizerBase,
                          PreTrainedTokenizerFast)

logger = logging.getLogger(__name__)


class BasicAdapter:
    hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            self.decode = self._decode_fast
        else:
            self.decode = self._decode_fallback

    def encode_and_decorate(self, prompt, add_special_tokens=False):
        """Encode prompt and decorate with template.

        Note: we leave <bos> to session manager to add
        """
        input_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors='pt',
        )
        logger.debug(f'Encode {prompt} to {input_ids}')
        return input_ids

    def _decode_fallback(self, value):
        """Fallback when tokenizer is not fast."""

        self.tokenizer: PreTrainedTokenizerBase

        tok = self.tokenizer.decode(value)
        return tok

    def _decode_fast(self, value):
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

    @property
    def stopping_criteria(self):
        return None