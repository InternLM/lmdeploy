# Copyright (c) OpenMMLab. All rights reserved.

import re

from transformers import PreTrainedTokenizerFast
from transformers.generation.streamers import BaseStreamer


class DecodeOutputStreamer(BaseStreamer):
    """Output generated tokens to shell."""

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            self.decode = self._decode_with_raw_id
            self.hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')
        else:
            self.decode = self._decode_fallback

    def _decode_with_raw_id(self, value):
        tok = self.tokenizer._convert_id_to_token(value)
        if tok.startswith('▁'):  # sentencepiece
            space = ' '
            tok = tok[1:]
        else:
            space = ''
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>':
            tok = '\n'
        return space + tok

    def _decode_fallback(self, value):
        tok = self.tokenizer.decode(value,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
        return tok + ' '

    def put(self, value):
        if self.gen_len == 0 and self.skip_prompt:
            pass
        else:
            tok = self.decode(value[0])
            print(tok, end='', flush=True)

        self.gen_len += 1

    def end(self):
        print('\n')
