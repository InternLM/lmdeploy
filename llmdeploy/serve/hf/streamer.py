# Copyright (c) OpenMMLab. All rights reserved.
import torch
from transformers.generation.streamers import BaseStreamer


class DecodeOutputStreamer(BaseStreamer):
    """Output generated tokens to shell."""

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0

    def put(self, value):
        if self.gen_len == 0 and self.skip_prompt:
            pass
        else:
            tok = self.tokenizer.decode(value[0],
                                        skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)
            print('', tok, end='', flush=True)

        self.gen_len += 1

    def end(self):
        print('\n')
