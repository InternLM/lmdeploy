# Copyright (c) OpenMMLab. All rights reserved.

import re

from transformers import (PreTrainedTokenizerFast, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer


def get_utils(model):
    """Get utils by model type."""
    name = model.__class__.__name__
    if name == 'InferenceEngine':
        name = model.module.__class__.__name__

    if name == 'InternLMForCausalLM':
        stop_criteria = InternLMStoppingCriteria()
        stop_criteria = StoppingCriteriaList([stop_criteria])
        return InternLMDecorator, InternLMStreamer, stop_criteria
    else:
        return BaseDecorator, DecodeOutputStreamer, None


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
        """Callback function to print generated tokens."""

        if self.gen_len == 0 and self.skip_prompt:
            pass
        else:
            tok = self.decode(value[0])
            print(tok, end='', flush=True)

        self.gen_len += 1

    def end(self):
        """Callback function to finish generation."""

        print('\n')


class InternLMStreamer(DecodeOutputStreamer):
    """Output generated tokens to shell."""

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        BaseStreamer().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0
        self.hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def decode(self, value):
        tok = self.tokenizer.decode(value)
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '<eoa>':
            tok = '\n'

        return tok


class BaseDecorator:
    """Base decorator for prompt and generated output."""

    @classmethod
    def decorate(cls, prompt):
        """Add special tokens to prompt."""
        return prompt

    @classmethod
    def extract(cls, gen_out):
        """Extract generated output from model output."""
        return gen_out


class InternLMDecorator(BaseDecorator):
    """Decorator for InternLM."""

    regex = re.compile(r'<\|Bot\|>:(.*)')

    @classmethod
    def decorate(cls, prompt):
        return f'<|User|>:{prompt}<eoh>'

    @classmethod
    def extract(cls, gen_out):
        return cls.regex.search(gen_out).group(1)


class InternLMStoppingCriteria(StoppingCriteria):
    """Stopping criteria for HF version of InternLM."""

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        return input_ids[0, -1] in [2, 103028]
