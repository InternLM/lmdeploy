# Copyright (c) OpenMMLab. All rights reserved.

import re

from transformers import (PreTrainedTokenizerFast, StoppingCriteria,
                          StoppingCriteriaList)
from transformers.generation.streamers import BaseStreamer

try:
    # To support command line history
    import readline
except ImportError:
    pass  #readline not available


class TerminalIO:
    """Terminal input and output."""

    end_of_output = '\n'

    def input(self):
        """Read input from terminal."""

        print('\ndouble enter to end input >>> ', end='')
        sentinel = ''  # ends when this string is seen
        return '\n'.join(iter(input, sentinel))

    def output(self, string):
        print(string, end='', flush=True)


class BasicStreamer(BaseStreamer):
    """Basic streamer for HuggingFace models."""

    def __init__(self,
                 decode_func,
                 output_func,
                 end_of_output='\n',
                 skip_prompt=True):
        self.decode = decode_func
        self.output = output_func
        self.end_of_output = end_of_output
        self.skip_prompt = skip_prompt
        self.gen_len = 0

    def put(self, value):
        """Callback before forwarding current token id to model."""

        if self.gen_len == 0 and self.skip_prompt:
            pass
        else:
            token = self.decode(value)
            self.output(token)

        self.gen_len += 1

    def end(self):
        """Callback at the end of generation."""
        self.output(self.end_of_output)
        self.gen_len = 0


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
    """Default streamer for HuggingFace models."""

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
        """Convert token ids to tokens and decode."""

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
        """Fallback decoder for non-fast tokenizer."""

        tok = self.tokenizer.decode(value,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
        return tok + ' '

    def put(self, value):
        """Callback function to decode token and output to stdout."""

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
    """Streamer for InternLM."""

    def __init__(self, tokenizer, skip_prompt=True) -> None:
        BaseStreamer().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.gen_len = 0
        self.hex_regex = re.compile(r'^<0x([0-9ABCDEF]+)>$')

    def decode(self, value):
        """Decode generated tokens for InternLM."""

        tok = self.tokenizer.decode(value)
        if res := self.hex_regex.match(tok):
            tok = chr(int(res.group(1), 16))
        if tok == '</s>' or tok == '<eoa>' or tok == '\r':
            tok = '\n'

        return tok


class BaseDecorator:
    """Base decorator for decorating prompt and extracting generated output."""

    @classmethod
    def decorate(cls, prompt):
        """Abstract method for adding Add special tokens to prompt."""
        return prompt

    @classmethod
    def extract(cls, gen_out):
        """Abstract methods for extract generated output from model output."""
        return gen_out


class InternLMDecorator(BaseDecorator):
    """Decorator for InternLM."""

    regex = re.compile(r'<\|Bot\|>:(.*)')

    @classmethod
    def decorate(cls, prompt):
        """Decorate prompt for InternLM."""
        return f'<|User|>:{prompt}<eoh>'

    @classmethod
    def extract(cls, gen_out):
        """Extract generated tokens for InternLM."""
        return cls.regex.search(gen_out).group(1)


class InternLMStoppingCriteria(StoppingCriteria):
    """Stopping criteria for HF version of InternLM."""

    def __call__(self, input_ids, *args, **kwargs) -> bool:
        return input_ids[0, -1] in [2, 103028]


def test_terminal_io(monkeypatch):
    import io
    tio = TerminalIO()
    inputs = 'hello\n\n'
    # inputs = 'hello\n\n\x1B[A\n\n'
    monkeypatch.setattr('sys.stdin', io.StringIO(inputs))
    string = tio.input()
    tio.output(string)
    assert string == 'hello'
    # string = tio.input()
    # tio.output(string)
    # assert string == 'hello'


def test_basic_streamer():
    output = []

    def decode_func(value):
        return value + 10

    def output_func(value):
        output.append(value)

    streamer = BasicStreamer(decode_func, output_func)
    for i in range(10):
        streamer.put(i)
        if i == 5:
            streamer.end()
    streamer.end()

    assert output == [11, 12, 13, 14, 15, '\n', 17, 18, 19, '\n']

    output.clear()
    streamer = BasicStreamer(decode_func, output_func, skip_prompt=False)
    for i in range(10):
        streamer.put(i)
        if i == 5:
            streamer.end()
    streamer.end()

    assert output == [10, 11, 12, 13, 14, 15, '\n', 16, 17, 18, 19, '\n']
