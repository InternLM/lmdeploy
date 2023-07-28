# Copyright (c) OpenMMLab. All rights reserved.

from transformers.generation.streamers import BaseStreamer

from .dist import get_rank, master_only, master_only_and_broadcast_general

try:
    import readline  # To support command line history # noqa: F401
except ImportError:  # readline not available
    pass

import logging

logger = logging.getLogger(__name__)


class TerminalIO:
    """Terminal input and output."""

    end_of_output = '\n'

    @master_only_and_broadcast_general
    def input(self):
        """Read input from terminal."""

        print('\ndouble enter to end input >>> ', end='')
        sentinel = ''  # ends when this string is seen
        try:
            return '\n'.join(iter(input, sentinel))
        except EOFError:
            print('Detect EOF, exit')
            exit()

    @master_only
    def output(self, string):
        """Output to terminal with flush."""

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


def control(prompt, gen_config, sm):
    """Allow user to control generation config and session manager.

    Return:
        True if control command applied, False otherwise.
    """

    if prompt == 'exit':
        exit(0)

    if prompt == 'clear':
        sm.new_session()
        logger.info('Session cleared')
        return True

    # Re-config during runtime
    if prompt.startswith('config set'):
        try:
            keqv = prompt.split()[-1]
            k, v = keqv.split('=')
            v = eval(v)
            gen_config.__setattr__(k, v)
            logger.info(f'Worker {get_rank()} set {k} to {repr(v)}')
            logger.info(f'Generator config changed to: {gen_config}')

            return True
        except:  # noqa
            logger.info(
                'illegal instruction, treated as normal conversation. ')

    return False


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
