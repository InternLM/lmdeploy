# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager
from typing import List


class Timer:
    """Debug timer."""

    def __init__(self):
        self.duration = None
        self.timer_type = None

    def tic_cpu(self):
        self.timer_type = 'cpu'
        import time
        self._start = time.perf_counter()

    def toc_cpu(self):
        assert self.timer_type == 'cpu'
        import time
        self._end = time.perf_counter()
        self.duration = (self._end - self._start) * 1000
        return self

    def tic_cuda(self):
        self.timer_type = 'cuda'
        import torch
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def toc_cuda(self):
        assert self.timer_type == 'cuda'
        import torch
        self._end.record()
        torch.cuda.synchronize()
        self.duration = self._start.elapsed_time(self._end)
        return self

    @classmethod
    def tic(cls, is_cuda: bool = False) -> 'Timer':
        timer = Timer()
        if is_cuda:
            timer.tic_cuda()
        else:
            timer.tic_cpu()
        return timer

    def toc(self):
        if self.timer_type == 'cpu':
            return self.toc_cpu()
        elif self.timer_type == 'cuda':
            return self.toc_cuda()
        else:
            raise RuntimeError(f'Unknown timer_type: {self.timer_type}')

    @classmethod
    @contextmanager
    def timing(cls, is_cuda: bool = False) -> 'Timer':
        timer = cls.tic(is_cuda=is_cuda)
        yield timer
        timer.toc()

    @staticmethod
    def format_duration(duration: float, acc: int = 3):
        """Format duration."""
        unit = 'ms'
        if duration < 1:
            duration *= 1000
            unit = 'μs'
        elif duration > 1000:
            duration /= 1000
            unit = 's'

        return f'{duration:.{acc}f} {unit}'

    @staticmethod
    def format_flops(flops: float, acc: int = 3):
        """Compute flops."""
        unit = ''
        if flops > (1 << 40):
            flops /= (1 << 40)
            unit = 'T'
        elif flops > (1 << 30):
            flops /= (1 << 30)
            unit = 'G'
        elif flops > (1 << 20):
            flops /= (1 << 20)
            unit = 'M'
        elif flops > (1 << 10):
            flops /= (1 << 10)
            unit = 'K'
        return f'{flops:.{acc}f} {unit}Flop/s'

    @staticmethod
    def formatted_print(out_info: dict, title: str = None):
        """Formatted print."""
        max_key_len = max(len(k) for k in out_info.keys())
        max_key_len = min(10, max_key_len)
        max_val_len = max(len(k) for k in out_info.values())
        max_val_len = min(10, max_val_len)

        if title is not None:
            print(title)
        for k, v in out_info.items():
            print(f'{k:>{max_key_len}} : {v:>{max_val_len}}')

    def print(self, flop: int = None, title: str = None):
        """print."""
        if self.duration is None:
            print('Please run Timer.tic() first.')
            return

        out_info = dict()

        formated_dur = self.format_duration(self.duration)
        out_info['Duration'] = f'{formated_dur}'

        if flop is not None:
            flops = flop / self.duration * 1000
            formated_flops = self.format_flops(flops)
            out_info['Flops'] = f'{formated_flops}'

        self.formatted_print(out_info, title)

    def toc_print(self, flop: int = None, title: str = None):
        return self.toc().print(flop=flop, title=title)


def visualize_pipe_out(outputs, enable_meta: bool = True):
    import os

    from lmdeploy.messages import Response

    try:
        from termcolor import colored
    except ImportError:

        def colored(text, color=None, on_color=None, attrs=None):
            return text

    if isinstance(outputs, Response):
        outputs = [outputs]
    elif outputs is None:
        outputs = [outputs]
    try:
        term_size = os.get_terminal_size().columns
    except Exception:
        term_size = 100

    border_color = 'cyan'
    meta_color = 'light_grey'
    number_color = 'green'

    def _print_title(title: str, color: str = border_color):
        title_text = f' {title} '
        print(colored(f'【{title_text}】', color, attrs=['bold']))

    def _print_section(title: str, content: str, color: str = border_color):
        """Simple title and content printing."""
        _print_title(title, color)
        print(content)

    def _print_meta(out: Response):
        """Enhanced meta information display."""
        # Create a clean table-like format
        finish_color = 'yellow' if out.finish_reason == 'stop' else 'red'
        meta_content = [
            f"{colored('• Input Tokens:', meta_color)}     {colored(out.input_token_len, number_color)}",
            f"{colored('• Generated Tokens:', meta_color)} {colored(out.generate_token_len, number_color)}",
            f"{colored('• Finish Reason:', meta_color)}    {colored(out.finish_reason, finish_color)}"
        ]
        if out.routed_experts is not None:
            shape = tuple(out.routed_experts.shape)
            meta_content.append(f"{colored('• Routed Experts:', meta_color)}  {colored(shape, number_color)}")
        if out.logits is not None:
            shape = tuple(out.logits.shape)
            meta_content.append(f"{colored('• Logits Shape:', meta_color)}     {colored(shape, number_color)}")
        if out.logprobs is not None:
            size = len(out.logprobs)
            meta_content.append(f"{colored('• Logprobs:', meta_color)}      {colored(size, number_color)}")

        lines = '\n'.join(meta_content)
        lines += '\n'
        _print_section('METADATA', lines, border_color)

    # Main loop
    print(colored('━' * term_size, border_color))

    outputs: List[Response] = outputs
    for idx, out in enumerate(outputs):
        header = f'OUTPUT [{idx + 1}/{len(outputs)}]'
        header_formatted = colored(f'✦ {header}', 'light_magenta', attrs=['bold'])
        print(header_formatted)
        print()

        if out is not None:
            if enable_meta:
                _print_meta(out)

            _print_section('TEXT', out.text, border_color)

        if idx < len(outputs) - 1:  # Add separator when it's not the last output
            print(colored('─' * (term_size), border_color, attrs=['dark']))
        else:
            print(colored('━' * term_size, border_color))


def visualize_chat_completions(outputs, enable_meta: bool = True):
    """Visualize chat completions."""
    from openai.types.chat import ChatCompletion

    from lmdeploy.messages import Response
    if isinstance(outputs, ChatCompletion):
        outputs = [outputs]

    resps = []
    for out in outputs:
        assert isinstance(out, ChatCompletion)
        choice = out.choices[0]
        resp = Response(text=choice.message.content,
                        input_token_len=out.usage.prompt_tokens,
                        generate_token_len=out.usage.completion_tokens,
                        finish_reason=choice.finish_reason)
        resps.append(resp)

    return visualize_pipe_out(resps, enable_meta=enable_meta)
