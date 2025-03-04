# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager


class Timer:
    """debug timer."""

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
        """format duration."""
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
        """compute flops."""
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
        """formatted print."""
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
