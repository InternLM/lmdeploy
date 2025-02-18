# Copyright (c) OpenMMLab. All rights reserved.
from contextlib import contextmanager


class Timer:
    """debug timer."""

    def __init__(self):
        self.duration = None

    def tic_cpu(self):
        import time
        start = time.perf_counter()
        yield self
        end = time.perf_counter()
        self.duration = (end - start) * 1000

    def tic_cuda(self):
        import torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        yield self
        end.record()
        torch.cuda.synchronize()
        self.duration = start.elapsed_time(end)

    @classmethod
    @contextmanager
    def tic(cls, is_cuda: bool = False) -> 'Timer':
        timer = Timer()
        if is_cuda:
            yield from timer.tic_cuda()
        else:
            yield from timer.tic_cpu()

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
    def formatted_print(out_info: dict):
        """formatted print."""
        max_key_len = max(len(k) for k in out_info.keys())
        max_key_len = min(10, max_key_len)
        max_val_len = max(len(k) for k in out_info.values())
        max_val_len = min(10, max_val_len)
        for k, v in out_info.items():
            print(f'{k:>{max_key_len}} : {v:>{max_val_len}}')

    def print(self, flop: int = None):
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

        self.formatted_print(out_info)
