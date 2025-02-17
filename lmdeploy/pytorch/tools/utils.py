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
    def format_duration(duration: float):
        """format duration."""
        unit = 'ms'
        if duration < 1:
            duration *= 1000
            unit = 'μs'
        elif duration > 1000:
            duration /= 1000
            unit = 's'
        
        return f'{duration:.3f} {unit}'

    @staticmethod
    def format_flops(flops: float):
        """compute flops"""
        unit = ''
        if flops > (1<<40):
            flops /= (1<<40)
            unit = 'T'
        elif flops > (1<<30):
            flops /= (1<<30)
            unit = 'G'
        elif flops > (1<<20):
            flops /= (1<<20)
            unit = 'M'
        elif flops > (1<<10):
            flops /= (1<<10)
            unit = 'K'
        return f'{flops:.3f} {unit}Flop/s'
    
    def print(self, flop: int = None):
        """print."""
        if self.duration is None:
            print('Please run Timer.tic() first.')
            return

        formated_dur = self.format_duration(self.duration)
        print(f'Take time: {formated_dur}')

        if flop is not None:
            flops = flop / self.duration * 1000
            formated_flops = self.format_flops(flops)
            print(f'Flops: {formated_flops}')