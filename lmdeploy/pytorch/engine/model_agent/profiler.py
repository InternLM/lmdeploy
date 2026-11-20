# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import time
from collections import deque

import torch
from torch.profiler import ProfilerActivity, profile

from lmdeploy.profiling import trace_phase
from lmdeploy.pytorch.distributed import DistContext
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class AgentProfiler:

    def __init__(self, dist_ctx: DistContext, stream: torch.Stream):
        from lmdeploy.pytorch import envs
        self.rank = dist_ctx.rank
        self.dp_rank = dist_ctx.dp_rank
        self.dp = dist_ctx.dist_config.dp
        self.stream = stream
        self.profiler = None
        self.name = f'rank[{self.rank}]'

        self.delay = envs.torch_profile_delay
        self.duration = envs.torch_profile_duration

        self.profiler = self._build_profiler()
        self.prefix = envs.torch_profile_output_prefix
        self.use_gzip = envs.torch_profile_use_gzip
        self._task = None
        self._started = False
        self._phase_events = deque()
        self._phase_step = None
        self._phase_marks = []
        self._phase_index = 0
        if self.dp > 1 and self.duration < 0 and self.profiler is not None:
            logger.warning('Do not support duration<=0 for dp > 1.')
            self.profiler = None

    def begin_step(self, inputs, *, access_id, worker_started_at, prepare_ms, dp_wait_ms):
        """Record stream time without adding a synchronization to inference."""
        start = torch.cuda.Event(enable_timing=True)
        start.record(self.stream)
        self._phase_index += 1
        self._phase_marks = []
        self._phase_step = (start, time.perf_counter(), dict(
            rank=self.rank, dp_rank=self.dp_rank, step=self._phase_index, started_at=time.time(),
            access_id=access_id, worker_started_at=worker_started_at, prepare_ms=prepare_ms, dp_wait_ms=dp_wait_ms,
            prefill=not inputs.is_decoding, dummy=inputs.is_dummy,
            skip_sampling=inputs.skip_sampling,
            batch=inputs.seq_length.numel(), tokens=inputs.input_ids.numel()))

    def mark_step(self, name: str):
        """Mark a stream boundary inside the current inference step."""
        event = torch.cuda.Event(enable_timing=True)
        event.record(self.stream)
        self._phase_marks.append((name, event))

    def end_step(self, graph_mode: str):
        """Keep events until their stream work completes; never wait here."""
        if self._phase_step is not None:
            start, submitted, fields = self._phase_step
            end = torch.cuda.Event(enable_timing=True)
            end.record(self.stream)
            fields['submit_ms'] = (time.perf_counter() - submitted) * 1000
            fields['graph_mode'] = graph_mode
            self._phase_events.append((start, end, fields, self._phase_marks))
            self._phase_step = None
        while self._phase_events and self._phase_events[0][1].query():
            start, end, fields, marks = self._phase_events.popleft()
            previous = start
            for name, event in marks:
                fields[name + '_ms'] = previous.elapsed_time(event)
                previous = event
            fields['finish_ms'] = previous.elapsed_time(end)
            # Stream elapsed time includes kernels, collectives and launch gaps;
            # it is not a sum of kernel execution times.
            trace_phase('worker_step', stream_ms=start.elapsed_time(end), **fields)

    def _build_profiler(self):
        from lmdeploy.pytorch import envs
        activities = []
        if envs.torch_profile_cpu:
            activities.append(ProfilerActivity.CPU)
        if envs.torch_profile_cuda:
            activities.append(ProfilerActivity.CUDA)
        if len(activities) > 0:
            logger.warning(f'Profiler start on {self.name}. '
                           'Please Note that profiling might harm performance.')
            profiler = profile(activities=activities)
            return profiler
        else:
            return None

    def dump(self):
        """Dump profile result."""
        if self.profiler is None:
            return

        if not self._started:
            logger.warning(f'Profiler {self.name} not started, skip dump.')
            return

        try:
            self.profiler.stop()
            rank = self.rank
            suffix = '.json.gz' if self.use_gzip else '.json'
            dump_path = f'{self.prefix}{rank}{suffix}'
            self.profiler.export_chrome_trace(dump_path)
            logger.warning(f'Profiler {self.name} dump to {dump_path}.')
        except Exception as e:
            logger.error(f'Failed to dump profile {self.name} result: {e}')
        finally:
            self.profiler = None

    async def profile_task(self):
        """Profile task."""
        if self.profiler is None:
            return

        # start profiler with delay
        await asyncio.sleep(self.delay)
        self.profiler.start()
        self._started = True

        if self.duration <= 0:
            return

        # dump profiler
        await asyncio.sleep(self.duration)
        self.dump()

    def create_task(self):
        """Create task."""
        event_loop = asyncio.get_event_loop()
        self._task = event_loop.create_task(self.profile_task())
