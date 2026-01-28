# Copyright (c) OpenMMLab. All rights reserved.
import asyncio

import torch
from torch.profiler import ProfilerActivity, profile

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
        self._task = None
        self._started = False
        if self.dp > 1 and self.duration < 0 and self.profiler is not None:
            logger.warning('Do not support duration<=0 for dp > 1.')
            self.profiler = None

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
            dump_path = f'{self.prefix}{rank}.json'
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
