# Copyright (c) 2024, DeepLink. All rights reserved.
import torch_npu
from torch_npu.profiler import profile as npu_profile

from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


def _build_npu_profiler(self):
    from lmdeploy.pytorch import envs

    activities = []
    if envs.torch_profile_cpu:
        activities.append(torch_npu.profiler.ProfilerActivity.CPU)
    if envs.torch_profile_cuda:
        activities.append(torch_npu.profiler.ProfilerActivity.NPU)
    if len(activities) > 0:
        logger.warning(f"Profiler start on {self.name}. Please Note that profiling might harm performance.")
        profiler = npu_profile(activities=activities)
        return profiler
    else:
        return None
