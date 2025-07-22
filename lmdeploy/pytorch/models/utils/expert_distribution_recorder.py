# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/dlblas/layers/moe/experts_distribution_recorder.py

import os
from datetime import datetime

import torch
import torch.distributed as dist

from lmdeploy.pytorch import envs
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class _NoOpExpertsDistributionRecorder:
    """A no-op version of the recorder that does nothing."""

    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass


class _ExpertsDistributionRecorderImpl:
    """The actual implementation of the recorder."""

    def __init__(self):
        self.output_dir = envs.expert_dump_dir
        self.dispatch_count = {}
        self.accum_token_counts = {}
        self.global_token_counts = {}
        self.last_dump_minute = -1
        self.dump_frequency = envs.expert_dump_frequency
        self.dump_rank = envs.expert_dump_rank

    def map_to_sorted_2d_array(self, data_map):
        sorted_keys = sorted(data_map.keys(), key=lambda k: int(k.split('_')[0]))
        data_2d_array = [data_map[key].cpu().tolist() for key in sorted_keys]
        return data_2d_array

    def record(self, topk_ids, layer_index, num_experts):
        key = f'{layer_index}_{num_experts}'
        if key not in self.dispatch_count:
            self.dispatch_count[key] = 0
        self.dispatch_count[key] += 1
        if key not in self.accum_token_counts:
            self.accum_token_counts[key] = torch.zeros(num_experts, dtype=torch.int64, device='cuda')
        topk_ids_flat = topk_ids.view(-1)
        step_local_counts = torch.bincount(topk_ids_flat, minlength=num_experts)
        self.accum_token_counts[key] += step_local_counts
        global_token_counts_tmp = self.accum_token_counts[key].clone()
        if dist.is_initialized():
            dist.all_reduce(global_token_counts_tmp, op=dist.ReduceOp.SUM)
        self.global_token_counts[key] = global_token_counts_tmp
        rank = dist.get_rank() if dist.is_initialized() else 0
        now = datetime.now()
        if (rank == self.dump_rank and now.minute % self.dump_frequency == 0 and now.minute != self.last_dump_minute):
            self.last_dump_minute = now.minute
            global_list = self.map_to_sorted_2d_array(self.global_token_counts)
            step = self.dispatch_count[key]
            os.makedirs(self.output_dir, exist_ok=True)
            token_counts_file_name = f'rank{rank}_step{step}_experts_counts.json'
            filepath = os.path.join(self.output_dir, token_counts_file_name)
            with open(filepath, 'w') as f:
                import json
                json.dump(global_list, f, indent=2)
            logger.info(f'[EPLB] Experts distribution dumped to {filepath}')


class ExpertsDistributionRecorder:
    """Factory class that returns a real or no-op recorder."""

    def __new__(cls, *args, **kwargs):
        if envs.dump_expert_distribution:
            logger.info('Expert distribution recorder is enabled.')
            return _ExpertsDistributionRecorderImpl(*args, **kwargs)
        else:
            logger.info('Expert distribution recorder is disabled.')
            return _NoOpExpertsDistributionRecorder(*args, **kwargs)
