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
        self.last_dump_time = None
        dump_frequency = envs.expert_dump_frequency
        if dump_frequency < 1:
            logger.warning(f'LMDEPLOY_EXPERT_DUMP_FREQUENCY={dump_frequency} is invalid; defaulting to 1 second.')
            dump_frequency = 1
        self.dump_frequency = dump_frequency
        self.dump_rank = envs.expert_dump_rank

    def _build_counts_tensor(self):
        """Stack per-layer counts into a (num_layers, num_experts) tensor,
        sorted by layer index."""
        sorted_keys = sorted(self.accum_token_counts.keys(), key=lambda k: int(k.split('_')[0]))
        return torch.stack([self.accum_token_counts[k].cpu() for k in sorted_keys])

    @staticmethod
    def _compute_balancedness(counts: torch.Tensor) -> torch.Tensor:
        """Per-layer balancedness: mean / max load. Shape: (num_layers,). Range (0,1]; 1 = perfect balance."""
        counts_f = counts.float()
        return (counts_f.mean(dim=1) + 1e-5) / (counts_f.max(dim=1).values + 1e-5)

    def record(self, topk_ids, layer_index, num_experts):
        key = f'{layer_index}_{num_experts}'
        if key not in self.dispatch_count:
            self.dispatch_count[key] = 0
        self.dispatch_count[key] += 1
        if key not in self.accum_token_counts:
            self.accum_token_counts[key] = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
        topk_ids_flat = topk_ids.reshape(-1).long()
        # scatter_add_ is graph-capturable; torch.bincount is not
        self.accum_token_counts[key].scatter_add_(
            0, topk_ids_flat, torch.ones(topk_ids_flat.numel(), dtype=torch.int64, device=topk_ids_flat.device))

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != self.dump_rank:
            return

        # skip until this layer has been seen at least twice, which guarantees
        # one full forward pass has completed and all layers have accumulated data.
        if self.dispatch_count[key] < 2:
            return

        now = datetime.now()
        dump_interval_seconds = self.dump_frequency
        if self.last_dump_time is not None and (now - self.last_dump_time).total_seconds() < dump_interval_seconds:
            return

        self.last_dump_time = now
        self._dump(rank, step=self.dispatch_count[key])

    def _dump(self, rank: int, step: int):
        # all-reduce local counts to get global distribution.
        for k, local_counts in self.accum_token_counts.items():
            if dist.is_initialized():
                dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)

        counts_tensor = self._build_counts_tensor()  # (num_layers, num_experts)
        balancedness = self._compute_balancedness(counts_tensor)  # (num_layers,)

        # log per-layer balancedness; highlight the most imbalanced layers.
        bal_list = balancedness.tolist()
        bottom3 = sorted(range(len(bal_list)), key=lambda i: bal_list[i])[:3]
        logger.info(
            f'[Expert Statistics] step={step} | avg_balancedness={sum(bal_list)/len(bal_list):.4f} | '
            f'most_imbalanced_layers={[(i, f"{bal_list[i]:.4f}") for i in bottom3]}'
        )

        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, f'rank{rank}_step{step}_expert_counts.pt')
        torch.save(
            {
                'counts': counts_tensor,       # (num_layers, num_experts), int64
                'balancedness': balancedness,  # (num_layers,), float32
                'step': step,
                'rank': rank,
            },
            filepath,
        )
        logger.info(f'[Expert Statistics] Expert distribution dumped to {filepath}, shape={list(counts_tensor.shape)}')

        if envs.expert_dump_visualize:
            from lmdeploy.pytorch.tools.utils import visualize_expert_distribution
            visualize_expert_distribution(filepath)
            logger.info(f'[Expert Statistics] Heatmap saved alongside {filepath}')


class ExpertsDistributionRecorder:
    """Factory class that returns a real or no-op recorder."""

    def __new__(cls, *args, **kwargs):
        if envs.dump_expert_distribution:
            logger.info('Expert distribution recorder is enabled.')
            return _ExpertsDistributionRecorderImpl(*args, **kwargs)
        else:
            logger.info('Expert distribution recorder is disabled.')
            return _NoOpExpertsDistributionRecorder(*args, **kwargs)
