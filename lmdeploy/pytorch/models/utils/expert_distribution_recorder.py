# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/DeepLink-org/dlBLAS/blob/main/dlblas/layers/moe/experts_distribution_recorder.py

import os

import torch
import torch.distributed as dist

from lmdeploy.pytorch import envs
from lmdeploy.utils import get_logger

logger = get_logger('lmdeploy')


class _ExpertsDistributionRecorderNoOp:
    """No-op recorder used when expert distribution recording is disabled."""

    def record(self, *args, **kwargs):
        pass

    def start_record(self):
        pass

    def stop_record(self):
        pass

    def dump_record(self):
        pass


class _ExpertsDistributionRecorderImpl:
    """Records per-expert token dispatch counts across MoE layers."""

    def __init__(self):
        self.output_dir = envs.expert_dump_dir
        self.dispatch_count = {}
        self.accum_token_counts = {}
        self.dump_rank = envs.expert_dump_rank
        self._recording = False

    def _reset_accumulators(self):
        self.dispatch_count.clear()
        self.accum_token_counts.clear()

    def _build_counts_tensor(self, counts_dict=None):
        """Stack per-layer counts into a (num_layers, num_experts) tensor."""
        if counts_dict is None:
            counts_dict = self.accum_token_counts
        sorted_keys = sorted(counts_dict.keys(), key=lambda k: int(k.split('_')[0]))
        return torch.stack([counts_dict[k].cpu() for k in sorted_keys])

    @staticmethod
    def _compute_balancedness(counts: torch.Tensor) -> torch.Tensor:
        """Per-layer balancedness: mean / max load. Shape: (num_layers,). Range (0,1]; 1 = perfect balance."""
        counts_f = counts.float()
        return (counts_f.mean(dim=1) + 1e-5) / (counts_f.max(dim=1).values + 1e-5)

    def record(self, topk_ids, layer_index, num_experts):
        if not self._recording:
            return

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

    def start_record(self):
        logger.info('[Expert Statistics] Recording started.')
        self._reset_accumulators()
        self._recording = True

    def stop_record(self):
        logger.info('[Expert Statistics] Recording stopped.')
        self._recording = False

    def dump_record(self):
        if not self._recording:
            logger.info('[Expert Statistics] dump_record called but recording is not active.')
            return None

        if not self.accum_token_counts:
            logger.info('[Expert Statistics] dump_record called but no data has been accumulated yet.')
            return None

        if torch.cuda.is_current_stream_capturing():
            logger.warning('[Expert Statistics] dump_record skipped during CUDA graph capture.')
            return None

        rank = dist.get_rank() if dist.is_initialized() else 0
        step = max(self.dispatch_count.values()) if self.dispatch_count else 0
        return self._dump(rank, step)

    def _dump(self, rank: int, step: int):
        logger.info(f'[Expert Statistics] Dumping expert distribution at step {step} from rank {rank}...')

        if dist.is_initialized():
            # clone before all_reduce to avoid corrupting the local accumulator.
            global_counts = {k: v.clone() for k, v in self.accum_token_counts.items()}
            for t in global_counts.values():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        else:
            global_counts = self.accum_token_counts

        if rank != self.dump_rank:
            return None

        counts_tensor = self._build_counts_tensor(global_counts)  # (num_layers, num_experts)
        balancedness = self._compute_balancedness(counts_tensor)  # (num_layers,)

        # log per-layer balancedness
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
                'counts': counts_tensor,
                'balancedness': balancedness,
                'total_tokens': counts_tensor.sum(dim=1),
                'step': step,
                'rank': rank,
            },
            filepath,
        )
        logger.info(f'[Expert Statistics] Expert distribution dumped to {filepath}, shape={list(counts_tensor.shape)}')

        if envs.expert_dump_visualize:
            from lmdeploy.pytorch.tools.utils import visualize_expert_distribution
            visualize_expert_distribution(filepath)

        return filepath


_global_recorder = None


def get_expert_distribution_recorder():
    global _global_recorder

    if _global_recorder is None:
        if envs.dump_expert_distribution:
            _global_recorder = _ExpertsDistributionRecorderImpl()
        else:
            _global_recorder = _ExpertsDistributionRecorderNoOp()

    return _global_recorder
