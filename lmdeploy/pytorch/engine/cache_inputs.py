# Copyright (c) OpenMMLab. All rights reserved.
"""Per-forward prefix-cache checkpoint operations."""
from dataclasses import dataclass

import torch

StateCacheCopyPlan = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class CacheCheckpointInputs:
    """Cache-copy plans consumed by exactly one model forward.

    KV plans contain physical GPU block-offset pairs at scheduler-block
    granularity with shape ``[2, N]``. They move to the cache device with the
    other forward payloads. State plans remain compact host integer sequences
    because ``StateCacheEngine`` schedules those copies from the host.

    Checkpoint operations deliberately do not belong to ``ModelInputs``:
    retained model inputs participate in decode-step merge, reindex, and
    advance operations, while these plans must execute only once.
    """

    kv_restore_plan: torch.Tensor | None = None
    kv_save_plan: torch.Tensor | None = None
    state_restore_plan: StateCacheCopyPlan | None = None
    state_save_plan: StateCacheCopyPlan | None = None

    @torch.inference_mode()
    def to_device(self, device: str, non_blocking: bool = False):
        """Move device-consumed plans while retaining host-scheduled plans."""
        kv_restore_plan = self.kv_restore_plan
        if kv_restore_plan is not None:
            kv_restore_plan = kv_restore_plan.to(device, non_blocking=non_blocking)
        kv_save_plan = self.kv_save_plan
        if kv_save_plan is not None:
            kv_save_plan = kv_save_plan.to(device, non_blocking=non_blocking)

        return CacheCheckpointInputs(
            kv_restore_plan=kv_restore_plan,
            kv_save_plan=kv_save_plan,
            state_restore_plan=self.state_restore_plan,
            state_save_plan=self.state_save_plan,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record forward-stream use of device-consumed plans."""
        for plan in (self.kv_restore_plan, self.kv_save_plan):
            if plan is not None and plan.is_cuda:
                plan.record_stream(stream)
