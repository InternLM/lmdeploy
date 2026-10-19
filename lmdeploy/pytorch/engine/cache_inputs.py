# Copyright (c) OpenMMLab. All rights reserved.
"""Per-forward prefix-cache checkpoint operations."""
from dataclasses import dataclass

import torch

from lmdeploy.pytorch.compact_state import CompactStatePrefill

StateCacheCopyPlan = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class CacheCheckpointInputs:
    """Cache operations and indexed layout consumed by one model forward.

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
    # Host-only checkpoint boundaries consumed by InputsMaker before submission.
    # None leaves checkpoint planning disabled; zero skips a row.
    compact_checkpoint_steps: tuple[int, ...] | None = None
    compact_prefill: CompactStatePrefill | None = None
    # Host owner's access identity, echoed only after actual GPU completion.
    # Fresh on each dispatch, including delta-only decode; never persistent IDs.
    compact_access_id: int | None = None
    # One-forward source binding: -1 retains runtime, 0 resets, >0 reads a CP.
    compact_source_ids: torch.Tensor | None = None
    # One-forward single-token AR output reservation; -1 means ordinary bank.
    compact_decode_ids: torch.Tensor | None = None

    def __post_init__(self):
        if self.compact_checkpoint_steps is not None:
            if (not isinstance(self.compact_checkpoint_steps, tuple)
                    or any(type(step) is not int or step < 0 for step in self.compact_checkpoint_steps)):
                raise ValueError('Compact checkpoint boundaries require nonnegative host integers.')
        if self.compact_access_id is not None and (type(self.compact_access_id) is not int
                                                  or self.compact_access_id <= 0):
            raise ValueError('Compact access identity must be a positive host integer.')
        for ids in (self.compact_source_ids, self.compact_decode_ids):
            if ids is not None and (ids.ndim != 1 or ids.dtype != torch.int64 or not ids.is_contiguous()):
                raise ValueError('Compact source/output binding requires contiguous per-request int64 IDs.')
        if self.compact_decode_ids is not None and (self.compact_prefill is not None
                                                   or self.compact_source_ids is not None):
            raise ValueError('Compact decode destinations cannot carry a prefill/source binding.')
        if ((self.compact_checkpoint_steps is not None or self.compact_prefill is not None
             or self.compact_access_id is not None
             or self.compact_source_ids is not None or self.compact_decode_ids is not None)
                and (self.state_restore_plan is not None or self.state_save_plan is not None)):
            raise ValueError('Compact execution cannot carry legacy state-copy plans.')

    @torch.inference_mode()
    def to_device(self, device: str, non_blocking: bool = False):
        """Move device-consumed plans while retaining host-scheduled plans."""
        if self.compact_checkpoint_steps is not None:
            raise ValueError('Unbound compact checkpoint planning must not reach a worker.')
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
            compact_prefill=(self.compact_prefill.to_device(device, non_blocking=non_blocking)
                             if self.compact_prefill is not None else None),
            compact_access_id=self.compact_access_id,
            compact_decode_ids=(self.compact_decode_ids.to(device, non_blocking=non_blocking)
                                if self.compact_decode_ids is not None else None),
            compact_source_ids=(self.compact_source_ids.to(device, non_blocking=non_blocking)
                                if self.compact_source_ids is not None else None),
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record forward-stream use of device-consumed plans."""
        for plan in (self.kv_restore_plan, self.kv_save_plan, self.compact_source_ids, self.compact_decode_ids):
            if plan is not None and plan.is_cuda:
                plan.record_stream(stream)
        if self.compact_prefill is not None:
            self.compact_prefill.record_stream(stream)
