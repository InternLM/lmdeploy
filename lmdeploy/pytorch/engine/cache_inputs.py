# Copyright (c) OpenMMLab. All rights reserved.
"""Prefix-cache checkpoint operations owned by one forward."""
from dataclasses import dataclass

import torch

StateCacheCopyPlan = tuple[tuple[int, ...], tuple[int, ...]]


@dataclass(frozen=True)
class CacheCheckpointInputs:
    """Cache-copy operations and owner completion identity for one forward.

    KV plans contain physical GPU block-offset pairs at scheduler-block
    granularity with shape ``[2, N]``. They move to the cache device with the
    other forward payloads. Unconditional state plans remain host integer
    sequences. Decode plans move to the device for acceptance and copying.

    Copy plans execute only once; model-consumed compact layouts and bindings
    travel with ModelInputs/ModelInputsDelta instead.
    """

    kv_restore_plan: torch.Tensor | None = None
    kv_save_plan: torch.Tensor | None = None
    state_restore_plan: StateCacheCopyPlan | None = None
    state_save_plan: StateCacheCopyPlan | None = None
    # Columns are (batch row, boundary, source slot, destination slot).
    # Decode copies are conditional on this forward's accepted prefix.
    decode_save_plan: torch.Tensor | None = None
    decode_save_id: int | None = None
    # Host-only checkpoint boundaries consumed by InputsMaker before submission.
    # None leaves checkpoint planning disabled; zero skips a row.
    ssm_checkpoint_steps: tuple[int, ...] | None = None
    # Host owner's access identity, echoed only after actual GPU completion.
    # Fresh on each dispatch, including delta-only decode; never persistent IDs.
    ssm_state_access_id: int | None = None

    def __post_init__(self):
        if self.ssm_checkpoint_steps is not None:
            if (not isinstance(self.ssm_checkpoint_steps, tuple)
                    or any(type(step) is not int or step < 0 for step in self.ssm_checkpoint_steps)):
                raise ValueError('Compact checkpoint boundaries require nonnegative host integers.')
        if self.ssm_state_access_id is not None and (type(self.ssm_state_access_id) is not int
                                                  or self.ssm_state_access_id <= 0):
            raise ValueError('Compact access identity must be a positive host integer.')
        if ((self.ssm_checkpoint_steps is not None or self.ssm_state_access_id is not None)
                and (self.state_restore_plan is not None or self.state_save_plan is not None)):
            raise ValueError('Compact execution cannot carry legacy state-copy plans.')

    @torch.inference_mode()
    def to_device(self, device: str, non_blocking: bool = False):
        """Move device-consumed plans while retaining host-scheduled plans."""
        if self.ssm_checkpoint_steps is not None:
            raise ValueError('Unbound compact checkpoint planning must not reach a worker.')
        kv_restore_plan = self.kv_restore_plan
        if kv_restore_plan is not None:
            kv_restore_plan = kv_restore_plan.to(device, non_blocking=non_blocking)
        kv_save_plan = self.kv_save_plan
        if kv_save_plan is not None:
            kv_save_plan = kv_save_plan.to(device, non_blocking=non_blocking)
        decode_save_plan = self.decode_save_plan
        if decode_save_plan is not None:
            decode_save_plan = decode_save_plan.to(device, non_blocking=non_blocking)

        return CacheCheckpointInputs(
            kv_restore_plan=kv_restore_plan,
            kv_save_plan=kv_save_plan,
            state_restore_plan=self.state_restore_plan,
            state_save_plan=self.state_save_plan,
            decode_save_plan=decode_save_plan,
            decode_save_id=self.decode_save_id,
            ssm_state_access_id=self.ssm_state_access_id,
        )

    def record_stream(self, stream: torch.cuda.Stream) -> None:
        """Record forward-stream use of device-consumed plans."""
        for plan in (self.kv_restore_plan, self.kv_save_plan, self.decode_save_plan):
            if plan is not None and plan.is_cuda:
                plan.record_stream(stream)
