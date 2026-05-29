# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.distributed as dist


class DistGatherScalar:
    """Distribute value gather."""

    def __init__(self, val, size: int, device: str = 'cpu', group: dist.ProcessGroup = None):
        self.val = val
        self.device = device
        self.group = group

        self.all_vals = torch.tensor([val] * size, device=device)
        self.worker = dist.all_gather_into_tensor(self.all_vals,
                                                  self.all_vals.new_tensor([val]),
                                                  group=group,
                                                  async_op=True)

    async def async_wait(self, timeout: float = 0.001):
        while not self.worker.is_completed():
            await asyncio.sleep(timeout)
        self.worker.wait()
        return self.all_vals


@dataclass
class DPForwardMeta:
    """Local scalar metadata to be gathered across DP ranks."""

    is_decoding: bool
    is_dummy: bool
    num_tokens: int
    is_sleeping: bool
    batch_size: int
    has_non_last_chunk: bool | None = None
    draft_num_tokens: int | None = None
    enable_microbatch: bool | None = None

    BASE_FIELDS: ClassVar[tuple[str, ...]] = (
        'is_decoding',
        'is_dummy',
        'num_tokens',
        'is_sleeping',
        'batch_size',
    )
    SPEC_FIELDS: ClassVar[tuple[str, ...]] = (
        'has_non_last_chunk',
        'draft_num_tokens',
    )
    MICRO_BATCH_FIELDS: ClassVar[tuple[str, ...]] = ('enable_microbatch', )

    @classmethod
    def field_names(cls, *, is_spec_enabled: bool, is_microbatch_enabled: bool) -> tuple[str, ...]:
        """Get the serialization schema for DP forward metadata."""
        field_names = cls.BASE_FIELDS
        if is_spec_enabled:
            field_names += cls.SPEC_FIELDS
        if is_microbatch_enabled:
            field_names += cls.MICRO_BATCH_FIELDS
        return field_names

    def values(self, *, is_spec_enabled: bool, is_microbatch_enabled: bool) -> list[int]:
        """Serialize local metadata for scalar all-gather."""
        raw_values = {
            'is_decoding': int(self.is_decoding),
            'is_dummy': int(self.is_dummy),
            'num_tokens': self.num_tokens,
            'is_sleeping': int(self.is_sleeping),
            'batch_size': self.batch_size,
            'has_non_last_chunk': int(self.has_non_last_chunk or False),
            'draft_num_tokens': self.draft_num_tokens if self.draft_num_tokens is not None else self.num_tokens,
            'enable_microbatch': int(self.enable_microbatch or False),
        }
        return [
            raw_values[name]
            for name in self.field_names(
                is_spec_enabled=is_spec_enabled,
                is_microbatch_enabled=is_microbatch_enabled,
            )
        ]


@dataclass
class GatheredDPForwardMeta:
    """Named DP metadata after scalar all-gather."""

    is_decoding: torch.Tensor
    is_dummy: torch.Tensor
    num_tokens: torch.Tensor
    is_sleeping: torch.Tensor
    batch_size: torch.Tensor
    has_non_last_chunk: torch.Tensor | None = None
    draft_num_tokens: torch.Tensor | None = None
    enable_microbatch: torch.Tensor | None = None

    @classmethod
    def from_values(cls, values: torch.Tensor, *, is_spec_enabled: bool, is_microbatch_enabled: bool):
        """Deserialize gathered scalar values into named tensors."""
        field_names = DPForwardMeta.field_names(
            is_spec_enabled=is_spec_enabled,
            is_microbatch_enabled=is_microbatch_enabled,
        )
        columns = dict(zip(field_names, values.unbind(dim=1)))
        return cls(**columns)

    @property
    def global_is_decoding(self):
        """Whether all DP ranks are decoding."""
        return self.is_decoding.all().item()

    @property
    def is_all_dummy(self):
        """Whether all DP ranks have dummy inputs."""
        return self.is_dummy.all().item()

    @property
    def is_all_sleeping(self):
        """Whether all DP ranks are sleeping."""
        return self.is_sleeping.all().item()

    @property
    def all_batch_sizes(self):
        """Batch size on each DP rank."""
        return self.batch_size.tolist()

    @property
    def all_num_tokens(self):
        """Main model token count on each DP rank."""
        return self.num_tokens.tolist()

    @property
    def all_draft_num_tokens(self):
        """Draft model token count on each DP rank."""
        assert self.draft_num_tokens is not None
        return self.draft_num_tokens.tolist()

    @property
    def dp_has_non_last_chunk(self):
        """Whether any DP rank is handling a non-last chunk."""
        assert self.has_non_last_chunk is not None
        return bool(self.has_non_last_chunk.any().item())

    @property
    def global_enable_microbatch(self):
        """Whether all DP ranks enable microbatch for this forward."""
        assert self.enable_microbatch is not None
        return self.enable_microbatch.all().item()
