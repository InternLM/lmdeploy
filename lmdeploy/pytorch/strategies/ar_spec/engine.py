# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import CacheConfig, SchedulerConfig

from ..base.engine import EngineStrategy


class ARSpecEngineStrategy(EngineStrategy):
    """AR Engine Strategy."""

    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, num_spec_tokens: int,
                 draft_query_len: int | None = None) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.num_spec_tokens = num_spec_tokens
        self.draft_query_len = draft_query_len

    def get_prealloc_size(self, is_decoding: bool):
        """Get prealloc_size."""
        if not is_decoding:
            return self.num_spec_tokens if self.draft_query_len is None else self.draft_query_len
        prealloc = self.scheduler_config.prefill_interval * (1 + self.num_spec_tokens)
        if self.draft_query_len is not None:
            # Even interval=1 must cover the verifier followed by a full query.
            prealloc = max(prealloc, self.get_num_required_tokens())
        return prealloc

    def get_num_loops(self, is_decoding: bool) -> int:
        """Get num_loops."""
        return self.scheduler_config.prefill_interval if is_decoding else 1

    def get_num_decode_tokens(self) -> int:
        """Get num_decode_tokens."""
        return self.num_spec_tokens + 1

    def get_num_required_tokens(self) -> int:
        """Get num_required_tokens."""
        if self.draft_query_len is not None:
            # The scheduler can be one verifier step behind. Block drafts
            # materialize that full step, then write Q NEW positions. Unlike
            # the shifted AR/MTP draft, DFlash (Q=N+1) needs one extra slot.
            return self.num_spec_tokens + 1 + self.draft_query_len
        return 2 * self.num_spec_tokens + 1
