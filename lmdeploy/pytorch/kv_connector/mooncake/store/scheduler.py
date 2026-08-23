# Copyright (c) OpenMMLab. All rights reserved.
"""Scheduler-side asynchronous lookup for the Mooncake Store connector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .data import MooncakeStoreConnectorMetadata, build_prefix_block_hashes
from .worker import LookupKeyClient

if TYPE_CHECKING:
    from lmdeploy.pytorch.config import CacheConfig
    from lmdeploy.pytorch.messages import SchedulerSequence
    from lmdeploy.pytorch.paging.scheduler import SchedulerOutput


@dataclass
class _RequestHashTracker:
    """Stable hashes already computed for an append-only request prefix."""

    adapter_identity: str
    block_hashes: tuple[bytes, ...] = ()


class MooncakeStoreScheduler:
    """Scheduler-side component of the Mooncake Store connector."""

    def __init__(self, cache_config: CacheConfig) -> None:
        kv_transfer_config = cache_config.kv_transfer_config
        if kv_transfer_config is None or not kv_transfer_config.is_kv_transfer_instance:
            raise ValueError('MooncakeStoreScheduler requires an enabled kv_transfer_config')
        if cache_config.states_shapes:
            raise ValueError('Mooncake Store does not support linear-attention state caches')

        lookup_async = kv_transfer_config.kv_connector_extra_config.get('lookup_async', True)
        if lookup_async is not True:
            raise ValueError('Mooncake Store requires lookup_async=true')

        self._cache_config = cache_config
        self._kv_transfer_config = kv_transfer_config
        self.lookup_async = True
        self.client = LookupKeyClient(cache_config)
        self._request_hash_trackers: dict[int, _RequestHashTracker] = {}

    def get_num_new_matched_tokens(
        self,
        request: SchedulerSequence,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """Poll the remote prefix beyond the locally matched GPU prefix."""
        if not self._kv_transfer_config.is_kv_consumer:
            return 0, False
        if (not request.history_multimodals.empty()
                or len(request.history_embeddings) > 0):
            return 0, False

        block_size = self._cache_config.block_size
        token_len = request.get_prefix_cache_max_match_step()
        token_len = token_len // block_size * block_size
        if token_len < block_size or num_computed_tokens >= token_len:
            return 0, False

        req_id = int(request.seq_id)
        block_hashes = self._get_request_block_hashes(
            request,
            req_id,
            token_len,
            block_size,
        )
        remote_token_len = self.client.lookup(
            req_id,
            token_len,
            block_hashes,
            non_block=self.lookup_async,
        )
        if remote_token_len is None:
            return None, False

        num_external_tokens = max(0, int(remote_token_len) - int(num_computed_tokens))
        if num_external_tokens == 0:
            return 0, False
        return num_external_tokens, True

    def _get_request_block_hashes(
        self,
        request: SchedulerSequence,
        req_id: int,
        token_len: int,
        block_size: int,
    ) -> tuple[bytes, ...]:
        """Return cached hashes, extending them only for newly full blocks."""
        num_blocks = token_len // block_size
        adapter_identity = request.adapter_name or ''
        tracker = self._request_hash_trackers.get(req_id)
        if tracker is None or tracker.adapter_identity != adapter_identity:
            tracker = _RequestHashTracker(adapter_identity=adapter_identity)
            self._request_hash_trackers[req_id] = tracker

        if num_blocks <= len(tracker.block_hashes):
            return tracker.block_hashes[:num_blocks]

        tracker.block_hashes = build_prefix_block_hashes(
            request.all_ids[:token_len],
            block_size,
            extra_identity=adapter_identity,
            previous_hashes=tracker.block_hashes,
        )
        return tracker.block_hashes

    def update_state_after_alloc(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
        num_external_tokens: int,
    ) -> None:
        """Record no allocation state until external loading is implemented."""
        return None

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> MooncakeStoreConnectorMetadata:
        """Build empty, serializable metadata for the current engine step."""
        return MooncakeStoreConnectorMetadata()

    def on_new_request(self, request: SchedulerSequence) -> None:
        """Drop stale request-local hashes before first scheduling."""
        self._request_hash_trackers.pop(int(request.seq_id), None)
        return None

    def is_lookup_pending(self, request_id: int) -> bool:
        """Return whether the request's lookup Future is still running."""
        return self.client.is_pending(int(request_id))

    def cancel_lookup(self, request_id: int) -> None:
        """Cancel only the current lookup, retaining incremental hashes."""
        self.client.discard(int(request_id))

    def update_connector_output(self, connector_output: Any) -> None:
        """Consume no worker output until asynchronous I/O is implemented."""
        return None

    def request_finished(
        self,
        request: SchedulerSequence,
        block_ids: Sequence[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Do not take ownership of finished request blocks yet."""
        req_id = int(request.seq_id)
        self.client.discard(req_id)
        self._request_hash_trackers.pop(req_id, None)
        return False, None

    def shutdown(self) -> None:
        """Cancel pending lookups and release the scheduler client."""
        self.client.close()
        self._request_hash_trackers.clear()
