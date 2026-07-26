# Copyright (c) OpenMMLab. All rights reserved.
"""Host-side lookup primitives for SSM prefix-cache checkpoints.

``BlockTrie`` owns trie topology, KV references, and prefix-match sequence
mutation.  ``StateCheckpointLifecycle`` owns recurrent-state slots and their
publication/pinning lifecycle.  This module owns the auxiliary sparse index
used to find checkpoint candidates and the immutable metadata used to verify a
candidate exactly.

The sparse key ``(adapter, step, last_block_hash)`` is only a coarse lookup.
``verify()`` compares the complete token and multimodal identities before a
checkpoint may be restored.  Keeping that distinction inside one small class
makes the matching contract visible without mixing it with cache allocation
and checkpoint lifecycle code.
"""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from lmdeploy.pytorch.messages import PrefixCacheExtraHashes, SchedulerSequence

if TYPE_CHECKING:
    from .node import Node


StateCheckpointKey: TypeAlias = tuple[str, int, int]
BlockKeyMaker: TypeAlias = Callable[[np.ndarray, PrefixCacheExtraHashes], int]


class StateCheckpointVerifyStatus(enum.Enum):
    """Outcome of exact verification after sparse checkpoint lookup."""

    HIT = enum.auto()
    REQUEST_MISMATCH = enum.auto()
    STALE_INDEX_ENTRY = enum.auto()
    STALE_CHECKPOINT = enum.auto()


@dataclass(frozen=True)
class StateCheckpointMatchData:
    """Immutable host metadata for proving and applying a checkpoint hit."""

    token_ids: np.ndarray
    multimodal_hashes: PrefixCacheExtraHashes
    blocks: np.ndarray


@dataclass
class StateCheckpointVerifyResult:
    """Verified checkpoint candidate details."""

    status: StateCheckpointVerifyStatus
    reason: str = ''
    matched_blocks: np.ndarray | None = None


def freeze_state_checkpoint_match_data(token_ids: np.ndarray,
                                       multimodal_hashes: PrefixCacheExtraHashes,
                                       blocks: np.ndarray):
    """Make already-owned checkpoint identity arrays read-only."""
    token_ids.flags.writeable = False
    blocks.flags.writeable = False
    return StateCheckpointMatchData(token_ids=token_ids,
                                    multimodal_hashes=multimodal_hashes,
                                    blocks=blocks)


def make_request_multimodal_identity(seq: SchedulerSequence, step: int):
    """Get the exact multimodal identity for a request prefix."""
    hashes = ((meta.start, meta.end, meta.modality, meta.content_hash) for meta in seq.prefix_cache.metas
              if meta.start < step and meta.end > 0)
    return tuple(sorted(hashes))


def make_node_multimodal_identity(nodes: tuple['Node', ...], block_size: int):
    """Recover one exact, deduplicated multimodal identity from trie nodes."""
    hashes = (extra_hash for block_id, block_node in enumerate(nodes) for extra_hash in block_node.extra_hashes
              if extra_hash[0] // block_size == block_id)
    return tuple(sorted(hashes))


class StateCheckpointIndex:
    """Sparse checkpoint candidates plus their exact verification contract.

    The index is auxiliary state: trie nodes remain the source of truth for KV
    and recurrent-state ownership.  Consequently callers must remove or
    release stale candidates reported by :meth:`verify`.
    """

    def __init__(self, block_size: int, make_block_key: BlockKeyMaker):
        self.block_size = block_size
        self._make_block_key = make_block_key
        self._buckets: dict[StateCheckpointKey, list[Node]] = {}
        self._steps_by_adapter: dict[str, set[int]] = {}

    def make_request_key(self, seq: SchedulerSequence, step: int) -> StateCheckpointKey:
        """Make the sparse lookup key for one request checkpoint step."""
        start = step - self.block_size
        tokens = seq.history_cache[start:step]
        extra_hashes = seq.get_prefix_cache_extra_hashes(start, step)
        return (seq.adapter_name, step, self._make_block_key(tokens, extra_hashes))

    @staticmethod
    def make_node_key(node: 'Node') -> StateCheckpointKey:
        """Make the canonical sparse-index key for a checkpoint node."""
        return (node.adapter_name, node.num_matched, node.hash_key)

    def add(self, node: 'Node'):
        """Add a prevalidated ready checkpoint node."""
        key = self.make_node_key(node)
        nodes = self._buckets.setdefault(key, [])
        if not any(indexed_node is node for indexed_node in nodes):
            nodes.append(node)
        self._steps_by_adapter.setdefault(node.adapter_name, set()).add(node.num_matched)

    def remove_entry(self, node: 'Node', key: StateCheckpointKey):
        """Remove a node from one sparse-index bucket."""
        nodes = self._buckets.get(key)
        if nodes is None:
            return False

        old_len = len(nodes)
        nodes[:] = [indexed_node for indexed_node in nodes if indexed_node is not node]
        if len(nodes) == old_len:
            return False
        if len(nodes) == 0:
            self._buckets.pop(key)
        self._refresh_step(key[0], key[1])
        return True

    def remove(self, node: 'Node'):
        """Remove a node from every sparse-index bucket."""
        removed = False
        for key in list(self._buckets):
            removed = self.remove_entry(node, key) or removed
        return removed

    def _refresh_step(self, adapter_name: str, step: int):
        """Drop an adapter step when no indexed checkpoint still owns it."""
        steps = self._steps_by_adapter.get(adapter_name)
        if steps is None or step not in steps:
            return
        has_step = any(key[0] == adapter_name and key[1] == step for key in self._buckets)
        if not has_step:
            steps.remove(step)
        if len(steps) == 0:
            self._steps_by_adapter.pop(adapter_name)

    def candidate_steps(self, adapter_name: str, after_step: int, max_step: int):
        """Return possible checkpoint steps from deepest to shallowest."""
        steps = self._steps_by_adapter.get(adapter_name, ())
        return sorted((step for step in steps if after_step < step <= max_step), reverse=True)

    def num_steps(self, adapter_name: str):
        """Return the number of indexed steps for one adapter."""
        return len(self._steps_by_adapter.get(adapter_name, ()))

    def candidates(self, key: StateCheckpointKey):
        """Snapshot a bucket so stale-entry cleanup is safe while iterating."""
        return tuple(self._buckets.get(key, ()))

    def unique_nodes(self):
        """Iterate indexed nodes once, even if stale duplicate keys exist."""
        seen_nodes = set()
        for nodes in self._buckets.values():
            for node in nodes:
                node_id = id(node)
                if node_id in seen_nodes:
                    continue
                seen_nodes.add(node_id)
                yield node

    def verify(self,
               seq: SchedulerSequence,
               node: 'Node',
               index_key: StateCheckpointKey,
               path_is_current: bool):
        """Prove that a sparse candidate is an exact, current prefix hit."""
        if node.state_idx < 0 or not node.state_ready:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint is not ready')

        step = node.num_matched
        if step <= 0:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason=f'invalid checkpoint step: {step}')

        match_data = node.state_match_data
        if match_data is None:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint exact-match metadata is missing')
        if len(match_data.blocks) * self.block_size != step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint exact-match path has the wrong length')
        if not path_is_current:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint owner is detached from its cached path')

        # A current path has already passed BlockTrie's topology-invalidation
        # contract.  Rewalking every ancestor here would make matching linear
        # in prefix blocks in Python.
        if index_key != self.make_node_key(node):
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_INDEX_ENTRY,
                                               reason='checkpoint is indexed under a stale key')
        if node.adapter_name != seq.adapter_name:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_INDEX_ENTRY,
                                               reason='checkpoint adapter differs from lookup adapter')

        max_step = ((seq.num_valid_ids - 1) // self.block_size) * self.block_size
        if step > max_step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint is longer than this request')
        if seq.clamp_prefix_cache_match_step(step) != step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint would stop inside a multimodal span')
        if not np.array_equal(seq.history_cache[:step], match_data.token_ids):
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint token identity differs from this request')
        if make_request_multimodal_identity(seq, step) != match_data.multimodal_hashes:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint multimodal identity differs from this request')
        if seq.return_routed_experts:
            block_node = node
            while block_node.parent is not None:
                if block_node.routed_experts is None:
                    return StateCheckpointVerifyResult(
                        StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                        reason=f'routed experts missing at step {block_node.num_matched}',
                    )
                block_node = block_node.parent

        return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.HIT, matched_blocks=match_data.blocks)
