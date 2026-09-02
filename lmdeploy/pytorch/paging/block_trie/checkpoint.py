# Copyright (c) OpenMMLab. All rights reserved.
"""Host-side lookup primitives for SSM prefix-cache checkpoints.

``BlockTrie`` owns trie topology, KV references, and prefix-match sequence
mutation.  ``StateCheckpointLifecycle`` owns recurrent-state slots and their
publication/pinning lifecycle.  This module owns the auxiliary sparse index
used to find checkpoint candidates and the immutable metadata used to verify a
candidate exactly.

The sparse key ``(adapter, step, tail_hash)`` is only a coarse lookup.
``verify_candidate()`` compares the complete token and multimodal identities
before a checkpoint may be restored. Keeping both operations here makes the
matching contract visible without mixing it with cache allocation or
checkpoint lifecycle code.
"""

import enum
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import numpy as np

from lmdeploy.pytorch.messages import SchedulerSequence
from lmdeploy.pytorch.prefix_cache_state import PrefixCacheExtraIdentity

if TYPE_CHECKING:
    from .node import Node


StateCheckpointKey: TypeAlias = tuple[str, int, int]
BlockHasher: TypeAlias = Callable[[np.ndarray, PrefixCacheExtraIdentity], int]


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
    extra_identity: PrefixCacheExtraIdentity
    block_ids: np.ndarray
    tail_hash: int


@dataclass
class StateCheckpointVerifyResult:
    """Verified checkpoint candidate details."""

    status: StateCheckpointVerifyStatus
    reason: str = ''
    matched_block_ids: np.ndarray | None = None


def checkpoint_anchor_step(step: int, block_size: int):
    """Return the full-block trie boundary owning an exact checkpoint."""
    return step - step % block_size


def checkpoint_tail_start(step: int, block_size: int):
    """Return the start of the final nonempty range used by the sparse key."""
    return ((step - 1) // block_size) * block_size


def freeze_state_checkpoint_match_data(token_ids: np.ndarray,
                                       extra_identity: PrefixCacheExtraIdentity,
                                       block_ids: np.ndarray,
                                       tail_hash: int):
    """Make already-owned checkpoint identity arrays read-only."""
    token_ids.flags.writeable = False
    block_ids.flags.writeable = False
    return StateCheckpointMatchData(token_ids=token_ids,
                                    extra_identity=extra_identity,
                                    block_ids=block_ids,
                                    tail_hash=tail_hash)


def make_request_multimodal_identity(seq: SchedulerSequence, step: int):
    """Get the exact multimodal identity for a request prefix."""
    return tuple(sorted(span for span in seq.prefix_cache.multimodal_spans if span.start < step and span.end > 0))


class StateCheckpointIndex:
    """Sparse checkpoint candidates plus their exact verification contract.

    The index is auxiliary state: trie nodes remain the source of truth for KV
    and recurrent-state ownership.  Consequently callers must remove or
    release stale candidates reported by :meth:`verify_candidate`.
    """

    def __init__(self, block_size: int, hash_block: BlockHasher):
        self.block_size = block_size
        self._hash_block = hash_block
        self._buckets: dict[StateCheckpointKey, list[Node]] = {}
        self._steps_by_adapter: dict[str, set[int]] = {}

    def make_request_key(self, seq: SchedulerSequence, step: int) -> StateCheckpointKey:
        """Make the sparse lookup key for one request checkpoint step."""
        tail_start = checkpoint_tail_start(step, self.block_size)
        token_ids = seq.history_cache[tail_start:step]
        extra_identity = seq.get_prefix_cache_extra_identity(tail_start, step)
        return (seq.adapter_name, step, self._hash_block(token_ids, extra_identity))

    def make_node_key(self, node: 'Node') -> StateCheckpointKey:
        """Make the sparse-index key stored for a checkpoint node."""
        checkpoint = node.state_checkpoint
        if checkpoint is None:
            raise RuntimeError('Cannot key a node without a state checkpoint.')
        match_data = checkpoint.exact_match_data
        if match_data is None:
            raise RuntimeError('Cannot key a checkpoint without exact-match metadata.')
        return (node.adapter_name, checkpoint.step, match_data.tail_hash)

    def add(self, node: 'Node'):
        """Add a prevalidated published checkpoint node."""
        key = self.make_node_key(node)
        nodes = self._buckets.setdefault(key, [])
        if not any(indexed_node is node for indexed_node in nodes):
            nodes.append(node)
        self._steps_by_adapter.setdefault(node.adapter_name, set()).add(key[1])

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
        self._remove_step_if_empty(key[0], key[1])
        return True

    def remove(self, node: 'Node'):
        """Remove a node from every sparse-index bucket."""
        removed = False
        for key in list(self._buckets):
            removed = self.remove_entry(node, key) or removed
        return removed

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

    def verify_candidate(self,
                         seq: SchedulerSequence,
                         node: 'Node',
                         index_key: StateCheckpointKey):
        """Prove that a sparse candidate is an exact attached prefix hit."""
        checkpoint = node.state_checkpoint
        if checkpoint is None or checkpoint.slot < 0 or not checkpoint.published:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint is not published')

        step = checkpoint.step
        if step <= 0:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason=f'invalid checkpoint step: {step}')
        anchor_step = checkpoint_anchor_step(step, self.block_size)
        is_partial = anchor_step != step
        if is_partial and checkpoint.frozen_block_id < 0:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint partial KV block is missing')
        if not is_partial and checkpoint.frozen_block_id >= 0:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='aligned checkpoint unexpectedly owns a partial KV block')

        match_data = checkpoint.exact_match_data
        if match_data is None:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint exact-match metadata is missing')
        if node.prefix_len != anchor_step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint owner is not its full-block anchor')
        if len(match_data.block_ids) * self.block_size != anchor_step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint exact-match path has the wrong length')
        if len(match_data.token_ids) != step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint exact token identity has the wrong length')
        if not node.is_attached_or_root():
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_CHECKPOINT,
                                               reason='checkpoint owner is detached from its cached path')

        # Monotonic topology guarantees that an attached node still has its
        # original ancestors. Rewalking them here would make matching linear
        # in prefix blocks in Python.
        if index_key != self.make_node_key(node):
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_INDEX_ENTRY,
                                               reason='checkpoint is indexed under a stale key')
        if node.adapter_name != seq.adapter_name:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.STALE_INDEX_ENTRY,
                                               reason='checkpoint adapter differs from lookup adapter')

        max_step = seq.get_prefix_cache_max_candidate_step()
        if step > max_step:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint is longer than this request')
        if not seq.is_prefix_cache_boundary_safe(step):
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint would stop inside a multimodal span')
        if is_partial and seq.return_routed_experts:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='partial checkpoint has no routed-expert tail history')
        if not np.array_equal(seq.history_cache[:step], match_data.token_ids):
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint token identity differs from this request')
        if make_request_multimodal_identity(seq, step) != match_data.extra_identity:
            return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                                               reason='checkpoint multimodal identity differs from this request')
        if seq.return_routed_experts:
            block_node = node
            while block_node.parent is not None:
                if block_node.routed_experts is None:
                    return StateCheckpointVerifyResult(
                        StateCheckpointVerifyStatus.REQUEST_MISMATCH,
                        reason=f'routed experts missing at step {block_node.prefix_len}',
                    )
                block_node = block_node.parent

        return StateCheckpointVerifyResult(StateCheckpointVerifyStatus.HIT, matched_block_ids=match_data.block_ids)

    def _remove_step_if_empty(self, adapter_name: str, step: int):
        """Drop an adapter step when no indexed checkpoint still owns it."""
        steps = self._steps_by_adapter.get(adapter_name)
        if steps is None or step not in steps:
            return
        has_step = any(key[0] == adapter_name and key[1] == step for key in self._buckets)
        if not has_step:
            steps.remove(step)
        if len(steps) == 0:
            self._steps_by_adapter.pop(adapter_name)
