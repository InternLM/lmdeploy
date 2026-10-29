# Copyright (c) OpenMMLab. All rights reserved.
"""Own provisional decode snapshots from dispatch through accepted
publication."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from lmdeploy.pytorch.kv_connector.base import KVConnectorStepInput
from lmdeploy.pytorch.long_context import get_decode_save_step
from lmdeploy.pytorch.messages import SchedulerSequence

from .compact_state import SSMStateManager, StateReference

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector.base import KVConnectorBase
    from lmdeploy.pytorch.model_inputs import ModelInputs, ModelInputsDelta

    from .block_manager.base_block_manager import BaseBlockManager
    from .block_trie import BlockTrie
    from .kv_save_coordinator import KVSaveCoordinator
    from .state_manager import StateManager


@dataclass
class _Snapshot:
    seq: SchedulerSequence
    input_end_pos: int
    step: int
    source: int
    local_slot: int
    store_slot: int
    blocks: np.ndarray
    copy_index: int
    reference: StateReference | None = None


class DecodeCheckpointCoordinator:
    """Reserve before dispatch; attach identity only after accepted output.

    Adjacent prefetched forwards may try the same boundary. They own separate slots, and the worker only copies when the
    boundary belongs to that forward's retained, accepted state range. No reservation publishes a hash.
    """

    def __init__(self, *, block_manager: 'BaseBlockManager', block_trie: 'BlockTrie',
                 state_manager: 'StateManager', connector: 'KVConnectorBase | None',
                 save_coordinator: 'KVSaveCoordinator'):
        self.block_manager = block_manager
        self.block_trie = block_trie
        self.state_manager = state_manager
        self.connector = connector
        self.save_coordinator = save_coordinator
        self._next_id = 0
        self._batches: dict[int, list[_Snapshot]] = {}

    def prepare(self, running: list[SchedulerSequence], inputs: 'ModelInputs | None',
                delta: 'ModelInputsDelta | None', *, access_id: int | None = None) -> tuple[int, torch.Tensor] | None:
        """Reserve candidates using a conservative range for one-step-stale CPU
        progress."""
        if inputs is not None:
            if inputs.is_dummy or (inputs.is_chunk and not inputs.is_last_chunk):
                return None
        elif delta is None:
            return None
        snapshots = []
        copies = []
        checkpoints = self.block_trie.state_checkpoints
        compact = isinstance(self.state_manager, SSMStateManager)
        for row, seq in enumerate(running):
            if seq.logical_state <= 0:
                continue
            if inputs is not None:
                minimum = maximum = seq.input_end_pos
            else:
                minimum = seq.num_history_ids + 1
                # The previous forward may still be running. num_all_ids
                # includes its query; this delta contributes one more query.
                maximum = min(seq.num_all_ids + delta.max_q_seqlen,
                              len(seq.logical_blocks) * self.block_trie.block_size)
            step = get_decode_save_step(seq, minimum, maximum)
            if step is None:
                continue
            pending = seq.prefix_cache.pending_save
            if compact:
                owner = self.state_manager
                if (owner.capacity.max_update_tokens == 1 and not seq._seq_meta.prefix_cache_decode_save_first
                        and not seq._seq_meta.enable_mooncake_store):
                    continue  # InputsMaker retains the direct-output AR interval path.
                local = (self.block_trie.enabled and owner.capacity.checkpoint_budget > 0
                         and not (pending.is_pending and pending.step == step))
                store = (self.connector is not None and owner.get_num_free_store_saves() > 0
                         and not seq.prefix_cache.decode_store_save_pending)
                if not local and not store:
                    continue
                if owner.get_num_free_optional() == 0:
                    continue
                reference = owner.reserve_decode_snapshot(access_id, owner.bank_for(seq), step)
                blocks = seq.logical_blocks.get_real_blocks()[:step // self.block_trie.block_size].copy()
                self.block_manager.pin_logical_blocks(blocks)
                snapshots.append(_Snapshot(seq, seq.input_end_pos, step, int(seq.logical_state),
                                           reference.unit_id if local else -1, reference.unit_id if store else -1,
                                           blocks, len(copies), reference))
                # The worker resolves the exact source unit from its verified output table.
                copies.append((row, step, -1, reference.unit_id))
                continue
            local_slot = (-1 if pending.is_pending and pending.step == step
                          else checkpoints.reserve_decode_snapshot())
            store_slot = -1 if self.connector is None else self.connector.reserve_decode_snapshot(seq)
            if local_slot < 0 and store_slot < 0:
                continue
            blocks = seq.logical_blocks.get_real_blocks()[:step // self.block_trie.block_size].copy()
            source = int(seq.logical_state)
            self.state_manager.retain_state(source)
            self.block_manager.pin_logical_blocks(blocks)
            snapshots.append(_Snapshot(seq, seq.input_end_pos, step, source, local_slot,
                                       store_slot, blocks, len(copies)))
            for slot in (local_slot, store_slot):
                if slot >= 0:
                    copies.append((row, step, source, slot))
        if not copies:
            return None
        batch_id = self._next_id
        self._next_id += 1
        self._batches[batch_id] = snapshots
        return batch_id, torch.tensor(copies, dtype=torch.int64).T.contiguous()

    def finish(self, batch_id: int, copied: torch.Tensor | None) -> None:
        """Publish confirmed snapshots and transfer Store ownership without
        waiting for I/O."""
        snapshots = self._batches.pop(batch_id)
        mask = copied.tolist() if copied is not None else ()
        checkpoints = self.block_trie.state_checkpoints
        try:
            for snapshot in snapshots:
                seq, step = snapshot.seq, snapshot.step
                accepted = (bool(mask) and mask[snapshot.copy_index]
                            and seq.input_end_pos == snapshot.input_end_pos and seq.logical_state == snapshot.source
                            and step <= seq.num_history_ids
                            and step + seq._seq_meta.prefix_cache_token_lookahead <= seq.num_valid_ids)
                if accepted:
                    reference = snapshot.reference
                    if snapshot.local_slot >= 0:
                        if reference is None:
                            self.block_trie.allocate(seq, token_limit=step)
                            if checkpoints.publish_decode_snapshot(seq, step, snapshot.local_slot):
                                snapshot.local_slot = -1
                        elif checkpoints.publish_decode_snapshot(seq, step, reference):
                            # Publication transferred the reservation to the trie. Store acquires its independent
                            # reader from that immutable unit before any later dispatch can change the node.
                            reference = seq.prefix_cache.decode_checkpoint_node.state_checkpoint.state_reference
                    if snapshot.store_slot >= 0:
                        metadata = self.connector.queue_decode_save(KVConnectorStepInput(
                            running=[seq], connector_token_lens=(step,),
                            connector_state_ids=(snapshot.store_slot,) if reference is None else (),
                            connector_snapshot_ids=(snapshot.store_slot,) if reference is None else (),
                            compact_state_references=(reference,) if reference is not None else (),
                            connector_block_ids=(tuple(int(x) for x in self.block_manager.get_block_table(seq)),),
                            connector_logical_block_ids=(tuple(int(x) for x in seq.logical_blocks.get_real_blocks()),),
                            is_decode_checkpoint=True))
                        if metadata is not None:
                            self.save_coordinator.acquire(metadata)
                            snapshot.store_slot = -1
        finally:
            for snapshot in snapshots:
                self._release(snapshot)

    def clear(self) -> None:
        """Release abandoned reservations only after worker queues have
        drained."""
        for snapshots in self._batches.values():
            for snapshot in snapshots:
                self._release(snapshot)
        self._batches.clear()

    def _release(self, snapshot: _Snapshot) -> None:
        if snapshot.reference is not None:
            self.state_manager.release_reference(snapshot.reference)
            self.block_manager.release_logical_blocks(snapshot.blocks)
            return
        if snapshot.local_slot >= 0:
            self.state_manager.free_checkpoint_state(snapshot.local_slot)
        if snapshot.store_slot >= 0:
            self.connector.release_decode_snapshot(snapshot.store_slot)
        self.state_manager.free_state(snapshot.source)
        self.block_manager.release_logical_blocks(snapshot.blocks)
