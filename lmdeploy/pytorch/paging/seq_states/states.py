# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.pytorch.messages import MessageStatus, SchedulerSequence, SequenceManager

if TYPE_CHECKING:
    from lmdeploy.pytorch.kv_connector.base import KVConnectorBase
    from lmdeploy.pytorch.paging.block_manager import BaseBlockManager
    from lmdeploy.pytorch.paging.block_trie.checkpoint_lifecycle import StateCheckpointLifecycle
    from lmdeploy.pytorch.paging.state_manager import StateManager


class SequenceLifecycle:
    """Own sequence registration, transitions, and paging cleanup."""

    def __init__(
        self,
        seq_manager: SequenceManager,
        block_manager: 'BaseBlockManager',
        state_manager: 'StateManager',
        state_checkpoints: 'StateCheckpointLifecycle',
        prefix_cache_enabled: bool,
        is_ssm: bool,
        connector: 'KVConnectorBase | None',
    ) -> None:
        self._seq_manager = seq_manager
        self._block_manager = block_manager
        self._state_manager = state_manager
        self._state_checkpoints = state_checkpoints
        self._prefix_cache_enabled = prefix_cache_enabled
        self._is_ssm = is_ssm
        self._connector = connector

    def new_sequence_id(self) -> int:
        return self._seq_manager.new_sequence_id()

    def add_sequence(self, seq: SchedulerSequence, status: MessageStatus) -> None:
        seq.session.sequences[seq.seq_id] = seq
        seq.set_state(StateBase.build(self, seq, status))
        self._seq_manager.register_sequence(seq)
        if self._connector is not None:
            self._connector.on_new_request(seq)

    def remove_sequence(self, seq: SchedulerSequence) -> None:
        """Release local ownership without a terminal connector event."""
        assert seq.seq_id in seq.session.sequences
        self.release_paging_resources(seq)
        seq.session.sequences.pop(seq.seq_id)
        self._seq_manager.unregister_sequence(seq)

    def end_sequence(self, seq: SchedulerSequence) -> None:
        """Notify the connector that the request ended, then remove it."""
        if self._connector is not None:
            self._connector.request_finished(seq)
        self.remove_sequence(seq)

    def transition(self, seq: SchedulerSequence, new_state: type['StateBase']) -> None:
        self._seq_manager.update_sequence_status(seq, new_state.status)
        seq.set_state(new_state(seq, self))

    def assert_allocated(self, seq: SchedulerSequence) -> None:
        num_required_blocks = self._block_manager.num_required_blocks(seq)
        assert seq.num_blocks >= num_required_blocks
        if self._is_ssm:
            assert seq.logical_state >= 0

    def release_paging_resources(self, seq: SchedulerSequence) -> None:
        """Release blocks and state without changing sequence status."""
        if self._prefix_cache_enabled:
            self._state_checkpoints.discard_save(seq)
            self._state_checkpoints.unpin_restore(seq)
            seq.prefix_cache.restore.clear()
            seq.prefix_cache.trie_cursor = None
            seq.prefix_cache.match_start_step = -1
            seq.prefix_cache.recompute_overlap.clear_tracking()
        seq.cached_tokens = 0
        seq.kv_token_limit = None
        if seq.num_blocks > 0:
            self._block_manager.free(seq)
        if seq.logical_state >= 0:
            self._state_manager.free(seq)
        seq.set_step(0)

    def disable_connector(self) -> None:
        self._connector = None


class StateBase:
    status = None
    _registry = dict()

    def __init_subclass__(cls, **kargs) -> None:
        super().__init_subclass__(**kargs)
        if cls.status:
            cls._registry[cls.status] = cls

    @classmethod
    def build(cls, lifecycle: SequenceLifecycle, seq: 'SchedulerSequence', status: MessageStatus) -> 'StateBase':
        """Build sequence state."""
        if status not in cls._registry:
            raise NotImplementedError(f'Unsupported status {status} for building seq state.')
        return cls._registry[status](seq, lifecycle)

    def __init__(self, seq: SchedulerSequence, lifecycle: SequenceLifecycle):
        self.seq = seq
        self.lifecycle = lifecycle

    def to_state(self, new_state):
        """Transition to a new state."""
        self.lifecycle.transition(self.seq, new_state)

    def evict(self):
        """Evict the state."""
        raise NotImplementedError(f'evict not implemented for state {self.status}')

    def activate(self):
        """Activate the state."""
        raise NotImplementedError(f'activate not implemented for state {self.status}')

    def deactivate(self):
        """Deactivate the state."""
        raise NotImplementedError(f'deactivate not implemented for state {self.status}')

    def finish(self):
        """Finish the state."""
        raise NotImplementedError(f'finish not implemented for state {self.status}')

    def stop(self):
        """Stop the state."""
        self.to_state(StoppedState)

    def release_paging_resources(self):
        """Release blocks and state without changing sequence status."""
        self.lifecycle.release_paging_resources(self.seq)

    def begin_remote_load(self):
        raise NotImplementedError(f'begin_remote_load not implemented for state {self.status}')

    def finish_remote_load(self):
        raise NotImplementedError(f'finish_remote_load not implemented for state {self.status}')


class WaitingState(StateBase):
    """State for waiting sequences."""
    status = MessageStatus.WAITING

    def activate(self):
        """From WAITING to READY."""
        self.lifecycle.assert_allocated(self.seq)
        self.to_state(ReadyState)

    def evict(self):
        self.to_state(WaitingState)

    def begin_remote_load(self):
        """Protect allocated destinations until every TP rank completes."""
        self.to_state(RemoteLoadingState)


class RemoteLoadingState(StateBase):
    """Sequence with an asynchronous external write into its KV blocks."""

    status = MessageStatus.WAITING_FOR_REMOTE_KVS

    def finish_remote_load(self):
        self.to_state(WaitingState)


class ReadyState(StateBase):
    """State for ready sequences."""
    status = MessageStatus.READY

    def activate(self):
        """From READY to RUNNING."""
        self.to_state(RunningState)

    def evict(self):
        # clean up meta before evict
        self.seq.cleanup()
        self.to_state(WaitingState)


class StoppedState(StateBase):
    """State for stopped sequences."""
    status = MessageStatus.STOPPED

    def activate(self):
        """From STOPPED to WAITING."""
        assert self.seq.num_token_ids > 0
        self.to_state(WaitingState)

    def evict(self):
        self.to_state(StoppedState)


class RunningState(StateBase):
    """State for running sequences."""
    status = MessageStatus.RUNNING

    def deactivate(self):
        self.to_state(ReadyState)

    def finish(self):
        if self.seq.preserve_cache:
            self.to_state(ToBeMigratedState)
        else:
            self.to_state(StoppedState)


class ToBeMigratedState(StateBase):
    """State for to be migrated sequences."""
    status = MessageStatus.TO_BE_MIGRATED

    def finish(self):
        self.to_state(StoppedState)


class MigrationWaitingState(StateBase):
    """State for migration waiting sequences."""
    status = MessageStatus.MIGRATION_WAITING

    def activate(self):
        self.to_state(MigrationReadyState)

    def evict(self):
        self.to_state(MigrationWaitingState)


class MigrationReadyState(StateBase):
    """State for migration ready sequences."""
    status = MessageStatus.MIGRATION_READY

    def activate(self):
        self.to_state(MigrationRunningState)

    def evict(self):
        self.to_state(MigrationWaitingState)


class MigrationDoneState(StateBase):
    """State for migration done sequences."""
    status = MessageStatus.MIGRATION_DONE

    def activate(self):
        self.to_state(WaitingState)

    def finish(self):
        self.to_state(WaitingState)


class MigrationRunningState(StateBase):
    """State for migration running sequences."""
    status = MessageStatus.MIGRATION_RUNNING

    def deactivate(self):
        self.to_state(MigrationDoneState)

    def finish(self):
        self.to_state(MigrationDoneState)
