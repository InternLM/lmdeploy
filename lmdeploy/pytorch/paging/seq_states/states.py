# Copyright (c) OpenMMLab. All rights reserved.
from typing import TYPE_CHECKING

from lmdeploy.pytorch.messages import MessageStatus, SchedulerSequence

if TYPE_CHECKING:
    from lmdeploy.pytorch.paging import Scheduler


def _free_seq(seq: SchedulerSequence, scheduler: 'Scheduler'):
    """Free the sequence."""
    if seq.num_blocks > 0:
        scheduler.block_manager.free(seq)
    if seq.logical_state >= 0:
        scheduler.state_manager.free(seq)
    seq.set_step(0)


class StateBase:
    status = None
    _registry = dict()

    def __init_subclass__(cls, **kargs) -> None:
        super().__init_subclass__(**kargs)
        if cls.status:
            cls._registry[cls.status] = cls

    @classmethod
    def build(cls, scheduler: 'Scheduler', seq: 'SchedulerSequence', status: MessageStatus) -> 'StateBase':
        """Build sequence state."""
        if status not in cls._registry:
            raise NotImplementedError(f'Unsupported status {status} for building seq state.')
        return cls._registry[status](seq, scheduler)

    def __init__(self, seq: SchedulerSequence, scheduler: 'Scheduler'):
        self.seq = seq
        self.scheduler = scheduler

    def to_state(self, new_state):
        """Transition to a new state."""
        self.scheduler.seq_manager.update_sequence_status(self.seq, new_state.status)
        self.seq.set_state(new_state(self.seq, self.scheduler))

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

    def free(self):
        """Free the state."""
        _free_seq(self.seq, self.scheduler)


class WaitingState(StateBase):
    """State for waiting sequences."""
    status = MessageStatus.WAITING

    def activate(self):
        """From WAITING to READY."""
        num_req_blocks = self.scheduler.block_manager.num_required_blocks(self.seq)
        assert self.seq.num_blocks >= num_req_blocks
        if self.scheduler.is_ssm:
            assert self.seq.logical_state >= 0
        self.to_state(ReadyState)

    def evict(self):
        self.to_state(WaitingState)


class ReadyState(StateBase):
    """State for ready sequences."""
    status = MessageStatus.READY

    def activate(self):
        """From READY to RUNNING."""
        self.to_state(RunningState)

    def evict(self):
        self.to_state(WaitingState)


class StoppedState(StateBase):
    """State for stopped sequences."""
    status = MessageStatus.STOPPED

    def activate(self):
        """From STOPPED to WAITING."""
        assert self.seq.num_token_ids > 0
        self.to_state(WaitingState)


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
        self.to_state(ReadyState)

    def finish(self):
        self.to_state(ReadyState)


class MigrationRunningState(StateBase):
    """State for migration running sequences."""
    status = MessageStatus.MIGRATION_RUNNING

    def deactivate(self):
        self.to_state(MigrationDoneState)

    def finish(self):
        self.to_state(MigrationDoneState)


def build_seq_state(scheduler: 'Scheduler', seq: 'SchedulerSequence', status: MessageStatus) -> StateBase:
    """Build sequence state."""
    return StateBase.build(scheduler, seq, status)
