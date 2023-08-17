# modify from: https://github.com/vllm-project/vllm
from typing import List, Dict
from collections import OrderedDict
from lmdeploy.pytorch_poc.config import SchedulerConfig
from lmdeploy.pytorch_poc.messages import (SchedulerSession, SchedulerMessage,
                                           MessageStatus)


def _find_message_with_session_id(message_list: List[SchedulerMessage],
                                  session_id: int):
    return [
        message for message in message_list
        if message.session.session_id == session_id
    ]


class Scheduler:

    def __init__(self, scheduler_config: SchedulerConfig) -> None:
        self.scheduler_config = scheduler_config
        self.waiting: List[SchedulerMessage] = []
        self.running: List[SchedulerMessage] = []
        self.sessions: Dict[int, SchedulerSession] = OrderedDict()

    def add_session(self, session: SchedulerSession):
        assert session.session_id not in self.sessions
        self.sessions[session.session_id] = self.sessions

    def add_message(self, message: SchedulerMessage):
        assert message.session_id in self.sessions, (
            f'Unknown session id {message.session_id}')

        # push message to waiting queue
        message.status = MessageStatus.WAITING
        self.waiting.append(message)

    def schedule(self):
        running = self.running
        running_session_ids = [msg.session_id for msg in running]
        max_batches = self.scheduler_config.max_batches

        remain = []  # msg poped from waiting but not runnable

        while len(running) < max_batches and self.waiting:
            msg = self.waiting.pop(0)

            # do not add to running if the session already exists.
            if msg.session_id in running_session_ids:
                remain.append(msg)
                continue

            # add to running
            msg.status = MessageStatus.RUNNING
            running.append(msg)

        self.waiting.extend(remain)
        self.running = running

        return [
            msg for msg in self.running if msg.status == MessageStatus.RUNNING
        ]

    def _set_status_with_session(self, session_id, status):
        running_msg = _find_message_with_session_id(self.running, session_id)
        waiting_msg = _find_message_with_session_id(self.waiting, session_id)

        for msg in running_msg:
            msg.status = status
        for msg in waiting_msg:
            msg.status = status

    def stop_session(self, session_id):
        self._set_status_with_session(session_id, MessageStatus.STOPPED)

    def end_session(self, session_id):
        self._set_status_with_session(session_id, MessageStatus.ENDED)

    def has_unfinished(self):
        return self.waiting or self.running

    def _remove_session(self, session_id: int):
        assert session_id in self.sessions
        self.sessions.pop(session_id)

        # TODO: remove resources used by session

    def update(self):

        session_id_to_remove = set()

        def _update_queue(que: List[SchedulerMessage],
                          expect_status: MessageStatus):
            for msg in que:
                if msg.status == expect_status:
                    continue

                if msg.status == MessageStatus.ENDED:
                    session_id_to_remove.add(msg.session_id)

            return [msg for msg in que if msg.status == expect_status]

        self.waiting = _update_queue(self.waiting, MessageStatus.WAITING)
        self.running = _update_queue(self.running, MessageStatus.RUNNING)

        # remove session
        for session_id in session_id_to_remove:
            self._remove_session(session_id)
