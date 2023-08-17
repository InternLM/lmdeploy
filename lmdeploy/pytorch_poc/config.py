from dataclasses import dataclass


@dataclass
class SchedulerConfig:
    max_batches: int
    max_session_len: int
