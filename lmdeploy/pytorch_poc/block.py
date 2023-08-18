# modify from: https://github.com/vllm-project/vllm
from dataclasses import dataclass


class LogicalTokenBlock:

    def __init__(self, block_id: int, block_size: int):
        self.block_id = block_id
        self.block_size = block_size

        self.num_tokens = 0

    def get_num_empty_slots(self):
        return self.block_size - self.num_tokens

    def is_empty(self):
        return self.num_tokens == 0

    def is_full(self):
        return self.num_tokens == self.block_size

    def append_tokens(self, num_tokens: int = 1):
        assert num_tokens <= self.get_num_empty_slots()
        self.num_tokens += num_tokens


@dataclass
class PhysicalTokenBlock:
    device: str
    block_id: int
    block_size: int
    ref_count: int = 0
