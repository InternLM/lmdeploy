import json
import threading
from typing import List


class Buffer:
    def __init__(self, ts: int, user_groups: List[str]):
        self.ts = ts
        # Per user usage
        self.uid_to_tokens_ps = dict()
        self.uid_to_reqs_ps = dict()

        # Per group usage
        self.group_to_tokens_ps = dict()
        self.group_to_reqs_ps = dict()

        for group in user_groups:
            self.group_to_tokens_ps[group] = 0
            self.group_to_reqs_ps[group] = 0


class UsageStats:
    def __init__(self, total_duration: int, buffer_count: int, start_index: int, user_groups: List[str]):
        self.total_duration = total_duration
        self.buffer_count = buffer_count
        self.start_index = start_index
        self.start_ts = int(0)

        self.buffer_duration = int(total_duration / buffer_count)
        self.circular_buffer = [Buffer(self.buffer_duration * i, user_groups) for i in range(buffer_count)]

        self.user_groups = user_groups
        
        self.lock = threading.Lock()

    def update_usage(self, uid: str, group: str, out_token_len: int, req_ts: int):
        """
        Update UsageStats when a request is returned
        """
        with self.lock:
            intervals = int((req_ts-self.start_ts) / self.buffer_duration)

            curr_idx = (self.start_index+intervals) % self.buffer_count
            curr_ts = self.start_ts + intervals*self.buffer_duration

            # Current request outside the sliding window
            if intervals >= self.buffer_count:
                reset_buf_cnt = intervals - self.buffer_count
                curr_buf_ts = 0

                if reset_buf_cnt >= self.buffer_count:
                    # All buffers are reset
                    for i in range(1, self.buffer_count):
                        reset_idx = (curr_idx+i) % self.buffer_count
                        self.circular_buffer[reset_idx] = Buffer(req_ts + i*self.buffer_duration, self.user_groups)
                    # Update self.start_index
                    self.start_index = curr_idx
                    self.start_ts = req_ts
                    curr_buf_ts = req_ts
                else:
                    # Only buffers between self.start_index and curr_idx are reset
                    for i in range(reset_buf_cnt):
                        reset_idx = (self.start_index+i) % self.buffer_count
                        reset_ts = self.circular_buffer[reset_idx].ts + self.total_duration
                        self.circular_buffer[reset_idx] = Buffer(reset_ts, self.user_groups)

                    # Update self.start_index
                    self.start_index = (curr_idx+1) % self.buffer_count
                    self.start_ts = self.circular_buffer[self.start_index].ts
                    curr_buf_ts = self.circular_buffer[curr_idx].ts + self.total_duration
                
                # Set corresponding buffer
                self.circular_buffer[curr_idx] = Buffer(curr_buf_ts, self.user_groups)
                self.circular_buffer[curr_idx].uid_to_reqs_ps[uid] = 1
                self.circular_buffer[curr_idx].uid_to_tokens_ps[uid] = out_token_len
                self.circular_buffer[curr_idx].group_to_reqs_ps[group] = 1
                self.circular_buffer[curr_idx].group_to_tokens_ps[group] = out_token_len
                
            # Otherwise update corresponding buffer
            else:
                self.circular_buffer[curr_idx].ts = curr_ts

                if uid in self.circular_buffer[curr_idx].uid_to_reqs_ps:
                    self.circular_buffer[curr_idx].uid_to_reqs_ps[uid] += 1
                else:
                    self.circular_buffer[curr_idx].uid_to_reqs_ps[uid] = 1
                
                if uid in self.circular_buffer[curr_idx].uid_to_tokens_ps:
                    self.circular_buffer[curr_idx].uid_to_tokens_ps[uid] += out_token_len
                else:
                    self.circular_buffer[curr_idx].uid_to_tokens_ps[uid] = out_token_len

                self.circular_buffer[curr_idx].group_to_reqs_ps[group] += 1
                self.circular_buffer[curr_idx].group_to_tokens_ps[group] += out_token_len

    def get_user_usage(self, uid: str, group: str):
        """
        Calculate usage stats of the given user and group
        """
        user_req_usage = 0
        user_token_usage = 0
        group_req_usage = 0
        group_token_usage = 0

        # TODO: use reader lock
        with self.lock:
            for i in range(self.buffer_count):
                if uid in self.circular_buffer[i].uid_to_reqs_ps:
                    user_req_usage += self.circular_buffer[i].uid_to_reqs_ps[uid]
                    user_token_usage += self.circular_buffer[i].uid_to_tokens_ps[uid]
                
                group_req_usage += self.circular_buffer[i].group_to_reqs_ps[group]
                group_token_usage += self.circular_buffer[i].group_to_tokens_ps[group]
            
        return user_req_usage, user_token_usage, group_req_usage, group_token_usage
