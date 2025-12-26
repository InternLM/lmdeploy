# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from lmdeploy.messages import EngineEvent, EngineOutput, ResponseType, ScheduleMetrics


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler.

    Attributes:
        num_total_reqs: The number of all requests received since server start.
        num_finished_reqs: The number of successfully completed requests since server start.
        num_running_reqs: Currently executing requests.
        num_waiting_reqs: Requests queued waiting for execution.
        gpu_cache_usage: Fraction of GPU KV blocks utilized (0.0 to 1.0).
        prefix_cache_hit_rate: Prefix caching hit rate.
    """

    num_total_reqs: int = 0
    num_finished_reqs: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    gpu_cache_usage: float = 0.0
    prefix_cache_hit_rate: float = 0.0

    def __repr__(self):
        return ('SchedulerStats(\n'
                f'  num_total_reqs={self.num_total_reqs},\n'
                f'  num_finished_reqs={self.num_finished_reqs},\n'
                f'  num_running_reqs={self.num_running_reqs},\n'
                f'  num_waiting_reqs={self.num_waiting_reqs},\n'
                f'  gpu_cache_usage={self.gpu_cache_usage:.6f},\n'
                f'  prefix_cache_hit_rate={self.prefix_cache_hit_rate:.6f},\n'
                ')')

    def update_from_schedule_metrics(self, scheduled_metrics: ScheduleMetrics):
        self.num_running_reqs = scheduled_metrics.active_seqs
        self.num_waiting_reqs = scheduled_metrics.waiting_seqs
        self.gpu_cache_usage = 1.0 - (scheduled_metrics.free_blocks / scheduled_metrics.total_blocks)
        self.prefix_cache_hit_rate = scheduled_metrics.prefix_cache_hit_rate


class RequestStats:
    """Stats associated with a request."""

    def __init__(self, arrival_time: float = None, prompt_tokens: int = 0):
        """Initialize the stats of a request.

        Args:
            arrival_time (float, optional): The timestamp when the request arrives.
                If not provided, the current time will be used. Defaults to None.
            prompt_tokens (int, optional): The number of tokens in the prompt. Defaults to 0.

        Attributes:
            generation_tokens (int): The number of tokens generated during the request inference.
                It will be updated by IterationStats.update_from_output.
            queued_time (float): Time when the request is put to the inference engine's queue.
                It will be updated according the EngineEvent.
            scheduled_time (float): Time when the request is scheduled to run.
                It will be updated according the EngineEvent.
            first_token_time (float): Time when the first token is generated.
                It will be updated by IterationStats.update_from_output.
            lastest_token_time (float): Time when the latest token is generated.
                It will be updated by IterationStats.update_from_output.
            finish_time (float): Time when a request finishes generation.
                It will be updated by IterationStats.update_from_output.
            finish_reason (ResponseType): The reason why the request finished.
        """
        self.arrival_time = time.time() if arrival_time is None else arrival_time
        self.prompt_tokens = prompt_tokens

        self.generation_tokens: int = 0
        self.queued_time: float = 0.0
        self.scheduled_time: float = 0.0
        self.first_token_time: float = 0.0
        self.lastest_token_time: float = 0.0
        self.finish_time: float = 0.0
        self.finish_reason: ResponseType = None

    def __repr__(self):
        return ('RequestStats(\n'
                f'  arrival_time={self.arrival_time:.6f},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  generation_tokens={self.generation_tokens},\n'
                f'  queued_time={self.queued_time:.6f},\n'
                f'  scheduled_time={self.scheduled_time:.6f},\n'
                f'  first_token_time={self.first_token_time:.6f},\n'
                f'  latest_token_time={self.lastest_token_time:.6f},\n'
                ')')

    def update_from_events(self, engine_events: List[EngineEvent]):
        # avoid circular dependency
        from lmdeploy.messages import EventType

        for event in engine_events:
            if event.type == EventType.QUEUED:
                self.queued_time = event.timestamp
            elif event.type == EventType.SCHEDULED:
                if self.scheduled_time == 0.0:  # ignore preemptions
                    self.scheduled_time = event.timestamp
            # FIXME: deal with preempted case
            # elif event.type == EventType.PREEMPTED:
            #     self.num_preempted_reqs += 1

    @property
    def e2e_latency(self) -> float:
        """End-to-end latency."""
        return self.finish_time - self.arrival_time

    @property
    def queued_time_interval(self) -> float:
        """Queued interval is from first QUEUED event to first SCHEDULED."""
        return self.scheduled_time - self.queued_time

    @property
    def prefill_time_interval(self) -> float:
        """Prefill interval is from first SCHEDULED to first NEW_TOKEN.

        Any preemptions during prefill is included in the interval.
        """
        return self.first_token_time - self.scheduled_time

    @property
    def decode_time_interval(self) -> float:
        """Decode interval is from first NEW_TOKEN to last NEW_TOKEN.

        Any preemptions during decode are included.
        """
        return self.finish_time - self.first_token_time

    @property
    def inference_time_interval(self) -> float:
        """Inference interval is from first SCHEDULED to last NEW_TOKEN.

        Any preemptions during prefill or decode are included.
        """
        return self.finish_time - self.scheduled_time


class IterationStats:
    """Stats associated with one token generation iteration of a request."""

    def __init__(self):
        """Initialize the stats of one iteration.

        Attributes:
            iteration_timestamp (float): The timestamp when this iteration finishes.
            new_generation_tokens (int): The number of newly generated tokens in this iteration.
            prompt_tokens (int): The number of prompt tokens processed in this iteration.
            ttft (float | None): Time to First Token (TTFT).
            tpot (float | None): Time per Output Token (TPOT).
            itl (float | None): Iter-Token Latency (ITL).
        """
        self.iteration_timestamp = time.time()
        self.new_generation_tokens = 0
        self.prompt_tokens = 0
        self.ttft: Optional[float] = None
        self.tpot: Optional[float] = None
        self.itl: Optional[float] = None

    def __repr__(self):
        return ('IterationStats(\n'
                f'  iteration_timestamp={self.iteration_timestamp:.6f},\n'
                f'  new_generation_tokens={self.new_generation_tokens},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  ttft={self.ttft},\n'
                f'  tpot={self.tpot},\n'
                f'  itl={self.itl},\n'
                ')')

    def _time_since(self, start: float) -> float:
        """Calculate an interval relative to this iteration's timestamp."""
        return self.iteration_timestamp - start

    def update_from_output(self, outputs: EngineOutput, req_stats: RequestStats):
        """Update the iteration statistics.

        Args:
            outputs (EngineOutput): The output from the engine containing information about the current iteration.
            req_stats (RequestStats): The stats of the request, including timestamps and token counts.
        """
        if outputs.req_metrics is None:
            # when users visit "/abort_request" endpoint, `req_metrics` might be None
            return

        new_generation_tokens = len(outputs.token_ids)
        if new_generation_tokens == 0:
            return

        self.new_generation_tokens = new_generation_tokens

        if req_stats.first_token_time == 0:
            # the first token is generated in this iteration
            req_stats.first_token_time = outputs.req_metrics.token_timestamp
            self.prompt_tokens = req_stats.prompt_tokens
            self.ttft = self._time_since(req_stats.arrival_time)
        else:
            self.itl = self._time_since(req_stats.lastest_token_time)
            self.tpot = self._time_since(req_stats.lastest_token_time) / self.new_generation_tokens

        req_stats.lastest_token_time = outputs.req_metrics.token_timestamp
        req_stats.generation_tokens += new_generation_tokens

        if outputs.status != ResponseType.SUCCESS:
            req_stats.finish_reason = outputs.status
            req_stats.finish_time = self.iteration_timestamp


# modify from vllm
@dataclass
class SpeculativeDecodingStats:
    """Speculative decoding stats."""

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: np.ndarray = None

    def __post_init__(self):
        assert self.num_spec_tokens > 0
        self.num_accepted_tokens_per_pos = np.zeros(self.num_spec_tokens)

    def update_from_output(self, outputs: EngineOutput):
        """Update from engine output."""
        spec_info = getattr(outputs.req_metrics, 'spec_info', None)
        if spec_info:
            self.num_drafts += 1
            self.num_draft_tokens += spec_info['num_draft_tokens']
            self.num_accepted_tokens += spec_info['num_accepted_tokens']
            self.num_accepted_tokens_per_pos[:spec_info['num_accepted_tokens']] += 1

    def update_per_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        """Update with per draft stats."""
        if num_draft_tokens > 0:
            self.num_drafts += 1
            self.num_draft_tokens += num_draft_tokens
            self.num_accepted_tokens += num_accepted_tokens
            self.num_accepted_tokens_per_pos[:num_accepted_tokens] += 1

    def __repr__(self):
        draft_acceptance_rate = (self.num_accepted_tokens / self.num_draft_tokens *
                                 100 if self.num_draft_tokens > 0 else float('nan'))

        # conventionally, mean acceptance length includes the bonus token
        mean_acceptance_length = 1 + (self.num_accepted_tokens /
                                      self.num_drafts) if self.num_drafts > 0 else float('nan')

        acceptance_rates = self.num_accepted_tokens_per_pos / self.num_drafts if self.num_drafts > 0 else [
            float('nan')
        ] * self.num_accepted_tokens
        rates_str = ', '.join(f'{p:.3f}' for p in acceptance_rates)

        return ('SpeculativeDecodingStats('
                f'num_spec_tokens={self.num_spec_tokens}, '
                f'num_drafts={self.num_drafts}, '
                f'num_draft_tokens={self.num_draft_tokens}, '
                f'num_accepted_tokens={self.num_accepted_tokens}, '
                f'draft_acceptance_rate={draft_acceptance_rate:.2f}%, '
                f'mean_acceptance_length={mean_acceptance_length:.2f}, '
                f'per_position_acceptance_rate={rates_str})')
