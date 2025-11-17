# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/vllm-project/vllm/blob/main/vllm/v1/metrics/stats.py

import time
from dataclasses import dataclass
from typing import List, Optional

from lmdeploy.messages import EngineEvent, EngineOutput, ResponseType, ScheduleMetrics


@dataclass
class SchedulerStats:
    """Stats associated with the scheduler.

    Attributes:
        num_total_reqs: the number of all requests received since server start.
        num_finished_reqs: the number of successfully completed requests since server start.
        num_running_reqs: currently executing requests.
        num_waiting_reqs: Requests queued waiting for execution.
        gpu_cache_usage: Fraction of GPU KV blocks utilized (0.0 to 1.0).
    """

    num_total_reqs: int = 0
    num_finished_reqs: int = 0
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    gpu_cache_usage: float = 0.0

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('SchedulerStats(\n'
                f'  num_total_reqs={self.num_total_reqs},\n'
                f'  num_finished_reqs={self.num_finished_reqs},\n'
                f'  num_running_reqs={self.num_running_reqs},\n'
                f'  num_waiting_reqs={self.num_waiting_reqs},\n'
                f'  gpu_cache_usage={self.gpu_cache_usage:.6f},\n'
                ')')

    def update_from_schedule_metrics(self, scheduled_metrics: ScheduleMetrics):
        self.num_running_reqs = scheduled_metrics.active_seqs
        self.num_waiting_reqs = scheduled_metrics.waiting_seqs
        self.gpu_cache_usage = 1.0 - (scheduled_metrics.free_blocks / scheduled_metrics.total_blocks)


class RequestState:
    """State of a request."""

    def __init__(self, arrival_time: float = None, prompt_tokens: int = 0):
        """Initialize the state of a request.

        Args:
            arrival_time (float, optional): The timestamp when the request arrives.
                If not provided, the current time will be used. Defaults to None.
            prompt_tokens (int, optional): The number of tokens in the prompt. Defaults to 0.
        """
        self.arrival_time = time.time() if arrival_time is None else arrival_time
        self.prompt_tokens = prompt_tokens

        # Number of tokens generated during the request inference.
        # It will be updated by IterationStats.update_from_output.
        self.generation_tokens: int = 0
        # Time when the request is put to the inference engine's queue. It will be updated according the EngineEvent
        self.queued_time: float = 0.0
        # Time when the request is scheduled to run. It will be updated according the EngineEvent
        self.scheduled_time: float = 0.0
        # Time when the first token is generated. It will be updated by IterationStats.update_from_output.
        self.first_token_time: float = 0.0
        # Time when the latest token is generated. It will be updated by IterationStats.update_from_output.
        self.lastest_token_time: float = 0.0
        # Time when a request finishes generation. It will be updated by IterationStats.update_from_output.
        self.finish_time: float = 0.0
        self.finish_reason: ResponseType = None

    def update_from_events(self, engine_events: List[EngineEvent]):
        # Avoid circular dependency
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
    def finish_stats(self) -> 'FinishedRequestStats':
        """Return stats of a finished request.

        It has to be called when a request is finished
        """

        e2e_latency = self.finish_time - self.arrival_time

        # Queued interval is from first QUEUED event to first SCHEDULED
        queued_time = self.scheduled_time - self.queued_time

        # Prefill interval is from first SCHEDULED to first NEW_TOKEN
        # Any preemptions during prefill is included in the interval
        prefill_time = self.first_token_time - self.scheduled_time

        # Decode interval is from first NEW_TOKEN to last NEW_TOKEN
        # Any preemptions during decode are included
        decode_time = self.finish_time - self.first_token_time

        # Inference interval is from first SCHEDULED to last NEW_TOKEN
        # Any preemptions during prefill or decode are included
        inference_time = self.finish_time - self.scheduled_time

        finished_req = \
            FinishedRequestStats(finish_reason=self.finish_reason,
                                 e2e_latency=e2e_latency,
                                 prompt_tokens=self.prompt_tokens,
                                 generation_tokens=self.generation_tokens,
                                 queued_time=queued_time,
                                 prefill_time=prefill_time,
                                 inference_time=inference_time,
                                 decode_time=decode_time)
        return finished_req

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('RequestState(\n'
                f'  arrival_time={self.arrival_time:.6f},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  generation_tokens={self.generation_tokens},\n'
                f'  queued_time={self.queued_time:.6f},\n'
                f'  scheduled_time={self.scheduled_time:.6f},\n'
                f'  first_token_time={self.first_token_time:.6f},\n'
                f'  latest_token_time={self.lastest_token_time:.6f},\n'
                ')')


@dataclass
class FinishedRequestStats:
    """Stats associated with a finished request."""
    finish_reason: ResponseType
    e2e_latency: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0
    queued_time: float = 0.0
    prefill_time: float = 0.0
    inference_time: float = 0.0
    decode_time: float = 0.0

    def __repr__(self):
        """Return a human-readable string representation."""
        return ('FinishedRequestStats(\n'
                f'  e2e_latency={self.e2e_latency:.6f},\n'
                f'  prompt_tokens={self.prompt_tokens},\n'
                f'  generation_tokens={self.generation_tokens},\n'
                f'  queued_time={self.queued_time:.6f},\n'
                f'  prefill_time={self.prefill_time:.6f},\n'
                f'  inference_time={self.inference_time:.6f},\n'
                f'  decode_time={self.decode_time:.6f}\n'
                ')')


class IterationStats:
    """Stats associated with one token generation iteration of a request."""

    def __init__(self):
        # Record the timestamp when this iteration finished
        self.iteration_timestamp = time.time()
        # The number of newly generated tokens in this iteration
        self.new_generation_tokens = 0
        # The number of prompt tokens processed in this iteration
        self.prompt_tokens = 0
        # Time to First Token (TTFT), initialized as None and will be updated later
        self.ttft: Optional[float] = None
        # Time per Output Token (TPOT), initialized as None and will be updated later
        self.tpot: Optional[float] = None
        # Iter-Token Latency, initialized as None and will be updated later
        self.itl: Optional[float] = None

    def __repr__(self):
        """Return a human-readable string representation."""
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

    def update_from_output(self, outputs: EngineOutput, req_state: RequestState):
        """Update the iteration statistics based on the engine output and
        request state.

        Args:
            outputs (EngineOutput): The output from the engine containing information about the current iteration.
            req_state (RequestState): The state of the request, including timestamps and token counts.
        """
        if outputs.req_metrics is None:
            # when users visit "/abort_request" endpoint, `req_metrics` might be None
            return
        new_generation_tokens = len(outputs.token_ids)
        if new_generation_tokens == 0:
            return
        self.new_generation_tokens = new_generation_tokens
        if req_state.first_token_time == 0:
            # It means the first token is generated in this iteration
            req_state.first_token_time = outputs.req_metrics.token_timestamp
            self.prompt_tokens = req_state.prompt_tokens
            self.ttft = self._time_since(req_state.arrival_time)
        else:
            self.itl = self._time_since(req_state.lastest_token_time)
            self.tpot = self._time_since(req_state.lastest_token_time) / self.new_generation_tokens
        # update the latest token generation time
        req_state.lastest_token_time = outputs.req_metrics.token_timestamp
        # update the number of generated tokens
        req_state.generation_tokens += new_generation_tokens

        if outputs.status != ResponseType.SUCCESS:
            req_state.finish_reason = outputs.status
            req_state.finish_time = self.iteration_timestamp
