# Copyright (c) OpenMMLab. All rights reserved.
from types import SimpleNamespace

from lmdeploy.messages import EngineOutput, EventType, ResponseType, ScheduleMetrics
from lmdeploy.metrics.stats import SchedulerStats


def test_scheduler_stats_uses_explicit_cache_usage():
    stats = SchedulerStats()
    stats.update_from_schedule_metrics(
        ScheduleMetrics(active_seqs=2,
                        waiting_seqs=3,
                        total_blocks=0,
                        free_blocks=0,
                        cache_usage=0.375,
                        prefix_cache_hit_rate=0.25))

    assert stats.num_running_reqs == 2
    assert stats.num_waiting_reqs == 3
    assert stats.gpu_cache_usage == 0.375
    assert stats.prefix_cache_hit_rate == 0.25


def test_scheduler_stats_retains_block_usage_fallback():
    stats = SchedulerStats()
    stats.update_from_schedule_metrics(ScheduleMetrics(total_blocks=10, free_blocks=4))

    assert stats.gpu_cache_usage == 0.6


def test_turbomind_request_metrics_preserve_cached_tokens():
    from lmdeploy.turbomind.turbomind import _get_metrics

    convert = _get_metrics(SimpleNamespace(enqueue_time=1_000_000, scheduled_time=2_000_000, cached_tokens=24))

    first = EngineOutput(ResponseType.SUCCESS, [])
    convert(first, 0)
    assert first.req_metrics.cached_tokens == 24
    assert [event.type for event in first.req_metrics.engine_events] == [EventType.QUEUED, EventType.SCHEDULED]

    second = EngineOutput(ResponseType.SUCCESS, [])
    convert(second, 0)
    assert second.req_metrics.cached_tokens == 24
    assert second.req_metrics.engine_events == []


def test_turbomind_request_metrics_omit_unscheduled_event():
    from lmdeploy.turbomind.turbomind import _get_metrics

    convert = _get_metrics(SimpleNamespace(enqueue_time=1_000_000, scheduled_time=0, cached_tokens=0))
    output = EngineOutput(ResponseType.INTERNAL_ENGINE_ERROR, [])
    convert(output, 0)

    assert [event.type for event in output.req_metrics.engine_events] == [EventType.QUEUED]
