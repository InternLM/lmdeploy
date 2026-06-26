# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from lmdeploy.messages import EngineEvent, EngineOutput, EventType, RequestMetrics, ResponseType
from lmdeploy.metrics.loggers import LoggingStatLogger, PrometheusStatLogger
from lmdeploy.metrics.stats import IterationStats, RequestStats, SpeculativeDecodingStats

prometheus_client = pytest.importorskip('prometheus_client')


def _get_sample_value(name: str, labels: dict[str, str]) -> float:
    for metric in prometheus_client.REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != name:
                continue
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                return sample.value
    raise AssertionError(f'Missing prometheus sample: {name}{labels}')


def test_prometheus_stat_logger_records_specdecode_metrics():
    logger = PrometheusStatLogger('test-model', max_model_len=16, dp_rank=2)
    stats = SpeculativeDecodingStats(num_spec_tokens=3)
    stats.update_per_draft(num_draft_tokens=3, num_accepted_tokens=2)
    stats.update_per_draft(num_draft_tokens=3, num_accepted_tokens=1)

    logger.record_specdecode(stats)

    labels = {'model_name': 'test-model', 'engine': '2'}
    assert _get_sample_value('lmdeploy:spec_decode_num_drafts_total', labels) == 2
    assert _get_sample_value('lmdeploy:spec_decode_num_draft_tokens_total', labels) == 6
    assert _get_sample_value('lmdeploy:spec_decode_num_accepted_tokens_total', labels) == 3
    assert _get_sample_value('lmdeploy:spec_decode_mean_accept_rate', labels) == 0.5
    assert _get_sample_value('lmdeploy:spec_decode_mean_accept_length', labels) == 2.5

    position_labels = labels | {'position': '0'}
    assert _get_sample_value('lmdeploy:spec_decode_num_accepted_tokens_per_pos_total', position_labels) == 2
    assert _get_sample_value('lmdeploy:spec_decode_per_position_accept_rate', position_labels) == 1
    position_labels = labels | {'position': '1'}
    assert _get_sample_value('lmdeploy:spec_decode_num_accepted_tokens_per_pos_total', position_labels) == 1
    assert _get_sample_value('lmdeploy:spec_decode_per_position_accept_rate', position_labels) == 0.5
    position_labels = labels | {'position': '2'}
    assert _get_sample_value('lmdeploy:spec_decode_num_accepted_tokens_per_pos_total', position_labels) == 0
    assert _get_sample_value('lmdeploy:spec_decode_per_position_accept_rate', position_labels) == 0


def test_iteration_stats_counts_preempted_events():
    stats = IterationStats()
    output = EngineOutput(status=ResponseType.SUCCESS,
                          token_ids=[],
                          req_metrics=RequestMetrics(engine_events=[
                              EngineEvent(EventType.PREEMPTED, 1.0),
                          ]))

    stats.update_from_output(output, req_stats=None)

    assert stats.num_preempted_reqs == 1


def test_request_stats_lifetime_intervals_include_preemptions():
    stats = RequestStats(arrival_time=100.0, prompt_tokens=4)
    stats.update_from_events([
        EngineEvent(EventType.QUEUED, 101.0),
        EngineEvent(EventType.SCHEDULED, 102.0),
        EngineEvent(EventType.PREEMPTED, 110.0),
        EngineEvent(EventType.SCHEDULED, 120.0),
    ])
    stats.first_token_time = 130.0
    stats.lastest_token_time = 150.0
    stats.finish_time = 155.0
    stats.generation_tokens = 3

    assert stats.queued_time_interval == 1.0
    assert stats.prefill_time_interval == 28.0
    assert stats.decode_time_interval == 20.0
    assert stats.inference_time_interval == 48.0
    assert stats.e2e_latency == 55.0


def test_iteration_stats_ttft_includes_preemption_before_first_token():
    req_stats = RequestStats(arrival_time=100.0, prompt_tokens=4)
    events = [
        EngineEvent(EventType.QUEUED, 101.0),
        EngineEvent(EventType.SCHEDULED, 102.0),
        EngineEvent(EventType.PREEMPTED, 110.0),
        EngineEvent(EventType.SCHEDULED, 120.0),
    ]
    req_stats.update_from_events(events)
    output = EngineOutput(status=ResponseType.SUCCESS,
                          token_ids=[1],
                          req_metrics=RequestMetrics(token_timestamp=145.0, engine_events=events))
    stats = IterationStats()
    stats.iteration_timestamp = 150.0

    stats.update_from_output(output, req_stats)

    assert req_stats.scheduled_time == 102.0
    assert stats.ttft == 50.0
    assert stats.num_preempted_reqs == 1


def test_logging_stat_logger_logs_preemptions(capsys):
    logger = LoggingStatLogger(dp_rank=0)
    stats = IterationStats()
    stats.new_generation_tokens = 1
    stats.num_preempted_reqs = 1

    logger.record_iteration(stats)
    logger.log()

    captured = capsys.readouterr()
    assert 'Preemptions: 1' in captured.out


def test_prometheus_stat_logger_records_preemptions():
    logger = PrometheusStatLogger('test-model', max_model_len=16, dp_rank=3)
    stats = IterationStats()
    stats.num_preempted_reqs = 2

    logger.record_iteration(stats)

    labels = {'model_name': 'test-model', 'engine': '3'}
    assert _get_sample_value('lmdeploy:num_preemptions_total', labels) == 2
