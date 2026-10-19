# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from lmdeploy.messages import EngineOutput, RequestMetrics, ResponseType, ScheduleMetrics
from lmdeploy.metrics.loggers import LoggingStatLogger, PrometheusStatLogger
from lmdeploy.metrics.stats import IterationStats, RequestStats, SchedulerStats, SpeculativeDecodingStats

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


@pytest.mark.parametrize('external_hit_rate', [None, 0.0, 12 / 17])
def test_logging_stat_logger_separates_local_and_enabled_external_cache(capsys, external_hit_rate):
    logger = LoggingStatLogger(dp_rank=2)
    stats = SchedulerStats()
    stats.update_from_schedule_metrics(ScheduleMetrics(
        prefix_cache_hit_rate=4 / 21, external_prefix_cache_hit_rate=external_hit_rate))
    logger.record_schedule(stats)
    logger.total_generation_tokens = 1
    logger.log()

    output = capsys.readouterr().out
    assert 'DP2' in output
    assert 'Prefix cache hit rate: 19.0%' in output
    if external_hit_rate is None:
        assert 'External prefix cache hit rate' not in output
    else:
        assert f'External prefix cache hit rate: {external_hit_rate * 100:.1f}%' in output


def test_prometheus_external_cache_counters_use_snapshot_deltas():
    logger = PrometheusStatLogger('test-model', max_model_len=64, dp_rank=2)
    stats = SchedulerStats()
    labels = {'model_name': 'test-model', 'engine': '2'}

    # Reading the same cumulative scheduler snapshot twice must not add tokens.
    for queries, hits in [(17, 12), (17, 12), (38, 12)]:
        stats.update_from_schedule_metrics(ScheduleMetrics(
            prefix_cache_hit_rate=4 / 42, external_prefix_cache_queries=queries,
            external_prefix_cache_hits=hits, external_prefix_cache_hit_rate=hits / queries))
        logger.record_schedule(stats)
        assert _get_sample_value('lmdeploy:external_prefix_cache_queries_total', labels) == queries
        assert _get_sample_value('lmdeploy:external_prefix_cache_hits_total', labels) == hits
        assert _get_sample_value('lmdeploy:external_prefix_cache_hit_rate', labels) == pytest.approx(hits / queries)
    assert _get_sample_value('lmdeploy:prefix_cache_hit_rate', labels) == pytest.approx(4 / 42)
    exposition = prometheus_client.generate_latest().decode()
    assert '# TYPE lmdeploy:external_prefix_cache_queries_total counter' in exposition
    assert '# TYPE lmdeploy:external_prefix_cache_hits_total counter' in exposition

    # A fresh engine starts its own cumulative totals without decreasing the
    # API process's Prometheus counters.
    stats.update_from_schedule_metrics(ScheduleMetrics(
        external_prefix_cache_queries=21, external_prefix_cache_hits=16,
        external_prefix_cache_hit_rate=16 / 21))
    logger.record_schedule(stats)
    assert _get_sample_value('lmdeploy:external_prefix_cache_queries_total', labels) == 59
    assert _get_sample_value('lmdeploy:external_prefix_cache_hits_total', labels) == 28


def test_zero_token_finish_records_request_reason():
    req_stats = RequestStats(prompt_tokens=4)
    iteration_stats = IterationStats()
    req_stats.scheduled_time = iteration_stats.iteration_timestamp - 2
    output = EngineOutput(ResponseType.FINISH, [], req_metrics=RequestMetrics())

    iteration_stats.update_from_output(output, req_stats)

    assert iteration_stats.new_generation_tokens == 0
    assert req_stats.generation_tokens == 0
    assert req_stats.finish_reason == ResponseType.FINISH
    assert req_stats.finish_time == iteration_stats.iteration_timestamp
    assert req_stats.first_token_time == req_stats.finish_time
    assert req_stats.prefill_time_interval == pytest.approx(2)
    assert req_stats.decode_time_interval == 0
    assert iteration_stats.prompt_tokens == 4
    assert iteration_stats.ttft is None
