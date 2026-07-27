# Copyright (c) OpenMMLab. All rights reserved.

import pytest

from lmdeploy.metrics.loggers import PrometheusStatLogger
from lmdeploy.metrics.stats import SpeculativeDecodingStats

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
