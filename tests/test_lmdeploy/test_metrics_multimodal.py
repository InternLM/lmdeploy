import sys

import pytest

from lmdeploy.metrics.loggers import PrometheusStatLogger
from lmdeploy.metrics.stats import MultimodalStats
from lmdeploy.serve.processors import MultimodalProcessor
from lmdeploy.vl.constants import Modality

multimodal_module = sys.modules[MultimodalProcessor.__module__]


def test_multimodal_stats_snapshot_and_emit_guard():
    stats = MultimodalStats()

    stats.add_item(Modality.IMAGE.value, count=2)
    stats.add_stage('media_io', 0.1, Modality.IMAGE.value)
    stats.record_failure('media_io', Modality.IMAGE.value)
    stats.finish()

    total_time, stage_times, item_counts, failures = stats.snapshot()

    assert total_time >= 0
    assert item_counts == {'image': 2}
    assert stage_times == {('media_io', 'image'): 0.1}
    assert failures == {('media_io', 'image'): 1}
    assert stats.mark_emitted()
    assert not stats.mark_emitted()


def test_prometheus_logger_records_multimodal_metrics():
    pytest.importorskip('prometheus_client')
    from prometheus_client import REGISTRY, generate_latest

    logger = PrometheusStatLogger('test-model', 128, dp_rank=0)
    stats = MultimodalStats()
    stats.add_item(Modality.IMAGE.value, count=2)
    stats.add_stage('media_io', 0.1, Modality.IMAGE.value)
    stats.record_failure('media_io', Modality.IMAGE.value)
    stats.finish()

    logger.record_multimodal(stats)
    metrics_text = generate_latest(REGISTRY).decode('utf-8')

    assert 'lmdeploy:multimodal_requests_total' in metrics_text
    assert 'lmdeploy:multimodal_items_total' in metrics_text
    assert 'lmdeploy:multimodal_preprocess_time_seconds' in metrics_text
    assert 'lmdeploy:multimodal_stage_time_seconds' in metrics_text
    assert 'lmdeploy:multimodal_item_count' in metrics_text
    assert 'lmdeploy:multimodal_processing_failures_total' in metrics_text
    assert 'modality="image"' in metrics_text
    assert 'stage="media_io"' in metrics_text


def test_parse_multimodal_item_records_multimodal_stats(monkeypatch):
    load_calls = []

    def fake_load_from_url(data_src, media_io):
        load_calls.append((data_src, type(media_io).__name__))
        return f'loaded:{data_src}'

    monkeypatch.setattr(multimodal_module, 'load_from_url', fake_load_from_url)

    messages = [{
        'role': 'user',
        'content': [{
            'type': 'image',
            'image': 'file:///tmp/a.png',
        }]
    }]
    stats = MultimodalStats()
    parsed = [None]

    MultimodalProcessor._parse_multimodal_item(0, messages, parsed, {}, stats)
    _, stage_times, item_counts, failures = stats.snapshot()

    assert parsed[0]['content'][0] == {'type': Modality.IMAGE, 'data': 'loaded:file:///tmp/a.png'}
    assert load_calls == [('file:///tmp/a.png', 'ImageMediaIO')]
    assert item_counts == {'image': 1}
    assert ('media_io', 'image') in stage_times
    assert failures == {}
