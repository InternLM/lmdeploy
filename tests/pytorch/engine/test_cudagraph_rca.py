# Copyright (c) OpenMMLab. All rights reserved.
import json
import re
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.backends.cuda import graph_runner as gr


class CaptureLogger:

    def __init__(self):
        self.messages = []

    def error(self, msg, *args):
        if args:
            msg = msg % args
        self.messages.append(msg)

    def exception(self, msg, *args):
        self.error(msg, *args)


class Qwen3_5MTPModel:
    def __call__(self, **kwargs):
        return {'called_with': kwargs}

    def make_output_buffers(self, output):
        return {'buffers': output}


class Qwen3_5Model(Qwen3_5MTPModel):
    pass


class FakeDP:

    def __init__(self, dp_is_decoding=True):
        self.dp_is_decoding = dp_is_decoding
        self.dp_batches = [2, 2]
        self.tp_sizes = [8]
        self.moe_tp_sizes = [8]
        self.dp_draft_num_tokens = [4, 4]


class FakeContext:

    sum_kv_seqlen = 24
    max_kv_seqlen = 12
    is_dummy = True
    is_chunk = True
    is_first_chunk = False
    is_last_chunk = True
    is_chunk_multimodal = False

    def __init__(self, *, local_is_decoding=True, global_is_decoding=True):
        self.is_decoding = local_is_decoding
        self.dp_meta = FakeDP(dp_is_decoding=global_is_decoding)
        self.q_seqlens = torch.tensor([4, 4])
        self.kv_seqlens = torch.tensor([12, 12])
        self.q_start_loc = torch.tensor([0, 4])
        self.block_offsets = torch.zeros((2, 2), dtype=torch.int32)

    def global_is_decoding(self):
        return self.dp_meta.dp_is_decoding


def make_runner(monkeypatch, logger=None, model=None):
    monkeypatch.setenv('LMDEPLOY_CUDAGRAPH_RCA_TRACE', '1')
    gr._CUDAGRAPH_RCA_DUMPED = False
    gr._CUDAGRAPH_RCA_RUNNERS.clear()
    if logger is not None:
        monkeypatch.setattr(gr, 'logger', logger)

    runner = object.__new__(gr.CUDAGraphRunner)
    runner.model = model or Qwen3_5MTPModel()
    runner.backend_config = SimpleNamespace(eager_mode=True)
    runner.ctx_mgr = SimpleNamespace(current_context=lambda: FakeContext())
    runner.enable_graph = lambda **kwargs: True
    runner._prepare_inputs = lambda **kwargs: kwargs
    runner._runner_map = {}
    runner.num_blocks = 12345
    runner._rca_forward_records = deque(maxlen=gr._rca_record_limit())
    runner._rca_forward_seq = 0
    runner._rca_last_deepep_mode = 'LOW_LATENCY'
    runner._rca_forward_total = 0
    runner._rca_local_prefill_forward_count = 0
    runner._rca_local_decode_forward_count = 0
    runner._rca_global_prefill_forward_count = 0
    runner._rca_global_decode_forward_count = 0
    runner._rca_deepep_normal_forward_count = 0
    runner._rca_deepep_low_latency_forward_count = 0
    runner._rca_deepep_none_forward_count = 0
    gr._CUDAGRAPH_RCA_RUNNERS.add(runner)
    return runner


def make_forward_kwargs():
    return dict(
        input_ids=torch.ones((1, 8), dtype=torch.long),
        position_ids=torch.arange(8, dtype=torch.long).view(1, 8),
        attn_metadata=SimpleNamespace(q_seqlens=torch.tensor([4, 4])),
    )


def test_rca_forward_record_has_required_fields_without_tensor_data(monkeypatch):
    runner = make_runner(monkeypatch)

    runner._record_forward(
        FakeContext(),
        make_forward_kwargs(),
        graph_action='replay',
        enable_graph=True,
        graph_key=(2, True, False, 4),
    )

    record = runner._rca_forward_records[-1]
    required_fields = {
        'seq',
        'timestamp',
        'pid',
        'rank',
        'local_rank',
        'runner_id',
        'role',
        'model_class',
        'kind',
        'local_is_decoding',
        'global_is_decoding',
        'forward_counters',
        'deepep_mode',
        'num_tokens',
        'batch_size',
        'query_len',
        'max_kv_seqlen',
        'sum_kv_seqlen',
        'is_dummy',
        'is_chunk',
        'is_first_chunk',
        'is_last_chunk',
        'is_chunk_multimodal',
        'dp_meta',
        'enable_graph',
        'graph_key',
        'graph_action',
        'single_runner_id',
        'runner_map_size',
        'num_gpu_blocks',
    }
    assert required_fields <= set(record)
    assert 'tensor_meta' not in record
    assert record['role'] == 'draft'
    assert record['model_class'] == 'Qwen3_5MTPModel'
    assert record['deepep_mode'] == 'LOW_LATENCY'
    assert record['local_is_decoding'] is True
    assert record['global_is_decoding'] is True
    assert record['is_dummy'] is True
    assert record['is_chunk'] is True
    assert record['is_first_chunk'] is False
    assert record['is_last_chunk'] is True
    assert record['dp_meta']['dp_is_decoding'] is True
    assert record['dp_meta']['dp_batches'] == [2, 2]
    assert record['forward_counters'] == {
        'total': 1,
        'local_prefill': 0,
        'local_decode': 1,
        'global_prefill': 0,
        'global_decode': 1,
        'deepep_normal': 0,
        'deepep_low_latency': 1,
        'deepep_none': 0,
    }
    assert record['num_tokens'] == 8
    assert record['batch_size'] == 2
    assert record['query_len'] == 4
    serialized = gr._json_line(record)
    assert '"value"' not in serialized
    assert 'data_ptr' not in serialized


def test_rca_forward_counters_track_local_and_global_prefill_decode(monkeypatch):
    runner = make_runner(monkeypatch)
    kwargs = make_forward_kwargs()

    runner._rca_last_deepep_mode = 'NORMAL'
    runner._record_forward(
        FakeContext(local_is_decoding=False, global_is_decoding=False),
        kwargs,
        graph_action='eager',
        enable_graph=False,
    )
    runner._rca_last_deepep_mode = 'NORMAL'
    runner._record_forward(
        FakeContext(local_is_decoding=True, global_is_decoding=False),
        kwargs,
        graph_action='eager',
        enable_graph=False,
    )
    runner._rca_last_deepep_mode = 'LOW_LATENCY'
    runner._record_forward(
        FakeContext(local_is_decoding=True, global_is_decoding=True),
        kwargs,
        graph_action='replay',
        enable_graph=True,
        graph_key=(2, True, False, 4),
    )

    assert runner._rca_forward_records[0]['forward_counters'] == {
        'total': 1,
        'local_prefill': 1,
        'local_decode': 0,
        'global_prefill': 1,
        'global_decode': 0,
        'deepep_normal': 1,
        'deepep_low_latency': 0,
        'deepep_none': 0,
    }
    assert runner._rca_forward_records[1]['forward_counters'] == {
        'total': 2,
        'local_prefill': 1,
        'local_decode': 1,
        'global_prefill': 2,
        'global_decode': 0,
        'deepep_normal': 2,
        'deepep_low_latency': 0,
        'deepep_none': 0,
    }
    assert runner._rca_forward_records[2]['forward_counters'] == {
        'total': 3,
        'local_prefill': 1,
        'local_decode': 2,
        'global_prefill': 2,
        'global_decode': 1,
        'deepep_normal': 2,
        'deepep_low_latency': 1,
        'deepep_none': 0,
    }


def test_rca_forward_records_are_capped_at_recent_1000(monkeypatch):
    monkeypatch.setenv('LMDEPLOY_CUDAGRAPH_RCA_RECORDS', '5000')
    runner = make_runner(monkeypatch)

    for idx in range(1005):
        runner._append_rca_record({'kind': 'forward', 'idx': idx})

    records = list(runner._rca_forward_records)
    assert len(records) == 1000
    assert records[0]['idx'] == 5
    assert records[-1]['idx'] == 1004


@pytest.mark.parametrize(
    'message',
    [
        'DeepEP error: timeout (dispatch CPU)',
        'DeepEP error: CPU recv timeout',
        'CUDA error: unspecified launch failure',
    ],
)
def test_cudagraph_rca_dump_triggers_from_call_for_fatal_errors(monkeypatch, message):
    logger = CaptureLogger()
    runner = make_runner(monkeypatch, logger=logger)
    runner._append_rca_record({'kind': 'forward', 'marker': 'recent-forward'})

    def raise_from_prepare_inputs(**kwargs):
        raise RuntimeError(message)

    runner._prepare_inputs = raise_from_prepare_inputs

    with pytest.raises(RuntimeError, match=re.escape(message)):
        gr.CUDAGraphRunner.__call__(runner)

    output = '\n'.join(logger.messages)
    assert '[CUDAGRAPH_RCA_DUMP]' in output
    assert '[CUDAGRAPH_RCA_RUNNER]' in output
    assert '[CUDAGRAPH_RCA_RECORD]' in output
    assert message in output
    assert 'recent-forward' in output


def test_cudagraph_rca_dump_does_not_trigger_for_nonfatal_error(monkeypatch):
    logger = CaptureLogger()
    runner = make_runner(monkeypatch, logger=logger)
    runner._append_rca_record({'kind': 'forward', 'marker': 'recent-forward'})

    def raise_from_prepare_inputs(**kwargs):
        raise RuntimeError('ordinary validation failure')

    runner._prepare_inputs = raise_from_prepare_inputs

    with pytest.raises(RuntimeError, match='ordinary validation failure'):
        gr.CUDAGraphRunner.__call__(runner)

    output = '\n'.join(logger.messages)
    assert '[CUDAGRAPH_RCA_DUMP]' not in output
    assert '[CUDAGRAPH_RCA_RECORD]' not in output


def test_cudagraph_rca_dump_writes_rank_jsonl_from_lmdeploy_log_file(monkeypatch, tmp_path):
    logger = CaptureLogger()
    monkeypatch.setenv('LMDEPLOY_LOG_FILE', str(tmp_path / 'server.log'))
    monkeypatch.setenv('RANK', '3')
    monkeypatch.setenv('LOCAL_RANK', '1')
    runner = make_runner(monkeypatch, logger=logger)
    runner._append_rca_record({'kind': 'forward', 'marker': 'recent-forward'})

    gr._dump_all_cudagraph_rca_records('DeepEP error: timeout (dispatch CPU)')

    files = list(tmp_path.glob('server.log.cudagraph_rca.rank3.local1.pid*.jsonl'))
    assert len(files) == 1
    records = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert [record['event'] for record in records] == ['dump', 'runner', 'record']
    assert records[0]['reason'] == 'DeepEP error: timeout (dispatch CPU)'
    assert records[0]['rank'] == '3'
    assert records[0]['local_rank'] == '1'
    assert records[2]['marker'] == 'recent-forward'
    assert str(files[0]) in '\n'.join(logger.messages)
