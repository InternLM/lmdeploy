# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from lmdeploy.pytorch.engine.model_agent.ts_forecast import TSForecastPostprocessor
from lmdeploy.pytorch.model_inputs import ModelInputs
from lmdeploy.pytorch.strategies.ar.model_inputs import index_select_model_inputs, merge_model_inputs


def _make_inputs(**kwargs):
    values = dict(
        input_ids=torch.tensor([[1]]),
        seq_length=torch.tensor([1]),
        history_lengths=torch.tensor([0]),
        block_offsets=torch.tensor([[0]], dtype=torch.int32),
        is_decoding=False,
        num_ignored_history=torch.tensor([0]),
        max_q_seqlen=1,
        max_kv_seqlen=1,
        sum_kv_seqlen=1,
    )
    values.update(kwargs)
    return ModelInputs(**values)


class _FakeForecastModel:

    def __init__(self):
        self.item_indices = None

    def run_ts_forecast_from_state(self, forecast_state, item_indices):
        self.item_indices = item_indices
        batch_indices = forecast_state['ts_batch_indices'][item_indices]
        payloads = [dict(point_forecast=[float(idx)], quantile_forecast=[float(idx)]) for idx in item_indices.tolist()]
        return payloads, batch_indices


def test_ts_forecast_postprocessor_suppresses_only_forecast_rows():
    postprocessor = TSForecastPostprocessor()
    model = _FakeForecastModel()
    forecast_state = dict(
        ts_batch_indices=torch.tensor([0, 2]),
        forecast_indices=torch.tensor([1]),
    )

    output_token_ids, stopped, multimodal_outputs = postprocessor.maybe_update_outputs_after_sampling(
        model=model,
        model_outputs=dict(ts_forecast_state=forecast_state),
        output_token_ids=torch.tensor([10, 11, 12]),
        stopped=torch.tensor([False, False, False]),
    )

    assert output_token_ids.tolist() == [10, 11, -1]
    assert stopped.tolist() == [False, False, True]
    assert multimodal_outputs == [
        None,
        None,
        dict(time_series_forecast=dict(point_forecast=[1.0], quantile_forecast=[1.0])),
    ]
    assert model.item_indices.tolist() == [1]


def test_ts_forecast_postprocessor_carries_chunk_state_to_last_chunk():
    postprocessor = TSForecastPostprocessor()
    base_state = dict(
        history=['series'],
        llm_embedding_input=torch.ones(1, 1, 2),
        llm_embedding_mask=torch.ones(1, 1, dtype=torch.bool),
        ts_encoder_embedding=torch.ones(1, 1, 2),
        ts_encoder_embedding_mask=torch.ones(1, 1, dtype=torch.bool),
        ts_batch_indices=torch.tensor([0]),
        forecast_horizon=[4],
        forecast_indices=torch.tensor([0]),
    )

    first_outputs = dict(
        ts_forecast_llm_hidden=torch.ones(1, 2, 2),
        ts_forecast_llm_mask=torch.ones(1, 2, dtype=torch.bool),
        ts_forecast_state=base_state,
    )
    out = postprocessor.update_chunk_state(_make_inputs(is_chunk=True, is_first_chunk=True), first_outputs)
    assert 'ts_forecast_state' not in out

    last_outputs = dict(
        ts_forecast_llm_hidden=torch.full((1, 1, 2), 2.0),
        ts_forecast_llm_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    out = postprocessor.update_chunk_state(_make_inputs(is_chunk=True, is_last_chunk=True), last_outputs)

    forecast_state = out['ts_forecast_state']
    assert forecast_state['llm_embedding_input'].shape == (1, 3, 2)
    assert forecast_state['llm_embedding_input'].tolist() == [[[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]]]
    assert forecast_state['llm_embedding_mask'].shape == (1, 3)
    assert forecast_state['history'] == ['series']


def test_multimodal_output_metas_merge_and_index_select():
    forecast_meta = dict(time_series_forecast=dict(enabled=True, forecast_horizon=4))
    left = _make_inputs(
        input_ids=torch.tensor([[10, 11]]),
        seq_length=torch.tensor([1, 1]),
        history_lengths=torch.tensor([0, 0]),
        block_offsets=torch.tensor([[0], [1]], dtype=torch.int32),
        is_decoding=True,
        num_ignored_history=torch.tensor([0, 0]),
        max_q_seqlen=1,
        max_kv_seqlen=1,
        sum_kv_seqlen=2,
        multimodal_output_metas=[forecast_meta, None],
    )
    right = _make_inputs(
        input_ids=torch.tensor([[12]]),
        is_decoding=True,
        block_offsets=torch.tensor([[2]], dtype=torch.int32),
    )

    merged = merge_model_inputs(left, right)

    assert merged.input_ids.tolist() == [[10, 11, 12]]
    assert merged.multimodal_output_metas == [forecast_meta, None, None]

    selected = index_select_model_inputs(
        merged,
        torch.tensor([2, 0]),
        indice_cpu=np.array([2, 0]),
    )

    assert selected.input_ids.tolist() == [[12, 10]]
    assert selected.multimodal_output_metas == [None, forecast_meta]
