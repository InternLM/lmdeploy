# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch

from lmdeploy.pytorch.model_inputs import ModelInputs


class TSForecastPostprocessor:
    """Postprocesses manual-trigger time-series forecast outputs."""

    def __init__(self) -> None:
        self._prev_chunk_state: dict | None = None

    def update_chunk_state(self, inputs: ModelInputs, model_outputs: dict) -> dict:
        """Carry TS forecast context across chunked long-context prefill."""
        if not inputs.is_chunk:
            self._prev_chunk_state = None
            return model_outputs

        if inputs.is_first_chunk:
            self._prev_chunk_state = None

        llm_hidden = model_outputs.pop('ts_forecast_llm_hidden', None)
        llm_mask = model_outputs.pop('ts_forecast_llm_mask', None)
        forecast_state = model_outputs.pop('ts_forecast_state', None)

        if self._prev_chunk_state is None:
            self._prev_chunk_state = dict(llm_hidden=[], llm_mask=[], forecast_state=None)
        chunk_state = self._prev_chunk_state

        if llm_hidden is not None:
            chunk_state['llm_hidden'].append(llm_hidden)
            chunk_state['llm_mask'].append(llm_mask)

        if forecast_state is not None:
            forecast_state = dict(forecast_state)
            forecast_state.pop('llm_embedding_input', None)
            forecast_state.pop('llm_embedding_mask', None)
            chunk_state['forecast_state'] = forecast_state

        if not inputs.is_last_chunk:
            return model_outputs

        forecast_state = chunk_state.get('forecast_state')
        if forecast_state is not None and chunk_state['llm_hidden']:
            forecast_state = dict(forecast_state)
            forecast_state['llm_embedding_input'] = torch.cat(chunk_state['llm_hidden'], dim=1)
            forecast_state['llm_embedding_mask'] = torch.cat(chunk_state['llm_mask'], dim=1)
            model_outputs['ts_forecast_state'] = forecast_state

        self._prev_chunk_state = None
        return model_outputs

    def maybe_update_outputs_after_sampling(
        self,
        model: Any,
        model_outputs: dict,
        output_token_ids: torch.Tensor,
        stopped: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any] | None] | None]:
        """Optionally run TS forecast and mutate user-visible outputs."""
        if not hasattr(model, 'run_ts_forecast_from_state'):
            return output_token_ids, stopped, None

        forecast_state = model_outputs.get('ts_forecast_state')
        if forecast_state is None:
            return output_token_ids, stopped, None

        forecast_indices = forecast_state.get('forecast_indices')
        if forecast_indices is None:
            return output_token_ids, stopped, None

        item_indices = forecast_indices.to(device=output_token_ids.device)
        if item_indices.numel() == 0:
            return output_token_ids, stopped, None

        payloads, batch_indices = model.run_ts_forecast_from_state(forecast_state, item_indices)
        batch_indices = batch_indices.to(device=output_token_ids.device)

        output_token_ids = output_token_ids.clone()
        stopped = stopped.clone()
        multimodal_outputs = [None] * stopped.numel()

        # Forecast responses are multimodal-only: hide the sampled text token and finish the row.
        output_token_ids.view(-1)[batch_indices] = -1
        stopped.view(-1)[batch_indices] = True

        for batch_idx, payload in zip(batch_indices.tolist(), payloads):
            multimodal_outputs[int(batch_idx)] = dict(time_series_forecast=payload)

        return output_token_ids, stopped, multimodal_outputs
