# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .interns1_pro_ts_encoder import InternS1ProTimeSeriesModel
from .interns2_preview_ts_encoder import InternS2PreviewTimeSeriesModel
from .interns2_preview_ts_forecaster import InternS2PreviewTimeSeriesForecaster, TSForecasterConfig
from .patch import add_prefix, get_build_model_context
from .qwen3_5_moe import (
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeInputProcessor,
    Qwen3_5MoeModel,
    Qwen3_5MoeTextModel,
    Qwen3_5MoeVisionModel,
)


class InternS2PreviewModel(Qwen3_5MoeModel):

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        nn.Module.__init__(self)
        self.config = config

        self.visual = Qwen3_5MoeVisionModel(config.vision_config,
                                            dtype=dtype,
                                            device=device,
                                            prefix=add_prefix('visual', prefix))
        self.language_model = Qwen3_5MoeTextModel(config.text_config,
                                                  dtype=dtype,
                                                  device=device,
                                                  prefix=add_prefix('language_model', prefix))

        self.has_ts_forecaster = getattr(config, 'ts_forecaster_config', None) is not None
        if self.has_ts_forecaster:
            self.time_series = InternS2PreviewTimeSeriesModel(config.ts_config, dtype=dtype, device=device)
        else:
            self.time_series = InternS1ProTimeSeriesModel(config.ts_config, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any,
        state_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vis_cu_seqlens: torch.Tensor | None = None,
        vis_pos_emb: torch.Tensor | None = None,
        multimodal_mask: torch.Tensor | None = None,
        pos_embeds: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        all_routed_experts: torch.Tensor | None = None,
        return_input_embeds: bool = False,
        return_ts_context: bool = False,
        ts_values: torch.Tensor | None = None,
        ts_lens: torch.Tensor | None = None,
        ts_sr: torch.Tensor | None = None,
        ts_channels: torch.Tensor | None = None,
    ):
        output_inputs_embeds = None
        ts_context = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                dtype = inputs_embeds.dtype
                pixel_values = pixel_values.to(dtype)
                vis_pos_emb = (vis_pos_emb[0].to(dtype), vis_pos_emb[1].to(dtype))

                image_embeds = self.visual(pixel_values,
                                           cu_seqlens=vis_cu_seqlens,
                                           rotary_pos_emb=vis_pos_emb,
                                           pos_embeds=pos_embeds)

                split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
                image_embeds = torch.split(image_embeds, split_sizes)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, dtype)

                multimodal_mask = multimodal_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(multimodal_mask, image_embeds)
            elif ts_values is not None:
                if self.has_ts_forecaster:
                    ts_embeds, ts_pad_mask, ts_embeds_before_project = self.time_series(
                        ts_values, ts_lens, ts_sr, ts_channels)
                    ts_valid_mask = ~ts_pad_mask
                    ts_features = ts_embeds[ts_valid_mask].to(inputs_embeds.device, inputs_embeds.dtype)

                    ts_placeholder = input_ids == self.config.ts_token_id
                    n_ts_placeholders = ts_placeholder.sum().item()
                    n_ts_tokens = ts_features.size(0)
                    assert n_ts_placeholders == n_ts_tokens, (
                        f'Mismatch: <TS_CONTEXT> tokens={n_ts_placeholders}, ts_embeds_valid={n_ts_tokens}')

                    flat_embeds = inputs_embeds.reshape(-1, inputs_embeds.size(-1))
                    flat_embeds[ts_placeholder.reshape(-1)] = ts_features
                    inputs_embeds = flat_embeds.reshape_as(inputs_embeds)

                    if return_ts_context:
                        ts_context = dict(
                            ts_history=[
                                ts_values[i, :ts_lens[i], :ts_channels[i]] for i in range(ts_values.shape[0])
                            ],
                            ts_encoder_embedding=ts_embeds_before_project,
                            ts_encoder_embedding_mask=ts_valid_mask,
                        )
                else:
                    ts_embeds = self.time_series(ts_values, ts_lens, ts_sr).to(inputs_embeds.device,
                                                                               inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(multimodal_mask[..., None], ts_embeds)

        output_inputs_embeds = inputs_embeds if return_input_embeds else None

        hidden_states = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            state_ids=state_ids,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
            all_routed_experts=all_routed_experts,
        )
        return hidden_states, output_inputs_embeds, ts_context


class InternS2PreviewForConditionalGeneration(Qwen3_5MoeForConditionalGeneration):
    """ModelForCausalLM."""

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 prefix: str = ''):
        nn.Module.__init__(self)
        self.config = config
        self.ctx_mgr = ctx_mgr

        self.input_processor = InternS2PreviewInputProcessor(self.config, dtype)

        # build model
        self.model = InternS2PreviewModel(config, dtype=dtype, device=device, prefix=add_prefix('model', prefix))

        # build lm_head
        self.lm_head = self.build_lm_head(config.text_config.hidden_size,
                                          config.text_config.vocab_size,
                                          bias=False,
                                          dtype=dtype,
                                          device=device)

        # build time-series forecaster if configured
        self.has_ts_forecaster = getattr(config, 'ts_forecaster_config', None) is not None
        self.time_series_forecaster = None
        if self.has_ts_forecaster:
            forecaster_config = config.ts_forecaster_config.to_dict()
            self.time_series_forecaster = InternS2PreviewTimeSeriesForecaster(
                TSForecasterConfig(**forecaster_config),
                dtype=dtype,
                device=device,
            )

        # for router replay
        bm_ctx = get_build_model_context()
        self.enable_return_routed_experts = bm_ctx.enable_return_routed_experts
        self.is_spec_decoding = get_build_model_context().num_spec_tokens > 0

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any,
        state_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vis_cu_seqlens: torch.Tensor | None = None,
        vis_pos_emb: torch.Tensor | None = None,
        multimodal_mask: torch.Tensor | None = None,
        pos_embeds: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        return_input_embeds: bool = False,
        ts_values: torch.Tensor | None = None,
        ts_lens: torch.Tensor | None = None,
        ts_sr: torch.Tensor | None = None,
        ts_channels: torch.Tensor | None = None,
        ts_forecast_meta: dict[str, Any] | None = None,
        ts_forecast_llm_meta: dict[str, Any] | None = None,
        **kwargs,
    ):
        all_routed_experts = None
        if self.enable_return_routed_experts:
            config = self.config.text_config
            num_tokens = input_ids.size(1)
            all_routed_experts = position_ids.new_empty(
                (num_tokens, config.num_hidden_layers, config.num_experts_per_tok), dtype=torch.uint16)

        need_ts_context = ts_values is not None and ts_forecast_meta is not None
        hidden_states, target_inputs_embeds, ts_context = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            state_ids=state_ids,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            multimodal_mask=multimodal_mask,
            pos_embeds=pos_embeds,
            grid_thw=grid_thw,
            all_routed_experts=all_routed_experts,
            return_input_embeds=return_input_embeds,
            return_ts_context=need_ts_context,
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            ts_channels=ts_channels,
        )

        output = dict(hidden_states=hidden_states,
                      all_routed_experts=all_routed_experts,
                      target_inputs_embeds=target_inputs_embeds)

        if ts_forecast_llm_meta is not None:
            llm_hidden, llm_mask = self._select_ts_llm_hidden(
                hidden_states,
                ts_forecast_llm_meta['q_seqlens'],
                ts_forecast_llm_meta['batch_indices'],
            )
            output['ts_forecast_llm_hidden'] = llm_hidden
            output['ts_forecast_llm_mask'] = llm_mask

        if need_ts_context:
            q_seqlens = ts_forecast_meta['q_seqlens']
            ts_batch_indices = ts_forecast_meta['ts_batch_indices']
            llm_hidden, llm_mask = self._select_ts_llm_hidden(hidden_states, q_seqlens, ts_batch_indices)
            output['ts_forecast_state'] = dict(
                history=ts_context['ts_history'],
                llm_embedding_input=llm_hidden,
                llm_embedding_mask=llm_mask,
                ts_encoder_embedding=ts_context['ts_encoder_embedding'],
                ts_encoder_embedding_mask=ts_context['ts_encoder_embedding_mask'],
                ts_batch_indices=ts_batch_indices,
                forecast_horizon=ts_forecast_meta['forecast_horizon'],
                forecast_indices=ts_forecast_meta['forecast_indices'],
            )

        return output

    @staticmethod
    def _select_ts_llm_hidden(hidden_states: torch.Tensor, q_seqlens: torch.Tensor,
                              ts_batch_indices: torch.Tensor):
        # Select TS requests from packed prefill hidden states for the forecaster.
        q_lens = [int(length) for length in q_seqlens.tolist()]
        if hidden_states.size(0) == 1:
            chunks = hidden_states.squeeze(0).split(q_lens, dim=0)
        else:
            chunks = [hidden_states[idx, :length] for idx, length in enumerate(q_lens)]

        selected = [chunks[int(idx)] for idx in ts_batch_indices.tolist()]
        lengths = torch.tensor([item.size(0) for item in selected], device=hidden_states.device)
        padded = pad_sequence(selected, batch_first=True)
        mask = torch.arange(padded.size(1), device=hidden_states.device)[None, :] < lengths[:, None]
        return padded, mask

    def _run_ts_forecaster(self, forecast_state: dict[str, Any], item_indices: torch.Tensor):
        """Run forecaster for TS items requested by generation metadata."""

        def to_payload(forecast, idx: int):
            # Convert one forecast item from tensors to an API-serializable payload.
            predicted_horizon = None
            if forecast.predicted_horizon is not None:
                predicted_horizon = int(forecast.predicted_horizon[idx].item())
            return dict(
                point_forecast=forecast.point_forecast[idx].detach().float().cpu().tolist(),
                quantile_forecast=forecast.quantile_forecast[idx].detach().float().cpu().tolist(),
                predicted_horizon=predicted_horizon,
            )

        item_ids = item_indices.tolist()
        override_horizon = forecast_state['forecast_horizon']
        if override_horizon is not None:
            override_horizon = [override_horizon[int(idx)] for idx in item_ids]
            if all(horizon == override_horizon[0] for horizon in override_horizon):
                override_horizon = override_horizon[0]

        if isinstance(override_horizon, list) and any(horizon is None for horizon in override_horizon):
            payloads = []
            for item_idx, horizon in zip(item_ids, override_horizon):
                forecast = self.time_series_forecaster(
                    history=[forecast_state['history'][item_idx]],
                    llm_embedding_input=forecast_state['llm_embedding_input'][item_idx:item_idx + 1],
                    llm_embedding_mask=forecast_state['llm_embedding_mask'][item_idx:item_idx + 1],
                    ts_encoder_embedding_input=forecast_state['ts_encoder_embedding'][item_idx:item_idx + 1],
                    ts_encoder_embedding_mask=forecast_state['ts_encoder_embedding_mask'][item_idx:item_idx + 1],
                    override_horizon=horizon,
                )
                payloads.append(to_payload(forecast, 0))
            return payloads

        forecast = self.time_series_forecaster(
            history=[forecast_state['history'][idx] for idx in item_ids],
            llm_embedding_input=forecast_state['llm_embedding_input'][item_indices],
            llm_embedding_mask=forecast_state['llm_embedding_mask'][item_indices],
            ts_encoder_embedding_input=forecast_state['ts_encoder_embedding'][item_indices],
            ts_encoder_embedding_mask=forecast_state['ts_encoder_embedding_mask'][item_indices],
            override_horizon=override_horizon,
        )
        return [to_payload(forecast, idx) for idx in range(len(item_ids))]

    def run_ts_forecast_from_state(
        self,
        forecast_state: dict[str, Any],
        item_indices: torch.Tensor,
    ):
        """Run TS forecaster for manually-enabled forecast items."""
        assert self.has_ts_forecaster, 'This InternS2Preview checkpoint does not include a time-series forecaster.'
        ts_batch_indices = forecast_state['ts_batch_indices'].to(device=item_indices.device)
        payloads = self._run_ts_forecaster(forecast_state, item_indices)
        batch_indices = ts_batch_indices[item_indices]
        return payloads, batch_indices

    @staticmethod
    def _get_ts_forecast_cfg(
        multimodal_output_metas: list[dict[str, Any] | None] | None,
        batch_idx: int,
    ):
        if multimodal_output_metas is None:
            return None
        output_meta = multimodal_output_metas[batch_idx]
        if output_meta is None:
            return None
        ts_forecast_cfg = output_meta.get('time_series_forecast')
        if ts_forecast_cfg is None or not ts_forecast_cfg.get('enabled', False):
            return None
        return ts_forecast_cfg

    @staticmethod
    def _collate_time_series_values(mm_inputs):
        """Pad variable-length TS inputs before batching."""
        max_len = max(inp.data.size(1) for inp in mm_inputs)
        max_channels = max(inp.data.size(2) for inp in mm_inputs)
        padded_values = []
        for inp in mm_inputs:
            data = inp.data
            if data.size(1) == max_len and data.size(2) == max_channels:
                padded_values.append(data)
                continue
            padded = data.new_zeros((data.size(0), max_len, max_channels))
            padded[:, :data.size(1), :data.size(2)] = data
            padded_values.append(padded)
        return torch.cat(padded_values)

    @staticmethod
    def _split_multimodal_inputs(context: StepContext):
        vision_mm_inputs = []
        ts_mm_inputs = []
        ts_batch_indices = []
        if context.input_multimodals is None:
            return vision_mm_inputs, ts_mm_inputs, ts_batch_indices

        for batch_idx, input_mm in enumerate(context.input_multimodals):
            if input_mm is None:
                continue
            for item in input_mm.get('mm_data', []):
                if item.modality == Modality.TIME_SERIES:
                    ts_mm_inputs.append(item)
                    ts_batch_indices.append(batch_idx)
                else:
                    vision_mm_inputs.append(item)
        return vision_mm_inputs, ts_mm_inputs, ts_batch_indices

    def _prepare_vision_inputs(self, input_ids: torch.Tensor, mm_inputs: list[MultiModalData]):
        # same image/video preparation as Qwen3.5; TS is handled separately.
        multimodal_mask = self.get_multimodal_mask(input_ids, mm_inputs)
        pixel_values = torch.cat([inp.data for inp in mm_inputs])
        grid_thw = torch.stack([data.meta['grid_thw'] for data in mm_inputs]).cpu()
        vis_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
        pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
        vis_cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                 grid_thw[:, 0]).to(pixel_values.device)
        vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
        vis_pos_emb = vis_pos_emb.repeat(1, 2)
        vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())
        return dict(
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            multimodal_mask=multimodal_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
        )

    def _prepare_ts_inputs(self, input_ids: torch.Tensor, context: StepContext, ts_mm_inputs: list[MultiModalData],
                           batch_indices: list[int]):
        multimodal_output_metas = context.multimodal_output_metas
        multimodal_mask = None
        ts_values = None
        ts_lens = None
        ts_sr = None
        ts_channels = None
        ts_forecast_meta = None
        ts_forecast_llm_meta = None

        # chunked prefill can need LLM hidden states from chunks without raw TS tensors.
        if (context.is_chunk_multimodal and not context.is_decoding and not self.is_spec_decoding
                and multimodal_output_metas is not None):
            forecast_batch_ids = [
                idx for idx in range(len(multimodal_output_metas))
                if self._get_ts_forecast_cfg(multimodal_output_metas, idx)
            ]
            if forecast_batch_ids:
                ts_forecast_llm_meta = dict(
                    q_seqlens=context.q_seqlens,
                    batch_indices=torch.tensor(forecast_batch_ids, dtype=torch.long, device=input_ids.device),
                )

        if ts_mm_inputs:
            # time series samples can have different lengths or channel counts, so pad before batching.
            ts_values = self._collate_time_series_values(ts_mm_inputs)
            ts_lens = torch.cat([inp.meta['ts_lens'] for inp in ts_mm_inputs])
            ts_sr = torch.cat([inp.meta['ts_sr'] for inp in ts_mm_inputs])
            ts_channels = torch.cat([inp.meta['ts_channels'] for inp in ts_mm_inputs])
            multimodal_mask = self.get_multimodal_mask(input_ids, ts_mm_inputs)
            ts_batch_indices = torch.tensor(batch_indices, dtype=torch.long, device=input_ids.device)
            if not context.is_decoding and not self.is_spec_decoding:
                # heterogeneous TS batches keep forecast_indices for forecast-only rows.
                forecast_horizon = []
                forecast_ids = []
                for item_idx, batch_idx in enumerate(batch_indices):
                    ts_forecast_cfg = self._get_ts_forecast_cfg(multimodal_output_metas, batch_idx)
                    if ts_forecast_cfg is None:
                        forecast_horizon.append(None)
                        continue
                    forecast_horizon.append(ts_forecast_cfg.get('forecast_horizon'))
                    forecast_ids.append(item_idx)
                if not any(horizon is not None for horizon in forecast_horizon):
                    forecast_horizon = None
                forecast_indices = None
                if forecast_ids:
                    forecast_indices = torch.tensor(forecast_ids, dtype=torch.long, device=input_ids.device)
                if forecast_indices is not None:
                    ts_forecast_meta = dict(
                        q_seqlens=context.q_seqlens,
                        ts_batch_indices=ts_batch_indices,
                        forecast_horizon=forecast_horizon,
                        forecast_indices=forecast_indices,
                    )

        return dict(
            multimodal_mask=multimodal_mask,
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            ts_channels=ts_channels,
            ts_forecast_meta=ts_forecast_meta,
            ts_forecast_llm_meta=ts_forecast_llm_meta,
        )

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        """Prepare input."""
        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # make past_key_values
        state_caches = list(cache.transpose(0, 1) for cache in context.state_caches)
        state_caches = list(zip(state_caches[0], state_caches[1]))
        past_key_values = list(past_key_values)
        new_past_key_values = []
        for layer_type in self.config.text_config.layer_types:
            if layer_type == 'linear_attention':
                new_past_key_values.append(state_caches.pop(0))
            elif layer_type == 'full_attention':
                new_past_key_values.append(past_key_values.pop(0))

        vision_mm_inputs, ts_mm_inputs, ts_batch_indices = self._split_multimodal_inputs(context)
        # one forward can insert either vision embeddings or TS embeddings, not both.
        if vision_mm_inputs and ts_mm_inputs:
            raise ValueError('InternS2Preview does not support vision and time-series inputs in the same batch.')

        # vlm inputs
        vision_inputs = dict(
            pixel_values=None,
            vis_cu_seqlens=None,
            vis_pos_emb=None,
            multimodal_mask=None,
            grid_thw=None,
            pos_embeds=None,
        )
        if vision_mm_inputs:
            vision_inputs = self._prepare_vision_inputs(input_ids, vision_mm_inputs)

        mrope_position_ids = getattr(context, 'mrope_position_ids', None)

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # return input embeds for spec decoding
        return_input_embeds = self.is_spec_decoding and (vision_inputs['pixel_values'] is not None
                                                         or context.is_chunk_multimodal)

        # time series inputs
        ts_inputs = self._prepare_ts_inputs(input_ids, context, ts_mm_inputs, ts_batch_indices)
        ts_multimodal_mask = ts_inputs.pop('multimodal_mask')
        vision_multimodal_mask = vision_inputs.pop('multimodal_mask')
        multimodal_mask = ts_multimodal_mask if ts_multimodal_mask is not None else vision_multimodal_mask

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=new_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            # mm inputs
            mrope_position_ids=mrope_position_ids,
            multimodal_mask=multimodal_mask,
            return_input_embeds=return_input_embeds,
            **vision_inputs,
            **ts_inputs,
        )

    def _load_forecaster_weight(self, name: str, loaded_weight: torch.Tensor,
                                params_dict: dict[str, nn.Parameter], buffers_dict: dict[str, torch.Tensor]) -> bool:
        if not name.startswith('time_series_forecaster.'):
            return False

        if name.endswith('.in_proj_weight'):
            base = name[:-len('.in_proj_weight')]
            suffix = 'weight'
        elif name.endswith('.in_proj_bias'):
            base = name[:-len('.in_proj_bias')]
            suffix = 'bias'
        else:
            base = None

        if base is not None:
            # HF QFormer packs Q/K/V as in_proj_*; LMDeploy stores split Q/K/V params.
            qkv_names = tuple(f'{base}.{shard}_proj.{suffix}' for shard in ('q', 'k', 'v'))
            if all(param_name in params_dict for param_name in qkv_names):
                for param_name, shard_weight in zip(qkv_names, loaded_weight.chunk(3, dim=0)):
                    load_weight(params_dict[param_name], shard_weight)
                return True

        if name in params_dict:
            load_weight(params_dict[name], loaded_weight)
        elif name in buffers_dict:
            load_weight(buffers_dict[name], loaded_weight)
        else:
            raise KeyError(f'Unexpected weight name: {name}')
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())

        def remaining_weights():
            for name, loaded_weight in weights:
                if self._load_forecaster_weight(name, loaded_weight, params_dict, buffers_dict):
                    continue
                yield name, loaded_weight

        super().load_weights(remaining_weights())


class InternS2PreviewInputProcessor(Qwen3_5MoeInputProcessor):
    """InternS2Preview input processor with time-series support."""

    def _make_time_series_mm_data(self, input_mm: dict[str, Any]) -> MultiModalData:
        ts_values = input_mm['ts_values'].to(self.dtype)
        offset = input_mm['offset']
        ts_token_id = input_mm['ts_token_id']
        ts_lens = input_mm['ts_lens']
        ts_sr = input_mm['ts_sr']
        ts_channels = input_mm.get('ts_channels')

        meta = dict(ts_lens=ts_lens, ts_sr=ts_sr, ts_token_id=ts_token_id)
        if ts_channels is not None:
            meta['ts_channels'] = ts_channels
        return MultiModalData(modality=Modality.TIME_SERIES,
                              data=ts_values,
                              start=offset[0],
                              end=offset[1],
                              meta=meta)

    def preprocess_input(self,
                         input_ids: list[int],
                         input_multimodals: list[dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_mm_data = []
        for input_mm in input_multimodals:
            modality = input_mm.get('modality')
            if modality == Modality.IMAGE:
                mm_data = self._make_image_mm_data(input_mm)
            elif modality == Modality.VIDEO:
                mm_data = self._make_video_mm_data(input_mm)
            elif modality == Modality.TIME_SERIES:
                mm_data = self._make_time_series_mm_data(input_mm)
            else:
                raise ValueError(f'unsupported modality {modality}')
            input_mm_data.append(mm_data)

        return PreprocessInputResult(input_ids=input_ids, input_multimodals=dict(mm_data=input_mm_data))
