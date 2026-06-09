# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .interns2_ts_encoder import InternS2PreviewTimeSeriesModel
from .interns2_ts_forecaster import InternS2PreviewTimeSeriesForecaster, TSForecasterConfig
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
        self.time_series = InternS2PreviewTimeSeriesModel(config.ts_config, dtype=dtype, device=device)

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

        self.input_processor = Qwen3_5MoeInputProcessor(self.config, dtype)

        # build model
        self.model = InternS2PreviewModel(config, dtype=dtype, device=device, prefix=add_prefix('model', prefix))

        # build lm_head
        self.lm_head = self.build_lm_head(config.text_config.hidden_size,
                                          config.text_config.vocab_size,
                                          bias=False,
                                          dtype=dtype,
                                          device=device)

        # build time series forecaster
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
        forecast_horizon: int | list[int] | None = None,
        **kwargs,
    ):
        all_routed_experts = None
        if self.enable_return_routed_experts:
            config = self.config.text_config
            num_tokens = input_ids.size(1)
            all_routed_experts = position_ids.new_empty(
                (num_tokens, config.num_hidden_layers, config.num_experts_per_tok), dtype=torch.uint16)

        need_forecast = forecast_horizon is not None and ts_values is not None
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
            return_ts_context=need_forecast,
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
            ts_channels=ts_channels,
        )

        output = dict(hidden_states=hidden_states,
                      all_routed_experts=all_routed_experts,
                      target_inputs_embeds=target_inputs_embeds)

        if need_forecast:
            forecast = self.time_series_forecaster(
                history=ts_context['ts_history'],
                llm_embedding_input=hidden_states,
                ts_encoder_embedding_input=ts_context['ts_encoder_embedding'],
                ts_encoder_embedding_mask=ts_context['ts_encoder_embedding_mask'],
                override_horizon=forecast_horizon,
            )
            output.update(ts_point_forecast=forecast.point_forecast,
                          ts_quantile_forecast=forecast.quantile_forecast,
                          ts_predicted_horizon=forecast.predicted_horizon)

        return output

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
    ):
        model_inputs = super().prepare_inputs_for_generation(past_key_values, inputs_embeds, context)

        ts_channels = None
        if context.input_multimodals is not None:
            mm_inputs = [input_mm.get('mm_data', []) for input_mm in context.input_multimodals]
            mm_inputs = [item for sublist in mm_inputs for item in sublist]
            if len(mm_inputs) > 0 and mm_inputs[0].modality == Modality.TIME_SERIES:
                ts_channels = torch.cat([inp.meta['ts_channels'] for inp in mm_inputs])

        model_inputs['ts_channels'] = ts_channels
        return model_inputs

    def _load_forecaster_in_proj(self, name: str, loaded_weight: torch.Tensor,
                                 params_dict: dict[str, nn.Parameter]) -> bool:
        if not name.startswith('time_series_forecaster.'):
            return False

        if name.endswith('.in_proj_weight'):
            base = name[:-len('.in_proj_weight')]
            suffix = 'weight'
        elif name.endswith('.in_proj_bias'):
            base = name[:-len('.in_proj_bias')]
            suffix = 'bias'
        else:
            return False

        q, k, v = loaded_weight.chunk(3, dim=0)
        load_weight(params_dict[f'{base}.q_proj.{suffix}'], q)
        load_weight(params_dict[f'{base}.k_proj.{suffix}'], k)
        load_weight(params_dict[f'{base}.v_proj.{suffix}'], v)
        return True

    def _load_forecaster_weight(self, name: str, loaded_weight: torch.Tensor,
                                params_dict: dict[str, nn.Parameter], buffers_dict: dict[str, torch.Tensor]) -> bool:
        if not name.startswith('time_series_forecaster.'):
            return False

        if self._load_forecaster_in_proj(name, loaded_weight, params_dict):
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


InternS2PreviewForCausalLM = InternS2PreviewForConditionalGeneration


__all__ = ['InternS2PreviewForConditionalGeneration', 'InternS2PreviewForCausalLM']
