# Copyright (c) OpenMMLab. All rights reserved.
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .interns1_pro_ts import InternS1ProTimeSeriesModel
from .patch import add_prefix, get_build_model_context
from .qwen3_moe import Qwen3MoeModel
from .qwen3_vl import Qwen3VLVisionModel
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixinV1


class InternS1ProForConditionalGeneration(nn.Module, DeployModelMixinV1, CudaGraphMixin):
    """ModelForCausalLM."""

    packed_modules_mapping = {
        'qkv_proj': [
            'q_proj',
            'k_proj',
            'v_proj',
        ],
        'gate_up_proj': [
            'gate_proj',
            'up_proj',
        ],
    }

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 prefix: str = ''):
        super().__init__()

        self.config = config
        self.ctx_mgr = ctx_mgr

        # build preprocessor
        self.input_processor = InternS1ProInputProcessor(self.config, dtype)

        # build vision model
        self.visual = Qwen3VLVisionModel(
            config.vision_config,
            dtype=dtype,
            device=device,
            prefix=add_prefix('visual', prefix=prefix),
        )

        # build text model
        self.language_model = Qwen3MoeModel(config.text_config,
                                            dtype=dtype,
                                            device=device,
                                            prefix=add_prefix('language_model', prefix=prefix))

        # build lm_head
        self.lm_head = self.build_lm_head(config.text_config.hidden_size,
                                          config.text_config.vocab_size,
                                          bias=False,
                                          dtype=dtype,
                                          device=device)

        # build time series model
        if hasattr(config, 'ts_config'):
            self.time_series = InternS1ProTimeSeriesModel(config.ts_config, dtype=dtype, device=device)

        # for router replay
        bm_ctx = get_build_model_context()
        self.enable_return_routed_experts = bm_ctx.enable_return_routed_experts

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        vis_cu_seqlens: torch.Tensor = None,
        vis_pos_emb: torch.Tensor = None,
        multimodal_mask: torch.Tensor = None,
        pos_embeds: torch.Tensor = None,
        grid_thw: torch.Tensor = None,
        # for time series
        ts_values: torch.Tensor = None,
        ts_lens: torch.Tensor = None,
        ts_sr: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                dtype = inputs_embeds.dtype
                pixel_values = pixel_values.to(dtype)
                vis_pos_emb = (vis_pos_emb[0].to(dtype), vis_pos_emb[1].to(dtype))

                # get image embeds
                # different from qwen3vl, interns1_1 does not use deepstack visual embeds
                image_embeds, _ = self.visual(pixel_values,
                                              cu_seqlens=vis_cu_seqlens,
                                              rotary_pos_emb=vis_pos_emb,
                                              pos_embeds=pos_embeds)

                # split image embeds per sample
                split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
                image_embeds = torch.split(image_embeds, split_sizes)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, dtype)

                # mask and scatter to create final input embeddings
                multimodal_mask = multimodal_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(multimodal_mask, image_embeds)
            elif ts_values is not None:
                ts_embeds = self.time_series(ts_values, ts_lens, ts_sr)  # [B, T, C]
                inputs_embeds = inputs_embeds.masked_scatter(multimodal_mask[..., None], ts_embeds)

        # router replay
        all_routed_experts = None
        if self.enable_return_routed_experts:
            all_routed_experts = input_ids.new_empty((input_ids.size(1), self.config.text_config.num_hidden_layers,
                                                      self.config.text_config.num_experts_per_tok),
                                                     dtype=torch.uint16)

        hidden_states = self.language_model(input_ids=input_ids,
                                            position_ids=position_ids,
                                            past_key_values=past_key_values,
                                            attn_metadata=attn_metadata,
                                            inputs_embeds=inputs_embeds,
                                            all_routed_experts=all_routed_experts)

        if all_routed_experts is None:
            return hidden_states
        return dict(hidden_states=hidden_states, all_routed_experts=all_routed_experts)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def get_multimodal_mask(self, input_ids: torch.Tensor, mm_inputs: list[MultiModalData]) -> torch.Tensor:
        """Get position masks for vision tokens."""
        image_token_id = next((m.meta.get('image_token_id') for m in mm_inputs if m.modality == Modality.IMAGE), None)
        video_token_id = next((m.meta.get('video_token_id') for m in mm_inputs if m.modality == Modality.VIDEO), None)
        ts_token_id = next((m.meta.get('ts_token_id') for m in mm_inputs if m.modality == Modality.TIME_SERIES), None)

        image_mask, video_mask, ts_mask = None, None, None
        if image_token_id is not None:
            image_mask = (input_ids == image_token_id)
        if video_token_id is not None:
            video_mask = (input_ids == video_token_id)
        if ts_token_id is not None:
            ts_mask = (input_ids == ts_token_id)

        multimodal_mask = None
        if image_mask is not None and video_mask is not None:
            multimodal_mask = image_mask | video_mask
        elif image_mask is not None:
            multimodal_mask = image_mask
        elif video_mask is not None:
            multimodal_mask = video_mask
        elif ts_mask is not None:
            multimodal_mask = ts_mask

        return multimodal_mask

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext = None,
    ):
        """Prepare input."""

        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        pixel_values = None
        vis_cu_seqlens = None
        vis_pos_emb = None
        multimodal_mask = None
        grid_thw = None
        pos_embeds = None
        # for time series
        ts_values = None
        ts_lens = None
        ts_sr = None
        if context.input_multimodals is not None:
            mm_inputs = [input_mm.get('mm_data', []) for input_mm in context.input_multimodals]
            # flatten batch
            mm_inputs = [item for sublist in mm_inputs for item in sublist]

            if len(mm_inputs) > 0:
                modality = mm_inputs[0].modality
                multimodal_mask = self.get_multimodal_mask(input_ids, mm_inputs)

                if modality == Modality.TIME_SERIES:
                    ts_values = torch.cat([inp.data for inp in mm_inputs])
                    ts_lens = mm_inputs[0].meta['ts_lens']
                    ts_sr = mm_inputs[0].meta['ts_sr']
                else:
                    pixel_values = torch.cat([inp.data for inp in mm_inputs])
                    grid_thw = torch.stack([data.meta['grid_thw'] for data in mm_inputs]).cpu()
                    vis_pos_emb = self.visual.rot_pos_emb(grid_thw)
                    pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)
                    vis_cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                             grid_thw[:, 0]).to(pixel_values.device)
                    vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                    vis_pos_emb = vis_pos_emb.repeat(1, 2)
                    vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            multimodal_mask=multimodal_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
            # for time series
            ts_values=ts_values,
            ts_lens=ts_lens,
            ts_sr=ts_sr,
        )

    @classmethod
    def rename_weight(cls, name: str) -> str:
        """Rename weight."""
        if name.startswith('model.language_model.'):
            return 'language_model.' + name[len('model.language_model.'):]
        elif name.startswith('model.visual.'):
            return 'visual.' + name[len('model.visual.'):]
        elif name.startswith('model.'):
            return name[len('model.'):]
        return name

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                             expert_params_mapping: list):
        """Load weight experts."""

        for (param_name, weight_name, expert_id, shard_id) in expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]
            load_weight(param, loaded_weight, expert_id=expert_id, shard_id=shard_id)
            break
        else:
            param = params_dict[name]
            load_weight(param, loaded_weight)

    # modify from vllm qwen3vlmoe fused expert loading
    def _load_weight_fused_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: dict[str, nn.Parameter],
                                   fused_expert_params_mapping: list):
        """Load weight of fused expert weights."""
        num_experts = self.config.text_config.num_experts

        for (param_name, weight_name) in fused_expert_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            param = params_dict[name]

            loaded_weight = loaded_weight.transpose(-1, -2)  # no bias
            if 'gate_up' in name:
                loaded_weight = loaded_weight.chunk(2, dim=-2)
                w1 = loaded_weight[0]
                w3 = loaded_weight[1]
                for expert_id in range(num_experts):
                    load_weight(param, w1[expert_id], expert_id=expert_id, shard_id='gate')
                    load_weight(param, w3[expert_id], expert_id=expert_id, shard_id='up')
            elif 'down' in name:
                w2 = loaded_weight
                for expert_id in range(num_experts):
                    load_weight(param, w2[expert_id], expert_id=expert_id, shard_id='down')

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights."""
        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
        ]

        # expert mapping
        num_experts = self.config.text_config.num_experts
        expert_params_mapping = []
        for exp_id in range(num_experts):
            # (param_name, weight_name, expert_id, shard_id)
            gate_param = ('.experts.gate_up', f'.experts.{exp_id}.gate_proj', exp_id, 'gate')
            up_param = ('.experts.gate_up', f'.experts.{exp_id}.up_proj', exp_id, 'up')
            down_param = ('.experts.down', f'.experts.{exp_id}.down_proj', exp_id, 'down')
            expert_params_mapping += [gate_param, up_param, down_param]

        # fused expert mapping
        fused_expert_params_mapping = [
            # (param_name, weight_name)
            ('.experts.gate_up.weight', '.experts.gate_up_proj'),
            ('.experts.down.weight', '.experts.down_proj'),
        ]

        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
            name = name.replace('.block_sparse_moe.', '.mlp.')
            if '.experts' in name:
                is_fused_expert = ('experts.gate_up_proj' in name or 'experts.down_proj' in name)
                if is_fused_expert:
                    self._load_weight_fused_experts(name,
                                                    loaded_weight,
                                                    params_dict,
                                                    fused_expert_params_mapping=fused_expert_params_mapping)
                else:
                    self._load_weight_experts(name,
                                              loaded_weight,
                                              params_dict,
                                              expert_params_mapping=expert_params_mapping)
            else:
                for (param_name, weight_name, shard_id) in stacked_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    load_weight(param, loaded_weight, shard_id=shard_id)
                    break
                else:
                    if '.qkv.' in name:
                        param = params_dict[name]
                        q, k, v = param.weight_spliter(loaded_weight)
                        load_weight(param, q, shard_id='q')
                        load_weight(param, k, shard_id='k')
                        load_weight(param, v, shard_id='v')
                    else:
                        if name in params_dict:
                            param = params_dict[name]
                            load_weight(param, loaded_weight)
                        elif name in buffers_dict:
                            param = buffers_dict[name]
                            load_weight(param, loaded_weight)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class InternS1ProInputProcessor(BaseModelInputProcessor):
    """InternS1Pro input processor."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype) -> None:
        self.config = config
        self.dtype = dtype

    def _make_image_mm_data(self, input_mm: dict[str, Any]) -> MultiModalData:
        """Make image MultiModalData."""
        pixel_values = input_mm['pixel_values'].to(self.dtype)
        image_grid_thw = input_mm['image_grid_thw']
        offset = input_mm['offset']
        image_token_id = input_mm['image_token_id']

        mm_data = MultiModalData(modality=Modality.IMAGE,
                                 data=pixel_values,
                                 start=offset[0],
                                 end=offset[1],
                                 meta=dict(grid_thw=image_grid_thw, image_token_id=image_token_id))
        return mm_data

    def _make_video_mm_data(self, input_mm: dict[str, Any]) -> MultiModalData:
        """Make video MultiModalData."""
        pixel_values_videos = input_mm['pixel_values_videos'].to(self.dtype)
        video_grid_thw = input_mm['video_grid_thw']
        offset = input_mm['offset']
        video_token_id = input_mm['video_token_id']

        mm_data = MultiModalData(modality=Modality.VIDEO,
                                 data=pixel_values_videos,
                                 start=offset[0],
                                 end=offset[1],
                                 meta=dict(
                                     grid_thw=video_grid_thw,
                                     video_token_id=video_token_id,
                                 ))
        return mm_data

    def _make_time_series_mm_data(self, input_mm: dict[str, Any]) -> MultiModalData:
        """Make time series MultiModalData."""
        ts_values = input_mm['ts_values'].to(self.dtype)
        offset = input_mm['offset']
        ts_token_id = input_mm['ts_token_id']
        ts_lens = input_mm['ts_lens']
        ts_sr = input_mm['ts_sr']

        mm_data = MultiModalData(modality=Modality.TIME_SERIES,
                                 data=ts_values,
                                 start=offset[0],
                                 end=offset[1],
                                 meta=dict(ts_lens=ts_lens, ts_sr=ts_sr, ts_token_id=ts_token_id))
        return mm_data

    def preprocess_input(self,
                         input_ids: list[int],
                         input_multimodals: list[dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""
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
            input_mm_data.append(mm_data)

        result = PreprocessInputResult(input_ids=input_ids, input_multimodals=dict(mm_data=input_mm_data))

        return result
