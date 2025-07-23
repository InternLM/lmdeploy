# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek_vl_v2.py

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_v2 import DeepseekV2ForCausalLM
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


@vlm_model
class MlpProjector(nn.Module):

    def __init__(self, cfg, dtype):

        super().__init__()

        self.cfg = cfg

        if cfg.projector_type == 'identity':
            modules = nn.Identity()

        elif cfg.projector_type == 'linear':
            modules = nn.Linear(cfg.input_dim, cfg.n_embed, dtype=dtype)

        elif cfg.projector_type == 'mlp_gelu':
            mlp_depth = cfg.depth
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed, dtype=dtype)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed, dtype=dtype))
            modules = nn.Sequential(*modules)

        elif cfg.projector_type == 'downsample_mlp_gelu':
            mlp_depth = cfg.depth
            mlp_ratio = cfg.mlp_ratio
            modules = [
                nn.Linear(cfg.input_dim * cfg.downsample_ratio * cfg.downsample_ratio,
                          cfg.n_embed * mlp_ratio,
                          dtype=dtype)
            ]
            for _ in range(1, mlp_depth - 1):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed * mlp_ratio, dtype=dtype))
            modules.append(nn.GELU())
            modules.append(nn.Linear(cfg.n_embed * mlp_ratio, cfg.n_embed, dtype=dtype))
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f'Unknown projector type: {cfg.projector_type}')

        if cfg.token_pooling:
            self.token_pooling_layer = nn.Linear(cfg.input_dim * 4, cfg.input_dim, dtype=dtype)

        self.layers = modules

    def forward(self, x):
        if self.cfg.token_pooling:
            batch_size, wxh, channels = x.shape
            w = h = int(wxh**0.5)
            x = x.view(batch_size, w, h, channels)
            x = x.permute(0, 3, 1, 2)
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # concatenate along the channel dimension
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # pass through the linear layer
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw)**0.5)
            """Compute padding."""
            if h % self.cfg.downsample_ratio:
                pad = self.cfg.downsample_ratio - h % self.cfg.downsample_ratio
            else:
                pad = 0
            x = x.reshape(bs, h, w, input_dim)
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad, 0, pad), 'constant', 0)
            """4 to 1 concat"""
            x = x.permute(0, 3, 1, 2)  # B, C, H, W
            x = F.unfold(x, kernel_size=self.cfg.downsample_ratio, stride=self.cfg.downsample_ratio,
                         padding=0)  # B, C*4, HW // 4
            x = x.permute(0, 2, 1)

        return self.layers(x)


class DeepseekVLV2ForCausalLM(nn.Module, CudaGraphMixin, DeployModelMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        # ----------- vision encoder ------------
        self.vision = self._init_vision_module(dtype=dtype)

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config, dtype)

        # image token format
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # special tokens used to format image token sequence
        embed_std = 1 / torch.sqrt(torch.tensor(projector_config.n_embed, dtype=torch.float32))
        if self.tile_tag == '2D':
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
            # fix the typo: view_seperater
            self.view_seperator = nn.Parameter(torch.randn(projector_config.n_embed) * embed_std)
        elif self.tile_tag == '1D':
            # <|tile_x|>, <|tile_global|>
            candidate_resolutions = config.candidate_resolutions
            if len(candidate_resolutions) == 0:
                raise ValueError(
                    f'len(candidate_resolutions) should be larger than 0, but got {len(candidate_resolutions)}')
            tile_variants_num = len(candidate_resolutions)
            self.tile_indicators = nn.Parameter(
                torch.randn(size=(tile_variants_num + 1, config.aligner.params.n_embed)) * embed_std)
        else:
            raise ValueError(f'tile tag should be either 1D or 2D, but got {self.tile_tag}')

        # ----------- language model ------------
        language_config = config.language_config
        self.language = DeepseekV2ForCausalLM(config=language_config, ctx_mgr=ctx_mgr, dtype=dtype, device=device)

        #  ----------- input processor ------------
        self.input_processor = DeepSeekVLV2InputProcessor(config, dtype)

    # adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_vl2.py#L359
    def _init_vision_module(
        self,
        dtype: torch.dtype,
    ) -> nn.Module:
        try:
            import timm
        except ImportError:
            raise ImportError('Please install timm') from ImportError

        model = timm.create_model(
            'vit_so400m_patch14_siglip_384.webli',
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        model = model.to(dtype=dtype)
        return model

    def prepare_inputs_embeds(self,
                              input_ids: torch.LongTensor,
                              images: Optional[torch.FloatTensor] = None,
                              images_seq_mask: Optional[torch.LongTensor] = None,
                              images_spatial_crop: Optional[torch.LongTensor] = None,
                              **ignore_kwargs):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            images (torch.FloatTensor): [b, max_n_images, 3, height, width]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_spatial_crop (torch.LongTensor): [b, max_n_images, 2]

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        bs, max_n_images, _ = images_spatial_crop.shape
        batch_num_tiles = [0 for _ in range(bs)]
        total_tiles = []
        for idx in range(bs):
            for jdx in range(max_n_images):
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break
                batch_num_tiles[idx] += (1 + num_width_tiles * num_height_tiles)

            total_tiles.append(images[idx, :batch_num_tiles[idx]])

        # [batch_all_tiles, 3, height, width]
        total_tiles = torch.cat(total_tiles, dim=0)
        assert total_tiles.shape[0] == sum(batch_num_tiles)
        if total_tiles.shape[0] == 0:
            return self.language.get_input_embeddings()(input_ids)

        # [batch_all_tiles, vit_seq_len, c]
        images_feature = self.vision.forward_features(total_tiles)  # timm siglip forward_features

        # [batch_all_tiles, hw, D]
        images_embeds = self.projector(images_feature)
        _, hw, n_dim = images_embeds.shape
        h = w = int(hw**0.5)

        # put image tokens into the input_embeds, [b, T, D]
        input_embeds = self.language.get_input_embeddings()(input_ids)

        # fill image token sequence according to self.tile_tag & self.global_view_pos
        tile_index = 0
        for idx in range(images_spatial_crop.shape[0]):
            images_in_this_batch = []
            for jdx in range(images_spatial_crop.shape[1]):

                # extra global & local features
                num_width_tiles, num_height_tiles = images_spatial_crop[idx, jdx]
                if num_width_tiles == 0 or num_height_tiles == 0:
                    break

                num_tiles_in_image = num_width_tiles * num_height_tiles

                # [hw, D]
                global_features = images_embeds[tile_index]

                # [num_height_tiles * num_width_tiles, hw, D]
                local_features = images_embeds[tile_index + 1:tile_index + 1 + num_tiles_in_image]

                tile_index += num_tiles_in_image + 1

                # format global and local features
                if self.tile_tag == '2D':

                    # ----------------- global view add newline -----------------
                    # [hw, D] -> [h, w, D]
                    global_features = global_features.view(h, w, n_dim)
                    # [D]     -> [h, 1, D]
                    new_lines_in_global = repeat(self.image_newline, 'd -> h 1 d', h=h)
                    # cat([h, w, D], [h, 1, D], dim=1) -> [h, w + 1, D]
                    global_features = torch.cat([global_features, new_lines_in_global], dim=1)
                    # [h, w + 1, D] -> [h * (w + 1), D]
                    global_features = global_features.view(-1, n_dim)

                    # ----------------- local view add newline -----------------
                    # [num_height_tiles * num_width_tiles, h * w, D] -> [num_height_tiles * h, num_width_tiles * w, D]
                    local_features = rearrange(local_features,
                                               '(th tw) (h w) d -> (th h) (tw w) d',
                                               th=num_height_tiles,
                                               tw=num_width_tiles,
                                               h=h,
                                               w=w)

                    # [D] -> [num_height_tiles * h, 1, D]
                    new_lines_in_local = repeat(self.image_newline, 'd -> (th h) 1 d', th=num_height_tiles, h=h)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    local_features = torch.cat([local_features, new_lines_in_local], dim=1)

                    # [num_height_tiles * h, num_width_tiles * w + 1, D]
                    #   --> [(num_height_tiles * h) * (num_width_tiles * w + 1), D]
                    local_features = local_features.view(-1, n_dim)

                    # ----------------- merge global and local tiles -----------------
                    if self.global_view_pos == 'head':
                        global_local_features = torch.cat(
                            [global_features, self.view_seperator[None, :], local_features], dim=0)
                    else:
                        global_local_features = torch.cat(
                            [local_features, self.view_seperator[None, :], global_features], dim=0)

                else:
                    # abandoned，will not step into this logic
                    global_features = torch.cat([self.tile_indicators[0:1], global_features], dim=0)
                    local_features = torch.cat(
                        [self.tile_indicators[1:num_tiles_in_image + 1].unsqueeze(1), local_features], dim=1)
                    local_features = rearrange(local_features, 'crop_num hw d -> (crop_num hw) d')

                    if self.global_view_pos == 'head':
                        global_local_features = torch.cat([global_features, local_features], dim=0)
                    else:
                        global_local_features = torch.cat([local_features, global_features], dim=0)

                images_in_this_batch.append(global_local_features)

            if len(images_in_this_batch) > 0:
                images_in_this_batch = torch.cat(images_in_this_batch, dim=0).to(input_embeds.dtype)
                crt_image_mask = images_seq_mask[idx].unsqueeze(-1).to(input_embeds.device)
                input_embeds[idx].masked_scatter_(crt_image_mask, images_in_this_batch)

        return input_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        images_spatial_crop: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        # process image embeddings
        if inputs_embeds is None and pixel_values is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids=input_ids,
                                                       images=pixel_values,
                                                       images_seq_mask=image_mask,
                                                       images_spatial_crop=images_spatial_crop)

        outputs = self.language.forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
        )
        return outputs

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """Prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # vision inputs
        pixel_values = None
        images_spatial_crop = None
        image_mask = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            images_spatial_crop = [p_value[0].meta.get('images_spatial_crop', None) for p_value in pixel_values]
            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                image_token_id = pixel_values[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                pixel_values = torch.cat([data.data for data in pixel_values]).unsqueeze(0)
            else:
                pixel_values = None
                image_mask = None

            if len(images_spatial_crop) > 0:
                images_spatial_crop = torch.cat([crop for crop in images_spatial_crop]).unsqueeze(0)
            else:
                images_spatial_crop = None

        return dict(
            input_ids=input_ids,  # [b, T]
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,  # [b, max_n_images, 3, height, width]
            images_spatial_crop=images_spatial_crop,  # [b, max_n_images, 2]
            image_mask=image_mask,  # [b, T]
            inputs_embeds=inputs_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        lang_prefix = 'language.'
        lang_prefix_length = len(lang_prefix)
        new_weights = dict()
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            if name.startswith(lang_prefix):
                new_key = name[lang_prefix_length:]
                new_weights[new_key] = loaded_weight
                continue

            if 'qkv' in name and 'vision' not in name:
                param = params_dict[name]
                q, k, v = param.weight_spliter(loaded_weight)
                load_weight(param, q, shard_id='q')
                load_weight(param, k, shard_id='k')
                load_weight(param, v, shard_id='v')
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

        self.language.load_weights(new_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class DeepSeekVLV2InputProcessor(BaseModelInputProcessor):
    """Deepseek-vl2 input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype
        vision_config = config.vision_config
        self.patch_size = vision_config.patch_size

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """Prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            image_token_id = input_mm['image_token_id']
            num_pad = input_mm['image_tokens']
            images_spatial_crop = input_mm.get('images_spatial_crop', None)
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(
                                           image_token_id=image_token_id,
                                           images_spatial_crop=images_spatial_crop,
                                       ))

            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )

        return result
