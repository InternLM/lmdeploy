# Copyright (c) OpenMMLab. All rights reserved.
# adapted from https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek_vl_v2.py

import gc
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .deepseek_v2 import DeepseekV2ForCausalLM
from .deepseek_vl2_config import DeepseekV2Config
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin

DEBUG_WITH_VISION = True


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
            # import ipdb; ipdb.set_trace()
            patches = x.unfold(2, 2, 2).unfold(3, 2, 2)
            batch_size, channels, h_patches, w_patches, _, _ = patches.size()
            # 在通道维度上拼接
            patches = patches.contiguous().view(batch_size, channels, h_patches * w_patches, -1)

            # 通过线性层
            patches = patches.permute(0, 2, 1, 3).contiguous()
            patches = patches.view(batch_size, h_patches * w_patches, channels * 4)

            x = self.token_pooling_layer(patches)

        elif self.cfg.projector_type == 'downsample_mlp_gelu':
            bs, hw, input_dim = x.shape
            h = w = int((hw)**0.5)
            """compute padding"""
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


class VisionEncoderConfig(PretrainedConfig):
    model_type: str = 'vision'
    model_name: str = 'siglip_large_patch16_384'
    image_size: int = 384
    patch_size: int = 16
    width: int = 1024
    layers: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    global_pool: str = 'map'
    ignore_head: bool = True
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False
    weight_init: str = 'skip'
    deterministic: bool = False
    num_recomputing_layers: int = 0

    def __init__(self,
                 model_name: str = 'siglip_large_patch16_384',
                 image_size: int = 384,
                 patch_size: int = 16,
                 width: int = 1024,
                 layers: int = 24,
                 heads: int = 16,
                 mlp_ratio: int = 4,
                 global_pool: str = 'map',
                 ignore_head: bool = True,
                 class_token: bool = False,
                 num_classes: int = 0,
                 use_checkpoint: bool = False,
                 **kwargs):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        self.global_pool = global_pool
        self.ignore_head = ignore_head
        self.class_token = class_token
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        super().__init__(**kwargs)


class MlpProjectorConfig(PretrainedConfig):
    model_type = 'mlp_projector'
    projector_type: str = 'downsample_mlp_gelu'
    input_dim: int = 1152
    n_embed: int = 2048
    depth: int = 2
    mlp_ratio: int = 1
    downsample_ratio: int = 2
    token_pooling: bool = False

    def __init__(self,
                 projector_type: str = 'downsample_mlp_gelu',
                 input_dim: int = 1152,
                 n_embed: int = 2048,
                 depth: int = 2,
                 mlp_ratio: int = 1,
                 downsample_ratio: int = 2,
                 **kwargs):
        self.projector_type = projector_type
        self.input_dim = input_dim
        self.n_embed = n_embed
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.downsample_ratio = downsample_ratio

        super().__init__(**kwargs)


@dataclass
class DeepSeekVLV2CausalLMOutputWithPast(ModelOutput):
    """Base class for DeepSeek-VL2 causal language model (or autoregressive)
    outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when
            `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when
            `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when
            `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        rope_deltas (`torch.LongTensor` of shape `(batch_size, )`, *optional*):
            The rope index difference between sequence length and multimodal rope.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class DeepseekVLV2Config(PretrainedConfig):
    model_type = 'deepseek_vl_v2'
    vision_config: VisionEncoderConfig
    projector_config: MlpProjectorConfig
    language_config: DeepseekV2Config

    tile_tag: str = '2D'
    global_view_pos: str = 'head'
    candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384), )

    def __init__(self,
                 tile_tag: str = 'tile_tag',
                 global_view_pos: str = 'head',
                 candidate_resolutions: Tuple[Tuple[int, int]] = ((384, 384), ),
                 **kwargs):
        super().__init__(**kwargs)

        vision_config = kwargs.get('vision_config', {})
        self.vision_config = VisionEncoderConfig(**vision_config)

        projector_config = kwargs.get('projector_config', {})
        self.projector_config = MlpProjectorConfig(**projector_config)

        language_config = kwargs.get('language_config', {})
        if isinstance(language_config, DeepseekV2Config):
            self.language_config = language_config
        else:
            self.language_config = DeepseekV2Config(**language_config)

        self.tile_tag = tile_tag
        self.global_view_pos = global_view_pos
        self.candidate_resolutions = candidate_resolutions


class DeepseekVLV2PreTrainedModel(PreTrainedModel):
    config_class = DeepseekVLV2Config
    base_model_prefix = 'deepseek_vl_v2'
    _no_split_modules = []
    _skip_keys_device_placement = 'past_key_values'


class DeepseekVLV2ForCausalLM(DeepseekVLV2PreTrainedModel, nn.Module, CudaGraphMixin, DeployModelMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(config)
        self.ctx_mgr = ctx_mgr
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'

        # ----------- vision encoder ------------
        self.vision = self._init_vision_module(dtype=dtype)

        # ----------- vl projector ------------
        projector_config = config.projector_config
        self.projector = MlpProjector(projector_config, dtype)

        # image token format 形式
        # FIXME 目前tile tag & global_view_pos的默认取值都是之前的实验策略；后续应当去掉默认取值，改为没有取值就raise error
        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # 用于format image token sequence的特殊token
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
        # TODO: refactor vision model through timm wrapper from transformers
        try:
            import timm
        except ImportError:
            raise ImportError('Please install timm') from ImportError

        model = timm.create_model(
            'vit_so400m_patch14_siglip_384.webli',
            # "vit_so400m_patch14_siglip_384",
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

        # import pdb; pdb.set_trace()
        if images is None or images_spatial_crop.sum() == 0:
            return self.language.get_input_embeddings()(input_ids)

        # 1, 1, 2
        bs, max_n_images, _ = images_spatial_crop.shape
        # [tensor(2, device='cuda:0')]
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
        # [2, 384, 384]
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

        # 根据self.tile_tag & self.global_view_pos填充image token sequence
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
                    # abandoned，实际上不会走这个逻辑
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

    @torch.no_grad()
    def incremental_prefilling(self,
                               input_ids: Optional[torch.LongTensor] = None,
                               attention_mask: Optional[torch.Tensor] = None,
                               inputs_embeds: Optional[torch.FloatTensor] = None,
                               images: Optional[torch.FloatTensor] = None,
                               images_seq_mask: Optional[torch.LongTensor] = None,
                               images_spatial_crop: Optional[torch.LongTensor] = None,
                               chunk_size: int = 1024):
        if inputs_embeds is None:
            inputs_embeds = self.prepare_inputs_embeds(
                input_ids=input_ids,
                images=images,
                images_seq_mask=images_seq_mask,
                images_spatial_crop=images_spatial_crop,
            )

            del images
            del images_seq_mask
            del images_spatial_crop

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

            self._clear_cuda_cache()

        bzs, seq_len, _ = inputs_embeds.shape
        past_key_values = None

        # remain the last token for the next forward
        prefilling_len = seq_len - 1
        for i in range(0, prefilling_len, chunk_size):
            chunk_start = i
            chunk_end = min(i + chunk_size, prefilling_len)
            chunk_inputs_embeds = inputs_embeds[:, chunk_start:chunk_end]
            chunk_attention_mask = attention_mask[:, 0:chunk_end]
            # print(f"start = {chunk_start}, end = {chunk_end}, prefilling_len = {prefilling_len}, seq_len = {seq_len}")

            # compute position_ids
            if past_key_values is not None:
                position_ids = torch.arange(chunk_start, chunk_end, dtype=torch.long,
                                            device=inputs_embeds.device).unsqueeze(0)
                past_key_values = self._move_past_key_values_to_gpu(past_key_values, inputs_embeds.device)
            else:
                position_ids = None

            # chunk-forward
            with torch.no_grad():
                outputs = self.forward(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=chunk_attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
                # update past_key_values
                past_key_values = outputs.past_key_values
                past_key_values = self._move_past_key_values_to_cpu(past_key_values)

                del outputs, position_ids
                self._clear_cuda_cache()

        prefilling_key_values = []
        for layer_past in past_key_values:
            prefilling_key_values.append((
                layer_past[0][:, :, 0:prefilling_len, ...].to(inputs_embeds.device),
                layer_past[1][:, :, 0:prefilling_len, ...].to(inputs_embeds.device),
            ))

        return inputs_embeds, prefilling_key_values

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
                                                       images=pixel_values.unsqueeze(0),
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

    def _clear_cuda_cache(self):
        """clear CUDA memory cache."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _move_past_key_values_to_cpu(self, past_key_values):
        # print(f"past_key_values -> cpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.cpu() for t in layer) for layer in past_key_values)

    def _move_past_key_values_to_gpu(self, past_key_values, device='cuda:0'):
        # print(f"past_key_values -> gpu")
        if past_key_values is None:
            return None
        return tuple(tuple(t.to(device) for t in layer) for layer in past_key_values)

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.language.get_logits(hidden_states)

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.language.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: torch.Tensor = None,
        context: StepContext = None,
    ):
        """prepare input."""
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        # vision inputs
        pixel_values = None
        # image_mask = None
        images_spatial_crop = None
        images_seq_mask = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            images_spatial_crop = [p_value[0].meta.get('images_spatial_crop', None) for p_value in pixel_values]
            images_seq_mask = [p_value[0].meta.get('images_seq_mask', None) for p_value in pixel_values]

            if images_spatial_crop is not None:
                images_spatial_crop = torch.tensor(images_spatial_crop)
            if images_seq_mask is not None:
                images_seq_mask = torch.tensor(images_seq_mask)

            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                # image_token_id = pixel_values[0].meta['image_token_id']
                # image_mask = input_ids == image_token_id
                pixel_values = torch.cat([data.data for data in pixel_values])
            else:
                pixel_values = None
                # image_mask = None

        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            pixel_values=pixel_values,
            images_spatial_crop=images_spatial_crop,
            image_mask=images_seq_mask,
            inputs_embeds=inputs_embeds,
        )

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past), )
        return reordered_past

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""

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
                print(f'qkv para => {name}')
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
        """get input processor."""
        return self.input_processor


class DeepSeekVLV2InputProcessor(BaseModelInputProcessor):
    """deepseek-vl2 input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype
        vision_config = config.vision_config
        self.patch_size = vision_config.patch_size

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
                         **kwargs) -> PreprocessInputResult:
        """prepare multimodal input."""
        if input_multimodals is None or len(input_multimodals) == 0:
            return input_ids, input_multimodals

        input_imgs = []
        for input_mm in input_multimodals:
            pixel_values = input_mm['pixel_values'].to(self.dtype)
            offset = input_mm['offset']
            image_token_id = input_mm.get('image_token_id', 0)
            num_pad = input_mm['image_tokens']
            images_spatial_crop = input_mm.get('images_spatial_crop', None)
            images_seq_mask = input_mm.get('images_seq_mask', None)
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id,
                                                 images_spatial_crop=images_spatial_crop,
                                                 images_seq_mask=images_seq_mask))

            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )

        return result


AutoConfig.register('vision', VisionEncoderConfig)
AutoConfig.register('mlp_projector', MlpProjectorConfig)
AutoConfig.register('deepseek_vl_v2', DeepseekVLV2Config)
AutoModelForCausalLM.register(DeepseekVLV2Config, DeepseekVLV2ForCausalLM)
