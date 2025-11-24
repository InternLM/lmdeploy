# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding as Qwen3VLVisionRotaryEmbedding
from .qwen2_5_vl import Qwen2_5_VLInputProcessor as Qwen3VLInputProcessor
from .qwen2_5_vl import Qwen2_5_VLVisionAttention as Qwen3VLVisionAttention
from .qwen3 import Qwen3model
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


class Qwen3VLTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: PretrainedConfig, device=None):
        super().__init__()
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get('rope_type', 'default')
        else:
            self.rope_type = 'default'
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get('mrope_section', [24, 20, 20])

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THTHWHTHW...TT], preserving frequency continuity.
        args:
            x: (3, bs, seq_len, head_dim // 2)
            mrope_section: (3,)
        returns:
            x_t: (bs, seq_len, head_dim // 2)
        """
        freqs_t = freqs[0]  # just overwrite the first dimension T
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        # In contrast to other models, Qwen3VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Qwen3VLTextModel(Qwen3model):
    """Text part of Qwen3VL.

    not a pure text-only model, as DeepStack integrates visual features into the early hidden states.
    """

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__(config=config, dtype=dtype, device=device)

        # build rotary embedding
        # TODO: zhouxinyu, add triton kernel for interleaved mrope
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_position_ids: torch.LongTensor = None,
        # args for deepstack
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
    ):
        """visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`,
        *optional*):

        The mask of the visual positions. deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):     The deepstack
        visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).     The feature is extracted from the
        different visual encoder layers, and fed to the decoder     hidden states. It's from the paper DeepStack (
        https://arxiv.org/abs/2406.04)
        """

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        if mrope_position_ids is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
        else:
            mrope_position_ids = mrope_position_ids.unsqueeze(1)
            cos, sin = self.rotary_emb(hidden_states, mrope_position_ids)

        cos, sin = cos[0], sin[0]
        rotary_pos_emb = (cos, sin)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx]
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                residual=residual,
                attn_metadata=attn_metadata,
            )

            # add visual features to the hidden states of first several layers
            if deepstack_visual_embeds is not None and idx in range(len(deepstack_visual_embeds)):
                hidden_states = hidden_states + residual
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[idx],
                )
                residual = None

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def _deepstack_process(self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                           visual_embeds: torch.Tensor):
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        local = torch.zeros_like(hidden_states)
        local.masked_scatter_(visual_pos_masks, visual_embeds)
        hidden_states += local
        return hidden_states


class Qwen3VLVisionPatchEmbed(nn.Module):

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels,
                              self.embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=True,
                              dtype=dtype,
                              device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size,
                                           self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionMLP(nn.Module):
    """Vision mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        from transformers.activations import ACT2FN
        hidden_dim = config.hidden_size
        intermediate_size = config.intermediate_size
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.linear_fc1 = build_colwise_linear(
            hidden_dim,
            intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # gelu_pytorch_tanh
        self.act = ACT2FN[config.hidden_act]

        # down
        self.linear_fc2 = build_rowwise_linear(intermediate_size,
                                               hidden_dim,
                                               bias=True,
                                               dtype=dtype,
                                               device=device,
                                               quant_config=quantization_config,
                                               is_tp=True)

    def forward(self, x):
        """forward."""
        return self.linear_fc2(self.act(self.linear_fc1(x)))


class Qwen3VLVisionBlock(nn.Module):
    """Vision block."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device)

        self.attn = Qwen3VLVisionAttention(config, dtype=dtype, device=device)

        self.mlp = Qwen3VLVisionMLP(config, dtype=dtype, device=device)

    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionPatchMerger(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 use_postshuffle_norm=False,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size,
                              eps=1e-6,
                              dtype=dtype,
                              device=device)
        self.linear_fc1 = build_colwise_linear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            is_tp=True,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = build_rowwise_linear(
            self.hidden_size,
            config.out_hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
            is_tp=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


@vlm_model
class Qwen3VLVisionModel(nn.Module):
    """Vision transformer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config=config, dtype=dtype, device=device)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size, dtype=dtype, device=device)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2, device=device)

        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config, layer_idx, dtype=dtype, device=device) for layer_idx in range(config.depth)])
        self.merger = Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False, dtype=dtype, device=device)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True, dtype=dtype, device=device)
            for _ in range(len(config.deepstack_visual_indexes))
        ])

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=self.pos_embed.weight.device)
        weight_tensor = torch.tensor(weight_list,
                                     dtype=self.pos_embed.weight.dtype,
                                     device=self.pos_embed.weight.device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size,
                                        -1).permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor, rotary_pos_emb: torch.Tensor,
                pos_embeds: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.patch_embed(hidden_states)
        hidden_states = hidden_states + pos_embeds
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merge_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merge_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


class Qwen3VLForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):
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
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        # build preprocessor
        self.input_processor = Qwen3VLInputProcessor(self.config)

        # build vision model
        self.visual = Qwen3VLVisionModel(
            config.vision_config,
            dtype=dtype,
            device=device,
        )

        # build text model
        self.language_model = Qwen3VLTextModel(config.text_config, dtype=dtype, device=device)

        # build lm_head
        self.lm_head = build_rowwise_linear(config.text_config.hidden_size,
                                            config.text_config.vocab_size,
                                            bias=False,
                                            dtype=dtype,
                                            device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        mrope_position_ids: torch.Tensor = None,
        pixel_values: torch.Tensor = None,
        vis_cu_seqlens: torch.Tensor = None,
        vis_pos_emb: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        pos_embeds: torch.Tensor = None,
        grid_thw: torch.Tensor = None,
        **kwargs,
    ):
        """Model forward, return logits."""

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                dtype = inputs_embeds.dtype
                pixel_values = pixel_values.to(dtype)
                vis_pos_emb = (vis_pos_emb[0].to(dtype), vis_pos_emb[1].to(dtype))

                # get image embeds and deepstack visual embeds
                image_embeds, deepstack_visual_embeds = self.visual(pixel_values,
                                                                    cu_seqlens=vis_cu_seqlens,
                                                                    rotary_pos_emb=vis_pos_emb,
                                                                    pos_embeds=pos_embeds)

                # split image embeds per sample
                split_sizes = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
                image_embeds = torch.split(image_embeds, split_sizes)
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, dtype)

                # mask and scatter to create final input embeddings
                expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(expanded_image_mask, image_embeds)

                visual_pos_masks = expanded_image_mask

        hidden_states = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
            # args for deepstack
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def update_weights(self):
        """Update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.language_model.embed_tokens.weight

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
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
        image_mask = None
        grid_thw = None
        pos_embeds = None
        if context.input_multimodals is not None:
            image_data = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            if len(image_data) > 0:
                # flatten batch
                image_data = [data for im_data in image_data for data in im_data]
                pixel_values = torch.cat([data.data for data in image_data])
                image_token_id = image_data[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                grid_thw = torch.cat([data.meta['grid_thw'] for data in image_data]).cpu()
                vis_pos_emb = self.visual.rot_pos_emb(grid_thw)
                pos_embeds = self.visual.fast_pos_embed_interpolate(grid_thw)
                vis_cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                                         grid_thw[:, 0]).to(pixel_values.device)
                vis_cu_seqlens = vis_cu_seqlens.cumsum(dim=0, dtype=torch.int32)
                vis_pos_emb = vis_pos_emb.repeat(1, 2)
                vis_pos_emb = (vis_pos_emb.cos(), vis_pos_emb.sin())

        mrope_position_ids = getattr(context, 'mrope_position_ids', None)

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
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            image_mask=image_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
        )

    def rename_weight(self, name: str) -> str:
        """Rename weight."""
        if name.startswith('model.language_model.'):
            return 'language_model.' + name[len('model.language_model.'):]
        elif name.startswith('model.visual.'):
            return 'visual.' + name[len('model.visual.'):]
        elif name.startswith('model.'):
            return name[len('model.'):]
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
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

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                continue
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
                    param = params_dict[name]
                    load_weight(param, loaded_weight)

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'] = mrope_position_ids.new_zeros(3, max_tokens)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)

        input_ids = kwargs.get('input_ids')
        num_tokens = input_ids.size(-1)
        new_batch_size = graph_meta.max_batchs

        is_decoding = graph_meta.is_decoding
        input_buffers = graph_meta.input_buffers
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'][:, :num_tokens] = mrope_position_ids
            if is_decoding:
                new_inputs['mrope_position_ids'] = input_buffers['mrope_position_ids'][:, :new_batch_size]
            else:
                new_inputs['mrope_position_ids'] = input_buffers['mrope_position_ids']

        return new_inputs

    def _get_model_metas(self, context: StepContext):
        """Get model metas."""
        model_metas = context.model_metas
        if model_metas is None:
            batch_size = context.q_seqlens.numel()
            return [dict(mrope_delta=0)] * batch_size
        return [dict(mrope_delta=0) if meta is None else meta for meta in model_metas]

    def _update_model_meta_decoding(self, context: StepContext):
        """Update model meta for decoding."""
        model_metas = self._get_model_metas(context)
        position_ids = context.position_ids

        mrope_deltas = [meta['mrope_delta'] for meta in model_metas]
        mrope_deltas = position_ids.new_tensor(mrope_deltas)
        mrope_position_ids = position_ids + mrope_deltas[None]
        mrope_position_ids = mrope_position_ids.expand(3, -1)

        context.mrope_position_ids = mrope_position_ids
        return model_metas

    def _get_multimodal_pos_ids(self, grid_thw: list, device: torch.device):
        """Get mrope ids."""
        t, h, w = grid_thw
        h //= 2
        w //= 2
        stride = torch.tensor([h * w, w, 1], device=device)[:, None]
        size = torch.tensor([t, h, w], device=device)[:, None]
        pos_ids = torch.arange(t * h * w, device=device)[None].expand(3, -1)
        pos_ids = pos_ids // stride % size
        return pos_ids

    def _update_model_meta_prefilling(self, context: StepContext):
        """Update model meta for prefilling."""
        model_metas = self._get_model_metas(context)
        input_multimodals = context.input_multimodals
        if input_multimodals is None:
            input_multimodals = [None] * len(model_metas)
        position_ids = context.position_ids
        batched_pos_ids = position_ids[0].split(context.q_seqlens.tolist())
        mrope_position_ids = []
        new_model_metas = []
        for pos_ids, model_meta, input_mm in zip(batched_pos_ids, model_metas, input_multimodals):
            images = []
            if input_mm is not None:
                images = input_mm.get('image', [])
            if model_meta is None or 'mrope_delta' not in model_meta:
                mrope_delta = 0
            else:
                mrope_delta = model_meta['mrope_delta']

            pos_start = pos_ids[0].item()
            mrope_pos_ids = pos_ids + mrope_delta
            mrope_pos_ids = mrope_pos_ids[None].expand(3, -1).clone()
            for img in images:
                grid_thw = img.meta['grid_thw'][0].tolist()
                _, h, w = grid_thw
                h //= 2
                w //= 2
                num_pad = img.end - img.start - max(h, w)
                mrope_delta -= num_pad
                fill_start = img.start - pos_start
                fill_end = img.end - pos_start
                img_pos_ids = self._get_multimodal_pos_ids(grid_thw, pos_ids.device)
                img_pos_ids += mrope_pos_ids[:, fill_start:fill_start + 1]
                mrope_pos_ids[:, fill_end:] -= num_pad
                mrope_pos_ids[:, fill_start:fill_end] = img_pos_ids

            mrope_position_ids.append(mrope_pos_ids)
            new_model_metas.append(dict(mrope_delta=mrope_delta))

        mrope_position_ids = torch.cat(mrope_position_ids, dim=1)
        context.mrope_position_ids = mrope_position_ids

        return new_model_metas

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: Optional[torch.Tensor] = None,
                           context: StepContext = None):
        """Update model meta."""
        if context.is_decoding:
            return self._update_model_meta_decoding(context)
        else:
            return self._update_model_meta_prefilling(context)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


InputMultiModalType = List[Dict[str, Any]]
