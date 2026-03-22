# Copyright (c) OpenMMLab. All rights reserved.

import math
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.multimodal.data_type import MultiModalData
from lmdeploy.pytorch.nn import ApplyRotaryEmb, FlashAttention, LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight
from lmdeploy.vl.constants import Modality

from .qwen3_vl import Qwen3VLVisionBlock, Qwen3VLVisionPatchEmbed, Qwen3VLVisionRotaryEmbedding
from .qwen3_vl_moe import Qwen3VLMoeTextModel
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


def _get_feat_extract_output_lengths(input_lengths):
    """Computes the output length of the convolutional layers and the output
    length of the audio encoder."""

    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths


class Qwen3OmniMoeAudioAttention(nn.Module):
    """Vision attention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        dim = config.d_model
        num_heads = config.encoder_attention_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            dim,
            num_q_heads=num_heads,
            num_kv_heads=num_heads,
            head_size=head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attention = FlashAttention(
            num_heads,
            head_dim,
            causal=False,
        )

        # o_proj
        self.out_proj = build_rowwise_linear(dim,
                                             dim,
                                             bias=True,
                                             quant_config=quantization_config,
                                             dtype=dtype,
                                             device=device,
                                             is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        q, k, v = self.qkv_proj.split_qkv(qkv_states)

        attn_output = self.attention(
            q,
            k,
            v,
            q_start_loc=cu_seqlens[:-1],
            q_seqlens=cu_seqlens[1:] - cu_seqlens[:-1],
        )

        attn_output = attn_output.reshape(seq_length, -1)

        # o proj
        attn_output = self.out_proj(attn_output)
        return attn_output


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    """Qwen3OmniMoeAudioEncoderLayer."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3OmniMoeAudioAttention(config, dtype=dtype, device=device)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, eps=1e-5, dtype=dtype, device=device)

        self.activation_fn = ACT2FN[config.activation_function]
        self.fc1 = build_colwise_linear(
            self.embed_dim,
            config.encoder_ffn_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.fc2 = build_rowwise_linear(
            config.encoder_ffn_dim,
            self.embed_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.final_layer_norm = LayerNorm(self.embed_dim, eps=1e-5, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states, )

        return outputs


class SinusoidsPositionEmbedding(nn.Module):

    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError('SinusoidsPositionEmbedding needs even channels input')
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            'positional_embedding',
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen3OmniMoeAudioEncoder(nn.Module):
    """Qwen3OmniMoeAudioEncoder."""

    def __init__(self, config, dtype: torch.dtype = None, device: torch.device = None) -> None:
        super().__init__()

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList(
            [Qwen3OmniMoeAudioEncoderLayer(config, dtype=dtype, device=device) for _ in range(config.encoder_layers)])
        self.ln_post = LayerNorm(config.d_model, eps=1e-5, dtype=dtype, device=device)
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1, dtype=dtype, device=device)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size,
                                 config.downsample_hidden_size,
                                 3,
                                 2,
                                 padding=1,
                                 dtype=dtype,
                                 device=device)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size,
                                 config.downsample_hidden_size,
                                 3,
                                 2,
                                 padding=1,
                                 dtype=dtype,
                                 device=device)
        conv_out_dim = config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)
        self.conv_out = nn.Linear(
            conv_out_dim,
            config.d_model,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model, dtype=dtype, device=device)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim, dtype=dtype, device=device)
        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

    def forward(
        self,
        input_features: torch.Tensor,
        feature_lens: torch.Tensor,
        aftercnn_lens=None,
    ):
        r"""feature_lens (`torch.LongTensor` of shape `(batch_size,)`):

        mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[:padded_embed.shape[1], :].unsqueeze(0).to(
                padded_embed.dtype))
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return hidden_states


class Qwen3OmniMoeVisionPatchMerger(nn.Module):
    """Vision patch merger.

    Different namings with qwen3vl, but actual calculations are the same.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 use_postshuffle_norm=False,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.ln_q = LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size,
                              eps=1e-6,
                              dtype=dtype,
                              device=device)
        self.mlp = nn.ModuleList([
            build_colwise_linear(
                self.hidden_size,
                self.hidden_size,
                bias=True,
                dtype=dtype,
                device=device,
                is_tp=True,
            ),
            nn.GELU(),
            build_rowwise_linear(
                self.hidden_size,
                config.out_hidden_size,
                bias=True,
                dtype=dtype,
                device=device,
                is_tp=True,
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        for layer in self.mlp:
            x = layer(x)
        return x


@vlm_model
class Qwen3OmniMoeVisionEncoder(nn.Module):
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
        self.merger = Qwen3OmniMoeVisionPatchMerger(config=config,
                                                    use_postshuffle_norm=False,
                                                    dtype=dtype,
                                                    device=device)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.merger_list = nn.ModuleList([
            Qwen3OmniMoeVisionPatchMerger(config=config, use_postshuffle_norm=True, dtype=dtype, device=device)
            for _ in range(len(config.deepstack_visual_indexes))
        ])

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size

        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Rotary position embedding."""
        pos_ids = []

        for t, h, w in grid_thw:
            base = self.rot_pos_ids(int(h), int(w), self.spatial_merge_size)
            pos_ids.append(base if t == 1 else base.repeat(t, 1))

        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)

        return rotary_pos_emb

    # copy from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py#L474
    def fast_pos_embed_interpolate(self, grid_thw: List[List[int]]) -> torch.Tensor:
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim
        device = self.pos_embed.weight.device

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(0, num_grid_per_side - 1, h, dtype=torch.float32, device=device)
            w_idxs = torch.linspace(0, num_grid_per_side - 1, w, dtype=torch.float32, device=device)

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing='ij')
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing='ij')
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing='ij')

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.pos_embed.weight.dtype, device=device)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(h // m_size, m_size, w // m_size, m_size, hidden_dim)
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

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
                deepstack_feature = self.merger_list[deepstack_merge_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


class Qwen3OmniMoeThinkerForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):
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
        thinker_config = config.thinker_config

        # build preprocessor
        self.input_processor = Qwen3OmniInputProcessor(self.config)

        # build audio encoder
        self.audio_tower = Qwen3OmniMoeAudioEncoder(
            thinker_config.audio_config,
            dtype=dtype,
            device=device,
        )

        # build vision encoder
        self.visual = Qwen3OmniMoeVisionEncoder(
            thinker_config.vision_config,
            dtype=dtype,
            device=device,
        )

        # build text model
        self.language_model = Qwen3VLMoeTextModel(thinker_config.text_config, dtype=dtype, device=device)

        # build lm_head
        self.lm_head = build_rowwise_linear(thinker_config.text_config.hidden_size,
                                            thinker_config.text_config.vocab_size,
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
        audio_values: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        audio_feature_lengths: torch.Tensor = None,
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

            if audio_values is not None:
                dtype = inputs_embeds.dtype
                audio_values = audio_values.to(dtype)
                audio_embeds = self.audio_tower(
                    input_features=audio_values,
                    feature_lens=audio_feature_lengths,
                )
                inputs_embeds = inputs_embeds.masked_scatter(audio_mask.unsqueeze(-1), audio_embeds)

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

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
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
        image_mask = None
        grid_thw = None
        pos_embeds = None
        audio_values = None
        audio_mask = None
        audio_feature_lengths = None
        if context.input_multimodals is not None:
            mm_inputs = [input_mm.get('mm_data', []) for input_mm in context.input_multimodals]
            # flatten batch
            mm_inputs = [item for sublist in mm_inputs for item in sublist]

            if len(mm_inputs) > 0:
                modality = mm_inputs[0].modality

                image_token_id = mm_inputs[0].meta.get('image_token_id')
                video_token_id = mm_inputs[0].meta.get('video_token_id')
                audio_token_id = mm_inputs[0].meta.get('audio_token_id')

                if modality == Modality.AUDIO:
                    audio_values = torch.cat([inp.data for inp in mm_inputs])
                    # FIXME: zhouxinyu, batch ?
                    audio_values = audio_values.squeeze(0)
                    audio_mask = (input_ids == audio_token_id)
                    # FIXME: zhouxinyu, list ?
                    audio_feature_lengths = mm_inputs[0].meta['audio_feature_lengths']
                elif modality in [Modality.IMAGE, Modality.VIDEO]:
                    pixel_values = torch.cat([inp.data for inp in mm_inputs])

                    mm_token_id = image_token_id if modality == Modality.IMAGE else video_token_id
                    image_mask = (input_ids == mm_token_id)

                    grid_thw = torch.cat([data.meta['grid_thw'] for data in mm_inputs]).cpu()
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
            audio_values=audio_values,
            audio_mask=audio_mask,
            audio_feature_lengths=audio_feature_lengths,
        )

    def rename_weight(self, name: str) -> str:
        """Rename weight."""
        if name.startswith('thinker.model.'):
            return 'language_model.' + name[len('thinker.model.'):]
        elif name.startswith('thinker.visual.'):
            return 'visual.' + name[len('thinker.visual.'):]
        elif name.startswith('thinker.audio_tower.'):
            return 'audio_tower.' + name[len('thinker.audio_tower.'):]
        # thinker_config.text_config tie_word_embeddings = False
        elif name.startswith('thinker.lm_head.'):
            return 'lm_head.' + name[len('thinker.lm_head.'):]
        return name

    def _load_weight_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                             expert_params_mapping: List):
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
    def _load_weight_fused_experts(self, name: str, loaded_weight: torch.Tensor, params_dict: Dict[str, nn.Parameter],
                                   fused_expert_params_mapping: List):
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

        # expert mapping
        num_experts = self.config.thinker_config.text_config.num_experts
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
        for name, loaded_weight in weights:
            if 'rotary_emb.inv_freq' in name:
                continue
            if ('rotary_emb.cos_cached' in name or 'rotary_emb.sin_cached' in name):
                continue
            # skip talker and code2wav weights
            if ('talker.' in name or 'code2wav.' in name):
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
            mm_data_list = []
            if input_mm is not None:
                mm_data_list.extend(input_mm.get('mm_data', []))

            if model_meta is None or 'mrope_delta' not in model_meta:
                mrope_delta = 0
            else:
                mrope_delta = model_meta['mrope_delta']

            pos_start = pos_ids[0].item()
            mrope_pos_ids = pos_ids + mrope_delta
            mrope_pos_ids = mrope_pos_ids[None].expand(3, -1).clone()

            for mm_data in mm_data_list:
                if mm_data.modality == Modality.IMAGE:
                    grid_thw = mm_data.meta['grid_thw'][0].tolist()
                    _, h, w = grid_thw
                    h //= 2
                    w //= 2
                    num_pad = mm_data.end - mm_data.start - max(h, w)
                    mrope_delta -= num_pad
                    fill_start = mm_data.start - pos_start
                    fill_end = mm_data.end - pos_start
                    img_pos_ids = self._get_multimodal_pos_ids(grid_thw, pos_ids.device)
                    img_pos_ids += mrope_pos_ids[:, fill_start:fill_start + 1]
                    mrope_pos_ids[:, fill_end:] -= num_pad
                    mrope_pos_ids[:, fill_start:fill_end] = img_pos_ids
                elif mm_data.modality == Modality.VIDEO:
                    second_per_grid = mm_data.meta.get('second_per_grid', 2.0)
                    position_id_per_seconds = self.config.thinker_config.position_id_per_seconds

                    grid_thw = mm_data.meta['grid_thw'][0].tolist()
                    t, h, w = grid_thw
                    llm_h = h // 2  # spatial_merge_size = 2
                    llm_w = w // 2

                    device = pos_ids.device
                    # Temporal indices as real timestamps (float, e.g. 0, 1.083, 2.167 for fps=24)
                    t_index = torch.arange(t, device=device).float() * (second_per_grid * position_id_per_seconds)
                    h_index = torch.arange(llm_h, device=device).float()
                    w_index = torch.arange(llm_w, device=device).float()

                    # Build [3, T*llm_h*llm_w] pos ids
                    t_expanded = t_index.view(-1, 1).expand(-1, llm_h * llm_w).flatten()
                    h_expanded = h_index.view(1, -1, 1).expand(t, -1, llm_w).flatten()
                    w_expanded = w_index.view(1, 1, -1).expand(t, llm_h, -1).flatten()
                    video_pos_ids = torch.stack([t_expanded, h_expanded, w_expanded])  # [3, T*llm_h*llm_w]

                    max_video_pos = max(
                        float((t - 1) * second_per_grid * position_id_per_seconds) if t > 1 else 0.0,
                        float(llm_h - 1),
                        float(llm_w - 1),
                    )
                    video_num_tokens = t * llm_h * llm_w
                    num_pad = video_num_tokens - max_video_pos - 1
                    mrope_delta -= num_pad

                    fill_start = mm_data.start - pos_start
                    fill_end = mm_data.end - pos_start

                    # Convert to float to hold non-integer temporal positions
                    mrope_pos_ids = mrope_pos_ids.float()
                    offset = mrope_pos_ids[0, fill_start].item()
                    mrope_pos_ids[:, fill_start:fill_end] = video_pos_ids + offset
                    mrope_pos_ids[:, fill_end:] -= num_pad

            mrope_position_ids.append(mrope_pos_ids)
            new_model_metas.append(dict(mrope_delta=mrope_delta))

        mrope_position_ids = torch.cat(mrope_position_ids, dim=1)
        context.mrope_position_ids = mrope_position_ids

        return new_model_metas

    def update_model_metas(self,
                           past_key_values: List[List[torch.Tensor]],
                           inputs_embeds: torch.Tensor | None = None,
                           context: StepContext = None):
        """Update model meta."""
        if context.is_decoding:
            return self._update_model_meta_decoding(context)
        else:
            return self._update_model_meta_prefilling(context)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class Qwen3OmniInputProcessor(BaseModelInputProcessor):
    """Qwen3 Omni input processor."""

    def __init__(self, config: PretrainedConfig) -> None:
        self.config = config

    def _make_image_mm_data(self, input_mm: Dict[str, Any]) -> MultiModalData:
        """Make image MultiModalData."""
        pixel_values = input_mm['pixel_values']
        image_grid_thw = input_mm['image_grid_thw']
        offset = input_mm['offset']
        start = offset
        image_token_id = input_mm['image_token_id']
        num_pad = input_mm['mm_token_num']
        if isinstance(num_pad, torch.Tensor):
            num_pad = num_pad.item()

        mm_data = MultiModalData(modality=Modality.IMAGE,
                                 data=pixel_values,
                                 start=start,
                                 end=start + num_pad,
                                 meta=dict(grid_thw=image_grid_thw, image_token_id=image_token_id))
        return mm_data

    def _make_video_mm_data(self, input_mm: Dict[str, Any]) -> MultiModalData:
        """Make video MultiModalData."""
        pixel_values_videos = input_mm['pixel_values_videos']
        video_grid_thw = input_mm['video_grid_thw']
        offset = input_mm['offset']
        start = offset
        video_token_id = input_mm['video_token_id']
        num_pad = input_mm['mm_token_num']
        if isinstance(num_pad, torch.Tensor):
            num_pad = num_pad.item()

        mm_data = MultiModalData(modality=Modality.VIDEO,
                                 data=pixel_values_videos,
                                 start=start,
                                 end=start + num_pad,
                                 meta=dict(
                                     grid_thw=video_grid_thw,
                                     video_token_id=video_token_id,
                                     second_per_grid=input_mm.get('second_per_grid'),
                                 ))
        return mm_data

    def _make_audio_mm_data(self, input_mm: Dict[str, Any]) -> MultiModalData:
        """Make audio MultiModalData."""
        input_features = input_mm['input_features']
        offset = input_mm['offset']
        start = offset
        audio_token_id = input_mm['audio_token_id']
        num_pad = input_mm['mm_token_num']
        if isinstance(num_pad, torch.Tensor):
            num_pad = num_pad.item()

        mm_data = MultiModalData(modality=Modality.AUDIO,
                                 data=input_features,
                                 start=start,
                                 end=start + num_pad,
                                 meta=dict(
                                     audio_token_id=audio_token_id,
                                     audio_feature_lengths=input_mm.get('audio_feature_lengths'),
                                 ))
        return mm_data

    def preprocess_input(self,
                         input_ids: List[int],
                         input_multimodals: List[Dict[str, Any]] = None,
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
            elif modality == Modality.AUDIO:
                mm_data = self._make_audio_mm_data(input_mm)

            input_mm_data.append(mm_data)

        result = PreprocessInputResult(input_ids=input_ids, input_multimodals=dict(mm_data=input_mm_data))

        return result
