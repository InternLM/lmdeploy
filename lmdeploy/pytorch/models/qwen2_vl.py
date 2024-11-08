# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Callable, Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import (ApplyRotaryEmb, Attention, RMSNorm, RopeType,
                                 SiluAndMul, build_rotary_embedding)
from lmdeploy.pytorch.nn.linear import (build_colwise_linear,
                                        build_merged_colwise_linear,
                                        build_qkv_proj, build_rowwise_linear)
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin, next_power_of_2
from .utils.multimodal import MultiModalMixin


def _apply_mrope_selection(hidden_states: torch.Tensor,
                           mrope_position_ids: torch.Tensor,
                           mrope_section: List[int],
                           position_ids: torch.Tensor,
                           rotary_emb_func: Callable):
    _mrope_position_ids = torch.zeros(3,
                                      position_ids.shape[-1],
                                      dtype=position_ids.dtype,
                                      device=position_ids.device)
    _mrope_position_ids[:, :mrope_position_ids.shape[-1]] = mrope_position_ids
    cos, sin = rotary_emb_func(hidden_states, _mrope_position_ids)
    _cos = torch.zeros(cos.shape[1],
                       cos.shape[-1],
                       dtype=cos.dtype,
                       device=cos.device)
    _sin = torch.zeros_like(_cos)
    mrope_section = mrope_section * 2

    def _apply_split(src, dst):
        start = 0
        for i, m in enumerate(src.split(mrope_section, dim=-1)):
            dst[:, start:start + mrope_section[i]] = m[i % 3]
            start += mrope_section[i]

    _apply_split(cos, _cos)
    _apply_split(sin, _sin)

    return _cos, _sin


class Qwen2Attention(nn.Module):
    """Rewrite module of Qwen2Attention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)

        # packed qkv
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        # rotary embedding
        self.apply_rotary_pos_emb = ApplyRotaryEmb()

        # attention
        self.attn_fwd = Attention(
            num_heads,
            head_dim,
            num_kv_heads=num_key_value_heads,
            v_head_size=head_dim,
            sliding_window=config.sliding_window,
        )

        # o_proj
        self.o_proj = build_rowwise_linear(num_heads * head_dim,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quantization_config,
                                           dtype=dtype,
                                           device=device,
                                           is_tp=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attn_metadata: Any = None,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(
            qkv_states)

        # apply rotary embedding
        cos, sin = rotary_pos_emb
        query_states, key_states = self.apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            inplace=True,
        )

        # attention
        attn_output = self.attn_fwd(
            query_states,
            key_states,
            value_states,
            past_key_value[0],
            past_key_value[1],
            attn_metadata,
            k_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None
            if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen2MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(config.intermediate_size,
                                              config.hidden_size,
                                              bias=False,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=True)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Qwen2DecoderLayer(nn.Module):
    """decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.self_attn = Qwen2Attention(config, dtype=dtype, device=device)

        # builf MLP
        self.mlp = Qwen2MLP(config, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
            quant_config=quantization_config,
            dtype=dtype,
            device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Optional[List[torch.FloatTensor]],
        residual: Optional[torch.Tensor] = None,
        attn_metadata: Any = None,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            past_key_value=past_key_value,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Qwen2Model(nn.Module):
    """model."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.mrope_section = config.rope_scaling['mrope_section']
        quantization_config = getattr(config, 'quantization_config', None)

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size,
                            config.rms_norm_eps,
                            quant_config=quantization_config,
                            dtype=dtype,
                            device=device)

        # build rotary embedding
        emb_type = RopeType.LinearScaling
        rope_dim = config.hidden_size // config.num_attention_heads
        rope_max_pos_emb = config.max_position_embeddings
        rope_base = config.rope_theta
        self.rotary_emb = build_rotary_embedding(
            rope_dim,
            rope_max_pos_emb,
            rope_base,
            emb_type=emb_type,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        mrope_position_ids: torch.LongTensor = None,
    ):
        """Rewrite of LlamaModel.forward."""

        # token embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # rotary embedding
        if mrope_position_ids is None:
            cos, sin = self.rotary_emb(hidden_states, position_ids)
            cos, sin = cos[0], sin[0]
        else:
            cos, sin = _apply_mrope_selection(hidden_states,
                                              mrope_position_ids,
                                              self.mrope_section, position_ids,
                                              self.rotary_emb)
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

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.embed_tokens


class PatchEmbed(nn.Module):
    """Patch Embed."""

    def __init__(self,
                 patch_size: int = 14,
                 temporal_patch_size: int = 2,
                 in_channels: int = 3,
                 embed_dim: int = 1152,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels,
                              embed_dim,
                              kernel_size=kernel_size,
                              stride=kernel_size,
                              bias=False,
                              dtype=dtype,
                              device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(-1, self.in_channels,
                                           self.temporal_patch_size,
                                           self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(
            -1, self.embed_dim)
        return hidden_states


class VisionRotaryEmbedding(nn.Module):
    """vision rotary embedding."""

    def __init__(self,
                 dim: int,
                 theta: float = 10000.0,
                 device: torch.device = None) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta**(
            torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen,
                           device=self.inv_freq.device,
                           dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor,
                                freqs: torch.Tensor) -> torch.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output


class VisionAttention(nn.Module):
    """Vision attention."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        dim = config.embed_dim
        num_heads = config.num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        # packed qkv
        self.qkv = build_qkv_proj(
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

        # o_proj
        self.proj = build_rowwise_linear(dim,
                                         dim,
                                         bias=True,
                                         quant_config=quantization_config,
                                         dtype=dtype,
                                         device=device,
                                         is_tp=True)

    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
        import math
        seq_length = hidden_states.shape[0]
        # qkv proj
        qkv_states = self.qkv(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        q, k, v = self.qkv.split_qkv(qkv_states)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                        rotary_pos_emb).squeeze(0)

        attention_mask = torch.full([1, seq_length, seq_length],
                                    torch.finfo(q.dtype).min,
                                    device=q.device,
                                    dtype=q.dtype)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                           cu_seqlens[i - 1]:cu_seqlens[i]] = 0

        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
            self.head_dim)
        attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)

        # o proj
        attn_output = self.proj(attn_output)
        return attn_output


class VisionMlp(nn.Module):
    """Vision mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        dim = config.embed_dim
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        quantization_config = getattr(config, 'quantization_config', None)
        # gate up
        self.fc1 = build_colwise_linear(
            dim,
            hidden_dim,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
        )

        # silu and mul
        self.act = nn.SiLU(True)

        # down
        self.fc2 = build_rowwise_linear(hidden_dim,
                                        dim,
                                        bias=True,
                                        quant_config=quantization_config,
                                        dtype=dtype,
                                        device=device,
                                        is_tp=True)

    def forward(self, x):
        """forward."""
        return self.fc2(self.act(self.fc1(x)))


class Qwen2VLVisionBlock(nn.Module):
    """Vision block."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(config.embed_dim,
                                  eps=1e-6,
                                  dtype=dtype,
                                  device=device)
        self.norm2 = nn.LayerNorm(config.embed_dim,
                                  eps=1e-6,
                                  dtype=dtype,
                                  device=device)

        self.attn = VisionAttention(config, dtype=dtype, device=device)

        self.mlp = VisionMlp(config, dtype=dtype, device=device)

    def forward(self, hidden_states, cu_seqlens,
                rotary_pos_emb) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class PatchMerger(nn.Module):
    """PatchMerger."""

    def __init__(self,
                 dim: int,
                 context_dim: int,
                 spatial_merge_size: int = 2,
                 dtype: torch.dtype = None,
                 device: torch.device = None) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = nn.LayerNorm(context_dim,
                                 eps=1e-6,
                                 dtype=dtype,
                                 device=device)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size,
                      self.hidden_size,
                      dtype=dtype,
                      device=device),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim, dtype=dtype, device=device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Qwen2VisionTransformerPretrainedModel(nn.Module):
    """Vision transformer."""

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
            dtype=dtype,
            device=device,
        )

        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2,
                                                    device=device)

        self.blocks = nn.ModuleList([
            Qwen2VLVisionBlock(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(config.depth)
        ])
        self.merger = PatchMerger(dim=config.hidden_size,
                                  context_dim=config.embed_dim,
                                  spatial_merge_size=config.spatial_merge_size,
                                  dtype=dtype,
                                  device=device)

    def rot_pos_emb(self, grid_thw):
        """rotary position embedding."""
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor,
                rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        """forward."""
        hidden_states = self.patch_embed(hidden_states)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                             grid_thw[:, 0]).cumsum(
                                                 dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)

        for blk in self.blocks:
            hidden_states = blk(hidden_states,
                                cu_seqlens=cu_seqlens,
                                rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)


OPTIONAL_KEYS = ['resized_height', 'resized_width', 'min_pixels', 'max_pixels']


class Qwen2VLForConditionalGeneration(nn.Module, CudaGraphMixin,
                                      MultiModalMixin):
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

        # preprocessor
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(config.name_or_path)

        # build vision model
        self.visual = Qwen2VisionTransformerPretrainedModel(
            config.vision_config,
            dtype=dtype,
            device=device,
        )
        # build model
        self.model = Qwen2Model(config, dtype=dtype, device=device)
        # build lm_head
        self.lm_head = build_rowwise_linear(config.hidden_size,
                                            config.vocab_size,
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
        grid_thw: torch.Tensor = None,
        vis_pos_emb: torch.Tensor = None,
        **kwargs,
    ):
        """model forward, return logits."""
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                dtype = inputs_embeds.dtype
                pixel_values = pixel_values.to(dtype)
                vis_pos_emb = vis_pos_emb.to(dtype)
                image_embeds = self.visual(pixel_values,
                                           grid_thw=grid_thw,
                                           rotary_pos_emb=vis_pos_emb)
                image_mask = ((input_ids == self.config.image_token_id
                               ).unsqueeze(-1).expand_as(inputs_embeds).to(
                                   inputs_embeds.device))
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask, image_embeds[None])

        hidden_states = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """compute logits of the model output."""
        return self.lm_head(hidden_states)

    def update_weights(self):
        """update weights."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        """get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
        inputs_embeds: Optional[torch.Tensor] = None,
        context: StepContext = None,
    ):
        """prepare input."""

        # get input_ids, position_ids and attention metadatas
        input_ids = context.input_ids
        position_ids = context.position_ids
        attn_metadata = context.attn_metadata

        pixel_values = None
        grid_thw = None
        vis_pos_emb = None
        if context.input_multimodals is not None:
            image_data = [
                input_mm['image'] for input_mm in context.input_multimodals
            ]
            # flatten batch
            image_data = [data for im_data in image_data for data in im_data]
            pixel_values = torch.cat([data.data for data in image_data])
            grid_thw = torch.cat(
                [data.meta['grid_thw'] for data in image_data])
            vis_pos_emb = self.visual.rot_pos_emb(grid_thw.cpu())

        # process vision embeddings
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = context.input_embedding_indexing
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:,
                          vision_embedding_indexing, :] = vision_embeddings.to(
                              inputs_embeds)

        # inputs of forward
        return dict(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=None,
            pixel_values=pixel_values,
            grid_thw=grid_thw,
            vis_pos_emb=vis_pos_emb,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """load weights."""
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
            if ('rotary_emb.cos_cached' in name
                    or 'rotary_emb.sin_cached' in name):
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
        """make cudagraph buffers from forward inputs."""
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta,
                                                       **kwargs)
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'] = mrope_position_ids.new_zeros(
                3, max_tokens)

        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """fill cudagraph buffers from forward inputs."""

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta,
                                                    **kwargs)

        input_ids = kwargs.get('input_ids')
        attn_metadata = kwargs.get('attn_metadata')
        block_offsets = attn_metadata.block_offsets
        num_tokens = input_ids.size(-1)
        batch_size, _ = block_offsets.size()
        new_batch_size = next_power_of_2(batch_size)

        is_decoding = graph_meta.is_decoding
        input_buffers = graph_meta.input_buffers
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers[
                'mrope_position_ids'][:, :num_tokens] = mrope_position_ids
            if is_decoding:
                new_inputs['mrope_position_ids'] = input_buffers[
                    'mrope_position_ids'][:, :new_batch_size]
            else:
                new_inputs['mrope_position_ids'] = input_buffers[
                    'mrope_position_ids']

        return new_inputs

    def prepare_multimodal_input(self, input_ids, input_multimodals, **kwargs):
        """prepare multimodal input."""
        from qwen_vl_utils import process_vision_info

        from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
        global OPTIONAL_KEYS

        multimodals_dict = dict()
        multimodals_dict['image'] = []

        input_multimodals = sorted(input_multimodals, key=lambda mm: mm.loc)

        cum_pad = 0
        image_token_id = self.config.image_token_id

        # image
        for in_mm in input_multimodals:
            image = in_mm.data
            param = in_mm.meta
            param = dict() if param is None else param
            item = dict(type='image', image=image)
            item.update({k: param[k] for k in OPTIONAL_KEYS if k in param})
            messages = [dict(content=[item])]
            image_inputs, _ = process_vision_info(messages)
            image_inputs = self.processor.image_processor(images=image_inputs,
                                                          videos=None,
                                                          return_tensors='pt')
            pixel_values = image_inputs['pixel_values']
            image_grid_thw = image_inputs['image_grid_thw']
            pad_size = pixel_values.size(0) // 4
            loc = in_mm.loc
            start = loc + cum_pad
            end = start + pad_size
            cum_pad += pad_size
            input_ids = input_ids[:start] + [image_token_id
                                             ] * pad_size + input_ids[start:]

            mm_tensor = MultiModalTensor(data=pixel_values,
                                         start=start,
                                         end=end,
                                         meta=dict(grid_thw=image_grid_thw))
            multimodals_dict['image'].append(mm_tensor)

        return input_ids, multimodals_dict
