# Copyright (c) OpenMMLab. All rights reserved.

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from packaging import version
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor, PreprocessInputResult
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.models.utils.micro_batch import enable_micro_batch, split_batch
from lmdeploy.pytorch.multimodal.data_type import MultiModalTensor
from lmdeploy.pytorch.nn import LayerNorm, RMSNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_o_proj, build_qkv_proj, build_rowwise_linear
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_weight

from .patch import build_model_from_hf_config
from .utils.cudagraph import CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


class Gating(nn.Module):

    def __init__(self, hidden_size=2048, expansion_factor=4, dtype=None, device=None):
        super().__init__()

        mid_dim = hidden_size * expansion_factor

        def mlp_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim, bias=True, dtype=dtype, device=device),
                nn.GELU(),
                nn.Identity(),
                nn.Linear(out_dim, in_dim, bias=True, dtype=dtype, device=device),
                nn.Identity(),
                nn.LayerNorm(in_dim, dtype=dtype, device=device),
            )

        self.block1 = mlp_block(hidden_size, mid_dim)
        self.block2 = mlp_block(hidden_size, mid_dim)
        self.block3 = mlp_block(hidden_size, mid_dim)
        self.block4 = mlp_block(hidden_size, mid_dim)

        self.gate = nn.Sequential(
            nn.LayerNorm(hidden_size, dtype=dtype, device=device),
            nn.Linear(hidden_size, 2, bias=True, dtype=dtype, device=device)  # 2 experts
        )

    def forward(self, x):
        x = x + self.block1(x)
        x = x + self.block2(x)
        x = x + self.block3(x)
        x = x + self.block4(x)

        logits = self.gate(x)  # shape: [B, 2]
        probs = torch.softmax(logits, dim=-1)
        return probs


class CrossAttentionPooling(nn.Module):

    def __init__(self, dim, num_heads=16, dtype=None, device=None):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, dim, dtype=dtype, device=device))  # [1, D]

        self.attn1 = nn.MultiheadAttention(embed_dim=dim,
                                           num_heads=num_heads,
                                           batch_first=True,
                                           dtype=dtype,
                                           device=device)
        self.norm1 = nn.LayerNorm(dim, dtype=dtype, device=device)

        self.attn2 = nn.MultiheadAttention(embed_dim=dim,
                                           num_heads=num_heads,
                                           batch_first=True,
                                           dtype=dtype,
                                           device=device)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype, device=device)

        self.attn3 = nn.MultiheadAttention(embed_dim=dim,
                                           num_heads=num_heads,
                                           batch_first=True,
                                           dtype=dtype,
                                           device=device)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype, device=device)

        self.attn4 = nn.MultiheadAttention(embed_dim=dim,
                                           num_heads=num_heads,
                                           batch_first=True,
                                           dtype=dtype,
                                           device=device)
        self.norm4 = nn.LayerNorm(dim, dtype=dtype, device=device)

    def forward(self, batched_tokens: list[torch.Tensor]):
        """
        batched_tokens: List of Tensors of shape [Ti, D], length = B
        """
        B = len(batched_tokens)
        D = batched_tokens[0].shape[-1]
        device = batched_tokens[0].device

        # 1. Padding
        max_len = max(t.shape[0] for t in batched_tokens)
        dtype = self.query_token.dtype
        padded = torch.zeros(B, max_len, D, dtype=dtype, device=device)
        padding_mask = torch.ones(B, max_len, dtype=torch.bool, device=device)

        for i, t in enumerate(batched_tokens):
            L = t.shape[0]
            padded[i, :L] = t
            padding_mask[i, :L] = False

        # 2. Query token: [B, 1, D]
        query = self.query_token.unsqueeze(0).expand(B, -1, -1)  # learnable token for each sample

        # 3. First attention
        out1, _ = self.attn1(query, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out1 = self.norm1(out1)

        # 4. Second attention
        out2, _ = self.attn2(out1, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out2 = self.norm2(out2)

        out3, _ = self.attn2(out2, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out3 = self.norm2(out3)

        out4, _ = self.attn2(out3, padded, padded, key_padding_mask=padding_mask)  # [B, 1, D]
        out4 = self.norm2(out4)

        return out4.squeeze(1)


class InternVisionEmbeddings(nn.Module):
    """Intern vision embedding."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.empty(1, 1, self.embed_dim, dtype=dtype, device=device), )

        self.patch_embedding = nn.Conv2d(in_channels=3,
                                         out_channels=self.embed_dim,
                                         kernel_size=self.patch_size,
                                         stride=self.patch_size,
                                         dtype=dtype,
                                         device=device)

        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(
            torch.empty(1, self.num_positions, self.embed_dim, dtype=dtype, device=device))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(1, self.image_size // self.patch_size, self.image_size // self.patch_size,
                                              -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic',
                                  align_corners=False).reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat(
            [self.position_embedding[:, :1, :],
             self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)],
            dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


NORM2FN = {
    'rms_norm': RMSNorm,
    'layer_norm': LayerNorm,
}


@torch.compile(dynamic=True)
def pre_rms_norm(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pre rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance_q = (q * q).sum(-1, keepdim=True)
    variance_k = (k * k).sum(-1, keepdim=True)
    variance = torch.stack([variance_q, variance_k], dim=0)
    return variance


@torch.compile(dynamic=True)
def post_rms_norm(q: torch.Tensor, k: torch.Tensor, weight_q: torch.Tensor, weight_k: torch.Tensor,
                  variance: torch.Tensor, eps: float, embed_dim: int, dtype: torch.dtype):
    """Post rms norm."""
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    variance = variance / embed_dim + eps
    variance_q, variance_k = variance
    q = q * torch.rsqrt(variance_q)
    q = q.to(dtype) * weight_q
    k = k * torch.rsqrt(variance_k)
    k = k.to(dtype) * weight_k
    return q, k


class InternAttention(nn.Module):
    """Intern vl attention."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv = build_qkv_proj(
            self.embed_dim,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            head_size=self.head_dim,
            bias=config.qkv_bias,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
        )

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                dtype=dtype,
                device=device,
                tp=True,
                align=self.head_dim,
            )
            self.k_norm = RMSNorm(
                self.embed_dim,
                eps=config.layer_norm_eps,
                dtype=dtype,
                device=device,
                tp=True,
                align=self.head_dim,
            )

        self.scale = self.head_dim**-0.5

        # o_proj
        self.proj = build_o_proj(self.embed_dim,
                                 self.embed_dim,
                                 bias=True,
                                 quant_config=quantization_config,
                                 dtype=dtype,
                                 device=device,
                                 is_tp=True,
                                 tp_align_size=self.head_dim)

    def pre_rms_norm(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Pre rms norm."""
        return pre_rms_norm(q, k)

    def post_rms_norm(self, q: torch.Tensor, k: torch.Tensor, variance: torch.Tensor,
                      dtype: torch.dtype) -> torch.Tensor:
        """Post rms norm."""
        eps = self.config.layer_norm_eps
        return post_rms_norm(q, k, self.q_norm.weight, self.k_norm.weight, variance, eps, self.embed_dim, dtype)

    def qkv_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        import lmdeploy.pytorch.distributed as dist
        q_shape = q.shape
        k_shape = k.shape
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)

        tp, _ = get_tp_world_rank()
        if tp == 1:
            q = self.q_norm(q).view(q_shape)
            k = self.k_norm(k).view(k_shape)
            return q, k

        # variance
        variance = self.pre_rms_norm(q, k)
        dist.all_reduce(variance)
        q, k = self.post_rms_norm(q, k, variance, q.dtype)
        q = q.view(q_shape)
        k = k.view(k_shape)

        return q, k

    def forward(self, hidden_states):
        """forward."""

        # qkv proj
        qkv_states = self.qkv(hidden_states)
        q, k, v = self.qkv.split_qkv(qkv_states)

        if self.qk_normalization:
            q, k = self.qkv_norm(q, k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        # o proj
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.flatten(-2, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class InternMLP(nn.Module):
    """Intern vl mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        from transformers.activations import ACT2FN
        self.config = config
        quantization_config = getattr(config, 'quantization_config', None)
        self.act = ACT2FN[config.hidden_act]

        self.fc1 = build_colwise_linear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=True,
            dp_disable_tp=True,
        )

        self.fc2 = build_rowwise_linear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quantization_config,
            dtype=dtype,
            device=device,
            is_tp=True,
            dp_disable_tp=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    """Intern vision encoder layer."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = getattr(config, 'norm_type', 'rms_norm')

        self.attn = InternAttention(config, dtype=dtype, device=device)
        self.mlp = InternMLP(config, dtype=dtype, device=device)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps, dtype=dtype, device=device)

        self.ls1 = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))
        self.ls2 = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype, device=device))

    @enable_micro_batch(param_name='hidden_states', index=0)
    def _attn(self, hidden_states):
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states).to(hidden_states[0].dtype)) * self.ls1
        return hidden_states

    @enable_micro_batch(param_name='hidden_states', index=0)
    def _mlp(self, hidden_states):
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states).to(hidden_states.dtype)) * self.ls2
        return hidden_states

    def forward(
        self,
        hidden_states,
    ):
        hidden_states = self._attn(hidden_states)
        hidden_states = self._mlp(hidden_states)
        return hidden_states


class InternVisionEncoder(nn.Module):
    """Intern vision encoder."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [InternVisionEncoderLayer(config, dtype=dtype, device=device) for idx in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
    ):
        """forward."""
        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            layer_outputs = encoder_layer(hidden_states, )
            hidden_states = layer_outputs
        return hidden_states


@vlm_model
class InternVisionModel(nn.Module):
    """Intern vision model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config

        self.embeddings = InternVisionEmbeddings(config, dtype=dtype, device=device)
        self.encoder = InternVisionEncoder(config, dtype=dtype, device=device)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        """forward."""
        assert pixel_values.dim() == 4
        hidden_states = self.embeddings(pixel_values)

        encoder_outputs = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = encoder_outputs

        return last_hidden_state


class InternVLChatModel(nn.Module, DeployModelMixin, CudaGraphMixin):

    def __init__(self,
                 config: PretrainedConfig,
                 ctx_mgr: StepContextManager,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.select_layer = config.select_layer

        llm_config = config.llm_config
        self.llm_arch_name = llm_config.architectures[0]
        self.is_mono = self.llm_arch_name == 'InternLM2VEForCausalLM'

        vision_config = config.vision_config
        if self.is_mono:
            from .internvl_patch import InternVisionPatchModel
            self.vision_model = InternVisionPatchModel(
                vision_config,
                dtype=dtype,
                device=device,
            )
        else:
            self.vision_model = InternVisionModel(vision_config, dtype=dtype, device=device)

        self.language_model = build_model_from_hf_config(llm_config, dtype=dtype, device=device)

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.downsample_ratio = config.downsample_ratio
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2, dtype=dtype, device=device),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2,
                      llm_hidden_size,
                      bias=True,
                      dtype=dtype,
                      device=device), nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size, bias=True, dtype=dtype, device=device))

        # for Mono-InternVL
        if self.is_mono:
            assert dtype != torch.float16, ('Currently Mono-InternVL does not support FP16 due to'
                                            'numerical instability. Please use BF16 instead.')

        self.input_processor = InternVLInputProcessor(self.config, dtype)

        self.compile_vit = False

        self.flash_mode = getattr(config, 'flash_mode', None)
        if self.flash_mode is not None:
            self.flash_relative_threshold = config.flash_relative_threshold
            self.flash_absolute_threshold = config.flash_absolute_threshold

            self.mlp2 = nn.Sequential(
                nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**4, dtype=dtype, device=device),
                nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**4,
                          llm_hidden_size * 2,
                          bias=True,
                          dtype=dtype,
                          device=device), nn.GELU(), nn.Identity(),
                nn.Linear(llm_hidden_size * 2, llm_hidden_size * 2, bias=True, dtype=dtype, device=device), nn.GELU(),
                nn.Identity(), nn.Linear(llm_hidden_size * 2, llm_hidden_size, bias=True, dtype=dtype, device=device))

            self.pooling_before_gating = CrossAttentionPooling(dim=vit_hidden_size, dtype=dtype, device=device)
            self.gating = Gating(hidden_size=vit_hidden_size, dtype=dtype, device=device)

    def compile_model(self):
        torch_version = version.parse(torch.__version__)
        if torch_version < version.parse('2.5.0'):
            return

        tp, _ = get_tp_world_rank()
        if torch_version >= version.parse('2.6.0') and tp > 1:
            torch._inductor.config.reorder_for_compute_comm_overlap = True
            if isinstance(self.vision_model, InternVisionModel):
                self.vision_model.encoder.forward = split_batch(self.vision_model.encoder.forward,
                                                                'inputs_embeds',
                                                                index=0)

        self.extract_feature = torch.compile(self.extract_feature, mode='max-autotune-no-cudagraphs')
        self.compile_vit = True
        self.has_compiled_vit = False

    def _mark_dynamic_once(self, pixel_values, dims):
        """Call torch._dynamo.mark_dynamic to avoid recompile."""
        if not self.compile_vit or self.has_compiled_vit or pixel_values is None:
            return

        torch._dynamo.mark_dynamic(pixel_values, dims)
        self.has_compiled_vit = True

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale -->
        # N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        """Extract vision feature."""
        assert self.select_layer == -1
        vit_embeds = self.vision_model(pixel_values)
        if self.is_mono:
            if int(vit_embeds.shape[1]**0.5)**2 != vit_embeds.shape[1]:
                vit_embeds = vit_embeds[:, 1:, :]
        else:
            vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def compress_visual_tokens_in_sentence(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        img_context_token_id: int,
        gate_result,
    ) -> tuple:
        # reshape
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)
        input_ids = input_ids.reshape(B * N)

        N, C = input_embeds.shape
        lengths, starts, ends = self.get_image_num_per_sample(input_ids, img_context_token_id)

        keep_mask = torch.ones(N, dtype=torch.bool, device=input_embeds.device)

        total_blocks = 0
        block_counts = []
        for length in lengths.tolist():
            if length % 256 != 0:
                raise ValueError(f'l % 256 != 0, l = {length}')
            num_blocks = length // 256
            block_counts.append(num_blocks)
            total_blocks += num_blocks

        flag_idx = 0
        for s, e, l, num_blocks in zip(starts.tolist(), ends.tolist(), lengths.tolist(), block_counts):
            for i in range(num_blocks):
                block_start = s + i * 256
                block_end = block_start + 256

                compress = gate_result[flag_idx]
                flag_idx += 1

                if compress:
                    keep_mask[block_start + 64:block_end] = False

        # update
        new_input_embeds = input_embeds[keep_mask.to(input_embeds.device), :]
        new_input_ids = input_ids[keep_mask.to(input_ids.device)]
        new_image_mask = (new_input_ids == img_context_token_id)

        # reshape back
        new_input_ids = new_input_ids.reshape(B, -1)
        new_input_embeds = new_input_embeds.reshape(B, -1, C)

        # since multiple sequences may concat together, we need to update the seqlens individually
        # we calculate compressed token len for each sequence, and get new len for each sequence
        crt_ctx = self.ctx_mgr.current_context()
        seq_lengths = crt_ctx.q_seqlens
        # split the keep_mask into chunks corresponding to each original sequence
        mask_chunks = torch.split(keep_mask, seq_lengths.tolist())
        # the new length of each sequence is the number of tokens kept (sum of True values)
        new_seq_lengths = [chunk.sum().item() for chunk in mask_chunks]

        return new_input_embeds, new_input_ids, new_image_mask, new_seq_lengths

    def get_image_num_per_sample(self, input_ids: torch.Tensor, img_context_token_id: int):
        input_ids = input_ids.squeeze(0)  # (N,)
        selected = (input_ids == img_context_token_id)
        padded = torch.cat(
            [torch.tensor([0], device=selected.device),
             selected.int(),
             torch.tensor([0], device=selected.device)])
        diff = torch.diff(padded)

        starts = (diff == 1).nonzero(as_tuple=True)[0]
        ends = (diff == -1).nonzero(as_tuple=True)[0]
        lengths = ends - starts

        return lengths, starts, ends

    def split_and_merge(self, features: torch.Tensor, split_sizes: torch.Tensor):
        """
        features: Tensor of shape [T, 1024, 1024]
        split_sizes: 1D Tensor like [3, 3, 4] — tile of each sample

        returns: List of Tensors of shape [tile_i * 1024, 1024]
        """
        # split features -> each sample a tile list
        tile_splits = torch.split(features, split_sizes, dim=0)

        # merge the first two dimensions: tile * 1024 × 1024
        merged = [x.reshape(-1, x.shape[-1]) for x in tile_splits]

        return merged

    def extract_feature_flash(self, pixel_values, lengths):

        vit_embeds_1024 = self.vision_model(pixel_values)

        vit_embeds_1024 = vit_embeds_1024[:, 1:, :]
        h = w = int(vit_embeds_1024.shape[1]**0.5)
        vit_embeds_1024 = vit_embeds_1024.reshape(vit_embeds_1024.shape[0], h, w, -1)

        # begin moe
        lengths = [int(x) for x in lengths.tolist()]
        vit_embeds_1024_split_and_merge = self.split_and_merge(vit_embeds_1024, lengths)

        gate = self.pooling_before_gating(vit_embeds_1024_split_and_merge)
        gate = self.gating(gate)

        vit_embeds_256 = vit_embeds_1024

        with torch.no_grad():
            vit_embeds_64 = self.pixel_shuffle(vit_embeds_1024, scale_factor=self.downsample_ratio**2)
            vit_embeds_64 = vit_embeds_64.reshape(vit_embeds_64.shape[0], -1, vit_embeds_64.shape[-1])
            vit_embeds_64 = self.mlp2(vit_embeds_64)

            vit_embeds_256 = self.pixel_shuffle(vit_embeds_256, scale_factor=self.downsample_ratio)
            vit_embeds_256 = vit_embeds_256.reshape(vit_embeds_256.shape[0], -1, vit_embeds_256.shape[-1])
            vit_embeds_256 = self.mlp1(vit_embeds_256)

        return vit_embeds_64, vit_embeds_256, gate

    def extract_and_compress(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, img_context_token_id: int):
        lang_embeds = self.language_model.get_input_embeddings()(input_ids)

        self._mark_dynamic_once(pixel_values, [0])

        lengths, starts, ends = self.get_image_num_per_sample(input_ids, img_context_token_id)
        lengths = lengths // 256
        lengths_sum = torch.ones(int(lengths.sum().item()), dtype=torch.int64)
        lengths = lengths_sum.repeat_interleave(1)
        vit_embeds_64, vit_embeds_256, gate_result = self.extract_feature_flash(pixel_values, lengths)

        relative_threshold_value = torch.quantile(gate_result[:, 0].to(torch.float32), self.flash_relative_threshold)
        gate_result = (gate_result[:, 0] > relative_threshold_value) & (gate_result[:, 0]
                                                                        >= self.flash_absolute_threshold)

        selected_embeds = [
            vit_embeds_64[i] if gate_result[i] else vit_embeds_256[i] for i in range(gate_result.size(0))
        ]

        vit_embeds = torch.cat(selected_embeds, dim=0)

        # compress visual tokens in sentence
        new_lang_embeds, new_input_ids, new_image_mask, new_seq_lengths = self.compress_visual_tokens_in_sentence(
            input_embeds=lang_embeds,
            input_ids=input_ids,
            img_context_token_id=img_context_token_id,
            gate_result=gate_result,
        )

        return vit_embeds, new_lang_embeds, new_input_ids, new_image_mask, new_seq_lengths

    def update_forward_inputs(self, input_ids: torch.Tensor, new_seqlens: List[int],
                              context: StepContext) -> StepContext:
        """Update the forward inputs, position_ids and attention metadata."""
        from lmdeploy.pytorch.model_inputs import ModelInputs

        crt_ctx = self.ctx_mgr.current_context()
        assert crt_ctx is not None, 'Current context cannot be None.'

        # update model metas
        prev_lens = [0] * len(context.model_metas)
        has_model_metas = context.model_metas is not None and context.model_metas[0] is not None
        if has_model_metas:
            prev_lens = [meta.get('new_seqlen', 0) for meta in context.model_metas]

            for idx, meta in enumerate(context.model_metas):
                meta.update({'new_seqlen': prev_lens[idx] + new_seqlens[idx]})
        else:
            context.model_metas = [dict(new_seqlen=seqlen) for seqlen in new_seqlens]

        # create new model inputs and context, to get updated position_ids and attn_metadata
        device = input_ids.device
        total_msgs = len(new_seqlens)
        kv_seqlens = torch.tensor([meta['new_seqlen'] for meta in context.model_metas], dtype=torch.long)
        new_model_inputs = ModelInputs(input_ids=input_ids,
                                       seq_length=torch.tensor(new_seqlens, device=device, dtype=torch.long),
                                       history_lengths=torch.tensor(prev_lens, device=device, dtype=torch.long),
                                       block_offsets=crt_ctx.block_offsets,
                                       is_decoding=False,
                                       num_ignored_history=torch.zeros(total_msgs, device=device, dtype=torch.long),
                                       max_q_seqlen=kv_seqlens.max().item(),
                                       max_kv_seqlen=kv_seqlens.max().item(),
                                       sum_kv_seqlen=kv_seqlens.sum().item(),
                                       model_metas=context.model_metas)
        new_ctx = self.ctx_mgr.build_context(new_model_inputs, crt_ctx.model_config)

        # update attributes of the context in model agent
        context.q_seqlens = new_ctx.q_seqlens

        return new_ctx.position_ids, new_ctx.attn_metadata

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any = None,
        pixel_values: torch.Tensor = None,
        image_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        vision_embedding_indexing: torch.Tensor = None,
        text_embedding_indexing: torch.Tensor = None,
        image_token_id: int = None,
        context: StepContext = None,
        **kwargs,
    ):
        if inputs_embeds is None and pixel_values is not None:
            if self.flash_mode:
                # extract feature and compress visual tokens
                vit_embeds, lang_embeds, input_ids, image_mask, new_seqlens = self.extract_and_compress(
                    pixel_values, input_ids, image_token_id)

                # update forward inputs
                position_ids, attn_metadata = self.update_forward_inputs(input_ids, new_seqlens, context)
            else:
                # extract feature
                self._mark_dynamic_once(pixel_values, [0])
                vit_embeds = self.extract_feature(pixel_values)
                lang_embeds = self.language_model.get_input_embeddings()(input_ids)

            lang_embeds.masked_scatter_(image_mask[..., None], vit_embeds)

            inputs_embeds = lang_embeds

        if self.is_mono:
            return self.language_model.forward(input_ids=input_ids,
                                               inputs_embeds=inputs_embeds,
                                               past_key_values=past_key_values,
                                               position_ids=position_ids,
                                               attn_metadata=attn_metadata,
                                               vision_embedding_indexing=vision_embedding_indexing,
                                               text_embedding_indexing=text_embedding_indexing)
        else:
            return self.language_model.forward(input_ids=input_ids,
                                               inputs_embeds=inputs_embeds,
                                               past_key_values=past_key_values,
                                               position_ids=position_ids,
                                               attn_metadata=attn_metadata)

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.language_model.get_logits(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()

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
        vision_embeddings = context.input_embeddings
        vision_embedding_indexing = None

        # vision inputs
        pixel_values = None
        image_mask = None
        image_token_id = None
        if context.input_multimodals is not None:
            pixel_values = [input_mm.get('image', []) for input_mm in context.input_multimodals]
            # flatten batch
            pixel_values = [data for im_data in pixel_values for data in im_data]
            if len(pixel_values) > 0:
                image_token_id = pixel_values[0].meta['image_token_id']
                image_mask = input_ids == image_token_id
                pixel_values = torch.cat([data.data for data in pixel_values])
            else:
                pixel_values = None
                image_mask = None

        if self.is_mono and pixel_values is not None:
            vision_embedding_indexing = torch.arange(input_ids.shape[1], device=input_ids.device)
            vision_embedding_indexing = vision_embedding_indexing[image_mask[0]]

        # get inputs from context
        if vision_embeddings is not None and len(vision_embeddings) > 0:
            vision_embedding_indexing = context.input_embedding_indexing
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds[:, vision_embedding_indexing, :] = vision_embeddings.to(inputs_embeds)

        has_model_metas = context.model_metas is not None and context.model_metas[0] is not None
        if context.is_decoding:
            if has_model_metas:
                # NOTE, zhouxinyu, we need to consider the increasing batch in the decoding stage
                # currently implementation will keep the batch size same as the prefill stage

                # model meta from the previous step, therefore +1 for the current decoding step
                new_kv_seqlens = [(meta['new_seqlen'] + 1) for meta in context.model_metas]

                # update model metas for the next step
                context.model_metas = [dict(new_seqlen=seqlen) for seqlen in new_kv_seqlens]

                # update position ids, attn_metadata
                new_kv_seqlens = torch.tensor(new_kv_seqlens, device=input_ids.device, dtype=torch.long)
                position_ids = new_kv_seqlens - 1
                attn_metadata.kv_seqlens = new_kv_seqlens
                attn_metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(new_kv_seqlens, dim=0, dtype=torch.int32), (1, 0))
        else:
            # in the case of long context, messages may be split into multiple segments and perform prefill sequentially
            # 1. this will only be done when flash_mode is on
            # 2. if it is a text segment, we update model metas before forward
            # 3. if it is an image segment, we update model metas later, after vision forward / compression
            is_text_segment = (inputs_embeds is None) and (pixel_values is None)

            if self.flash_mode and is_text_segment:
                crt_ctx = self.ctx_mgr.current_context()
                seq_lengths = crt_ctx.q_seqlens

                if has_model_metas:
                    prev_lens = [meta.get('new_seqlen', 0) for meta in context.model_metas]

                    for idx, meta in enumerate(context.model_metas):
                        meta.update({'new_seqlen': prev_lens[idx] + seq_lengths[idx].item()})

                    # update position ids, attn_metadata
                    prev_lens = torch.tensor(prev_lens, device=input_ids.device, dtype=torch.long)
                    ranges = torch.arange(0, input_ids.shape[1], device=input_ids.device)
                    position_ids = prev_lens[:, None] + ranges[None, :]
                    position_ids = position_ids
                    attn_metadata.kv_seqlens = prev_lens + seq_lengths
                else:
                    # init model metas
                    context.model_metas = [{'new_seqlen': seqlen} for seqlen in seq_lengths.tolist()]

        if self.is_mono and vision_embedding_indexing is not None:
            all_indices = torch.arange(input_ids.shape[1]).to(input_ids)
            text_embedding_indexing = all_indices[~torch.isin(all_indices, vision_embedding_indexing)]
            if vision_embedding_indexing.numel() == 0:
                vision_embedding_indexing = None
            if text_embedding_indexing.numel() == 0:
                text_embedding_indexing = None
            return dict(input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        attn_metadata=attn_metadata,
                        pixel_values=pixel_values,
                        image_mask=image_mask,
                        inputs_embeds=inputs_embeds,
                        vision_embedding_indexing=vision_embedding_indexing,
                        text_embedding_indexing=text_embedding_indexing,
                        image_token_id=image_token_id,
                        context=context)
        else:
            return dict(input_ids=input_ids,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        attn_metadata=attn_metadata,
                        pixel_values=pixel_values,
                        image_mask=image_mask,
                        inputs_embeds=inputs_embeds,
                        image_token_id=image_token_id,
                        context=context)

    def load_lora_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], adapter_id: int):
        """Load lora weights."""

        if hasattr(self.language_model, 'load_lora_weights'):
            return self.language_model.load_lora_weights(weights, adapter_id)
        else:
            from lmdeploy.pytorch.adapter.adapter import load_lora_weights

            return load_lora_weights(weights, adapter_id)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        lang_prefix = 'language_model.'
        lang_prefix_length = len(lang_prefix)
        new_weights = dict()
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.startswith(lang_prefix):
                new_key = name[lang_prefix_length:]
                new_weights[new_key] = loaded_weight
                continue

            if 'qkv' in name:
                param = params_dict[name]
                q, k, v = param.weight_spliter(loaded_weight)
                load_weight(param, q, shard_id='q')
                load_weight(param, k, shard_id='k')
                load_weight(param, v, shard_id='v')
            else:
                param = params_dict[name]
                load_weight(param, loaded_weight)

        self.language_model.load_weights(new_weights.items())

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor


class InternVLInputProcessor(BaseModelInputProcessor):
    """Internvl input processor."""

    def __init__(self, config: PretrainedConfig, dtype) -> None:
        self.config = config
        self.dtype = dtype

        vision_config = config.vision_config
        self.image_size = vision_config.image_size
        self.patch_size = vision_config.patch_size
        self.num_patches = (self.image_size // self.patch_size)**2
        self.num_positions = self.num_patches + 1
        self.vision_token_num = self.num_patches // 4

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
            if isinstance(num_pad, torch.Tensor):
                num_pad = num_pad.item()

            mm_data = MultiModalTensor(data=pixel_values,
                                       start=offset,
                                       end=offset + num_pad,
                                       meta=dict(image_token_id=image_token_id))
            input_imgs.append(mm_data)

        result = PreprocessInputResult(
            input_ids=input_ids,
            input_multimodals=dict(image=input_imgs),
        )
        return result
