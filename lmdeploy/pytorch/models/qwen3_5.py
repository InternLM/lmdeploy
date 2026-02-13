# Copyright (c) OpenMMLab. All rights reserved.

from functools import lru_cache
from typing import Any, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

import lmdeploy.pytorch.nn.gated_delta as gated_delta_util
from lmdeploy.pytorch.distributed import get_tp_world_rank
from lmdeploy.pytorch.engine.input_process import BaseModelInputProcessor
from lmdeploy.pytorch.model_inputs import StepContext, StepContextManager
from lmdeploy.pytorch.nn import ApplyRotaryEmb, Attention, LayerNorm, RMSNorm, SiluAndMul
from lmdeploy.pytorch.nn.gated_delta import CausalConv1d, GatedDelta, GatedDeltaMeta, build_rmsnorm_gated
from lmdeploy.pytorch.nn.linear import (build_colwise_linear, build_merged_colwise_linear, build_o_proj, build_qkv_proj,
                                        build_rowwise_linear)
from lmdeploy.pytorch.nn.rotary_embedding import get_rope_parameters
from lmdeploy.pytorch.weight_loader.model_weight_loader import default_weight_loader, load_weight

from .qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding as Qwen3_5VisionRotaryEmbedding
from .qwen2_5_vl import Qwen2_5_VLInputProcessor as Qwen3_5InputProcessor
from .qwen2_5_vl import Qwen2_5_VLVisionAttention as Qwen3_5VisionAttention
from .utils.cudagraph import CudaGraphMeta, CudaGraphMixin
from .utils.model import DeployModelMixin, vlm_model


class Qwen3_5VisionPatchEmbed(nn.Module):

    def __init__(self, config, dtype: torch.dtype | None = None, device: torch.device | None = None) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = (self.temporal_patch_size, self.patch_size, self.patch_size)
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


class Qwen3_5VisionMLP(nn.Module):
    """Vision mlp."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
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


class Qwen3_5VisionBlock(nn.Module):
    """Vision block."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = LayerNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device)
        self.norm2 = LayerNorm(config.hidden_size, eps=1e-6, dtype=dtype, device=device)

        self.attn = Qwen3_5VisionAttention(config, dtype=dtype, device=device)

        self.mlp = Qwen3_5VisionMLP(config, dtype=dtype, device=device)

    def forward(self,
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor | None = None) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3_5VisionPatchMerger(nn.Module):

    def __init__(self,
                 config: PretrainedConfig,
                 use_postshuffle_norm=False,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None) -> None:
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
class Qwen3_5VisionModel(nn.Module):
    """qwen3.5 vision model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = Qwen3_5VisionPatchEmbed(config=config, dtype=dtype, device=device)

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size, dtype=dtype, device=device)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2, device=device)

        self.blocks = nn.ModuleList(
            [Qwen3_5VisionBlock(config, layer_idx, dtype=dtype, device=device) for layer_idx in range(config.depth)])
        self.merger = Qwen3_5VisionPatchMerger(config=config, use_postshuffle_norm=False, dtype=dtype, device=device)

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

        for _, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        hidden_states = self.merger(hidden_states)

        return hidden_states


class Qwen3_5MLP(nn.Module):
    """mlp."""

    def __init__(self,
                 config: PretrainedConfig,
                 intermediate_size: int | None = None,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 is_tp: bool = True,
                 all_reduce: bool = True):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        if intermediate_size is None:
            intermediate_size = config.intermediate_size
        # gate up
        self.gate_up_proj = build_merged_colwise_linear(
            config.hidden_size,
            [intermediate_size, intermediate_size],
            bias=False,
            dtype=dtype,
            device=device,
            quant_config=quantization_config,
            is_tp=is_tp,
        )

        # silu and mul
        self.act_fn = SiluAndMul(inplace=True)

        # down
        self.down_proj = build_rowwise_linear(intermediate_size,
                                              config.hidden_size,
                                              bias=False,
                                              quant_config=quantization_config,
                                              dtype=dtype,
                                              device=device,
                                              is_tp=is_tp,
                                              all_reduce=all_reduce)

    def forward(self, x):
        """forward."""
        gate_up = self.gate_up_proj(x)
        act = self.act_fn(gate_up)
        return self.down_proj(act)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated deltanet."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.kv_ratio = self.num_v_heads // self.num_k_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = CausalConv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            split=[self.key_dim, self.key_dim, self.value_dim],
            bias=False,
            groups=self.conv_dim,
            dtype=dtype,
            device=device,
        )

        # projection of the input hidden states
        projection_size_qkv = self.key_dim * 2 + self.value_dim
        self.in_proj_qkv = build_colwise_linear(self.hidden_size,
                                                projection_size_qkv,
                                                bias=False,
                                                dtype=dtype,
                                                device=device,
                                                is_tp=True)
        self.in_proj_qkv.weight.weight_loader = self.weight_loader_qkv
        self.in_proj_zba = build_merged_colwise_linear(
            self.hidden_size,
            [self.value_dim, self.num_v_heads, self.num_v_heads],
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=True,
            out_names=['z', 'b', 'a'],
        )

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.make_params(self.num_v_heads, device=device)
        self.A_log_exp = None

        self.norm = build_rmsnorm_gated(self.head_v_dim,
                                        eps=self.layer_norm_epsilon,
                                        activation=self.activation,
                                        dtype=dtype,
                                        device=device)
        self.out_proj = build_o_proj(self.value_dim,
                                     self.hidden_size,
                                     bias=False,
                                     dtype=dtype,
                                     device=device,
                                     is_tp=True)

        self.gated_delta = GatedDelta()

    def get_A_log_exp(self):
        if self.A_log_exp is None:
            self.A_log_exp = -self.A_log.float().exp()

        return self.A_log_exp

    def make_params(self, num_v_heads: int, device: torch.device | None):
        tp, _ = get_tp_world_rank()
        num_v_heads = num_v_heads // tp
        A = torch.empty(num_v_heads, device=device)
        dt_bias = torch.empty(num_v_heads, device=device)

        self.register_parameter('A_log', nn.Parameter(torch.log(A)))
        self.register_parameter('dt_bias', nn.Parameter(dt_bias))
        self.A_log.weight_loader = self.weight_loader_a_dt
        self.dt_bias.weight_loader = self.weight_loader_a_dt

    def weight_loader_qkv(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader for qkv projection."""
        tp, rank = get_tp_world_rank()
        q, k, v = loaded_weight.split([self.key_dim, self.key_dim, self.value_dim], dim=0)
        q = q.chunk(tp, dim=0)[rank]
        k = k.chunk(tp, dim=0)[rank]
        v = v.chunk(tp, dim=0)[rank]
        loaded_weight = torch.cat([q, k, v], dim=0)
        default_weight_loader(param, loaded_weight)

    def weight_loader_a_dt(self, param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        """Weight loader."""
        tp, rank = get_tp_world_rank()
        loaded_weight = loaded_weight.chunk(tp, dim=0)[rank]
        default_weight_loader(param, loaded_weight)

    def fix_zba_ordering(self, mixed_zba: torch.Tensor):
        """Derives `query`, `key` and `value` tensors from `mixed_qkv` and
        `mixed_zba`."""

        # zba
        split_arg_list_zba = [self.head_v_dim * self.kv_ratio, self.kv_ratio, self.kv_ratio]
        num_heads = mixed_zba.size(-1) // sum(split_arg_list_zba)
        split_arg_list_zba = [num_heads * x for x in split_arg_list_zba]
        z, b, a = torch.split(mixed_zba, split_arg_list_zba, dim=-1)
        # [..., ng, np/ng * hn] -> [..., np, hn]
        z = z.unflatten(-1, (-1, self.head_v_dim))
        return z, b, a

    def _load_state(self, past_key_value: Tuple[torch.Tensor, torch.Tensor], gated_delta_meta: GatedDeltaMeta):
        """Load states from cache."""
        return gated_delta_util.load_state(past_key_value=past_key_value, gated_delta_meta=gated_delta_meta)

    def _store_state(self, conv_state: torch.Tensor, recurrent_state: torch.Tensor,
                     past_key_value: Tuple[torch.Tensor, torch.Tensor], gated_delta_meta: GatedDeltaMeta):
        """Store states to cache."""
        return gated_delta_util.store_state(conv_state=conv_state,
                                            recurrent_state=recurrent_state,
                                            past_key_value=past_key_value,
                                            gated_delta_meta=gated_delta_meta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        gated_delta_meta: GatedDeltaMeta,
    ):
        """forward."""

        # load states
        conv_state, recurrent_state = self._load_state(past_key_value, gated_delta_meta)

        # inputs proj
        projected_states_qkv = self.in_proj_qkv(hidden_states)
        projected_states_zba = self.in_proj_zba(hidden_states)
        z, b, a = self.fix_zba_ordering(projected_states_zba)

        mixed_qkv = projected_states_qkv
        mixed_qkv, conv_state = self.conv1d(mixed_qkv, conv_state, gated_delta_meta=gated_delta_meta)

        tp = (self.key_dim * 2 + self.value_dim) // mixed_qkv.size(-1)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // tp,
                self.key_dim // tp,
                self.value_dim // tp,
            ],
            dim=-1,
        )
        query = query.unflatten(-1, (-1, self.head_k_dim))
        key = key.unflatten(-1, (-1, self.head_k_dim))
        value = value.unflatten(-1, (-1, self.head_v_dim))

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = self.get_A_log_exp() * F.softplus(a.float() + self.dt_bias)
        if self.kv_ratio > 1:
            query = query.repeat_interleave(self.kv_ratio, dim=-2)
            key = key.repeat_interleave(self.kv_ratio, dim=-2)

        core_attn_out, recurrent_state = self.gated_delta(
            query,
            key,
            value,
            g=g,
            beta=beta,
            recurrent_state=recurrent_state,
            gated_delta_meta=gated_delta_meta,
        )

        # store states
        self._store_state(conv_state, recurrent_state, past_key_value, gated_delta_meta)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        return output


class Qwen3_5Attention(nn.Module):
    """Rewrite module of Qwen3MoeAttention."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        quantization_config = getattr(config, 'quantization_config', None)
        num_heads = config.num_attention_heads
        num_key_value_heads = config.num_key_value_heads
        hidden_size = config.hidden_size
        head_dim = getattr(config, 'head_dim', hidden_size // num_heads)
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        num_replicate_kv_heads = getattr(config, 'num_replicate_key_value_heads', 1)

        # packed qkv
        # Qwen3 uses 'config.attention_bias = False' for q/k/o projections
        self.qkv_proj = build_qkv_proj(
            hidden_size,
            num_q_heads=num_heads * 2,
            num_kv_heads=num_key_value_heads,
            head_size=head_dim,
            bias=config.attention_bias,
            quant_config=quantization_config,
            num_replicate_kv_heads=num_replicate_kv_heads,
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
        )

        # o_proj
        self.o_proj = build_o_proj(num_heads * head_dim,
                                   hidden_size,
                                   bias=config.attention_bias,
                                   quant_config=quantization_config,
                                   dtype=dtype,
                                   device=device,
                                   is_tp=True)

        # q, k norm
        self.q_norm = RMSNorm(head_dim,
                              config.rms_norm_eps,
                              quant_config=quantization_config,
                              dtype=dtype,
                              device=device)
        self.k_norm = RMSNorm(head_dim,
                              config.rms_norm_eps,
                              quant_config=quantization_config,
                              dtype=dtype,
                              device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: Any,
    ):
        """Rewrite of LlamaAttention.forward."""
        # qkv proj
        qkv_states = self.qkv_proj(hidden_states)
        # (-1, heads, head_dim)
        qkv_states = qkv_states.flatten(0, -2)
        query_states, key_states, value_states = self.qkv_proj.split_qkv(qkv_states)
        query_states, gate = query_states.view(*query_states.shape[:-2], -1, 2 * self.head_dim).chunk(2, dim=-1)

        # apply q, k norm
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

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
            k_scales_zeros=None if len(past_key_value) == 2 else past_key_value[2],
            v_scales_zeros=None if len(past_key_value) == 2 else past_key_value[3],
            inplace=True,
        )
        attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
        gate = gate.reshape(*hidden_states.shape[:-1], -1)
        attn_output = attn_output * gate.sigmoid()

        # o proj
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3_5DecoderLayer(nn.Module):
    """Decoder layer."""

    def __init__(self,
                 config: PretrainedConfig,
                 layer_idx: int,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.layer_idx = layer_idx
        quantization_config = getattr(config, 'quantization_config', None)

        # build attention layer
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == 'linear_attention':
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx, dtype=dtype, device=device)
        elif self.layer_type == 'full_attention':
            self.self_attn = Qwen3_5Attention(config, layer_idx, dtype=dtype, device=device)

        # build MLP
        self.mlp = Qwen3_5MLP(config, intermediate_size=config.intermediate_size, dtype=dtype, device=device)

        # build input layer norm
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       config.rms_norm_eps,
                                       quant_config=quantization_config,
                                       dtype=dtype,
                                       device=device)

        # build attention layer norm
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Tuple[torch.FloatTensor, torch.FloatTensor],
        past_key_value: List[torch.FloatTensor],
        residual: torch.Tensor | None,
        attn_metadata: Any,
        gated_delta_meta: GatedDeltaMeta,
    ):

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        if self.layer_type == 'linear_attention':
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                gated_delta_meta=gated_delta_meta,
            )
        elif self.layer_type == 'full_attention':
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_value,
                attn_metadata=attn_metadata,
            )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        outputs = (hidden_states, residual)
        return outputs


class Qwen3_5TextRotaryEmbedding(nn.Module):
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
        if self.rope_type != 'default':
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        else:
            self.rope_init_fn = self.compute_default_rope_parameters

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        self.mrope_section = config.rope_scaling.get('mrope_section', [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: PretrainedConfig | None = None,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple['torch.Tensor', float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        rope_parameters = get_rope_parameters(config)
        base = rope_parameters['rope_theta']
        partial_rotary_factor = rope_parameters.get('partial_rotary_factor', 1.0)
        head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base**(torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, attention_factor

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


class Qwen3_5TextModel(nn.Module):
    """qwen3.5 text model."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size,
                                         config.hidden_size,
                                         self.padding_idx,
                                         dtype=dtype,
                                         device=device)

        # build all decode layers
        # TODO: use full config.num_hidden_layers
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(config, layer_idx, dtype=dtype, device=device)
            for layer_idx in range(self.config.num_hidden_layers)
        ])

        # build norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype=dtype, device=device)

        # build rotary embedding
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config, device=device)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any,
        state_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
    ):
        """Rewrite of LlamaModel.forward."""

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

        # make seq_idx
        gated_delta_meta = GatedDeltaMeta(hidden_states.size(1), self.config.linear_conv_kernel_dim, state_ids,
                                          attn_metadata)

        # decoding
        residual = None
        for idx, decoder_layer in enumerate(self.layers):
            hidden_states, residual = decoder_layer(
                hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                past_key_value=past_key_values[idx],
                residual=residual,
                attn_metadata=attn_metadata,
                gated_delta_meta=gated_delta_meta,
            )

        # norm
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.embed_tokens


class Qwen3_5Model(nn.Module):

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype | None = None, device: torch.device | None = None):
        super().__init__()

        self.visual = Qwen3_5VisionModel(config.vision_config, dtype=dtype, device=device)
        self.language_model = Qwen3_5TextModel(config.text_config, dtype=dtype, device=device)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: List[List[torch.Tensor]],
        attn_metadata: Any,
        state_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vis_cu_seqlens: torch.Tensor | None = None,
        vis_pos_emb: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        pos_embeds: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
    ):
        """Model forward, return logits."""

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                dtype = inputs_embeds.dtype
                pixel_values = pixel_values.to(dtype)
                vis_pos_emb = (vis_pos_emb[0].to(dtype), vis_pos_emb[1].to(dtype))

                # get image embeds and deepstack visual embeds
                image_embeds = self.visual(pixel_values,
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

        hidden_states = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            state_ids=state_ids,
            inputs_embeds=inputs_embeds,
            mrope_position_ids=mrope_position_ids,
        )
        return hidden_states

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.language_model.get_input_embeddings()


class Qwen3_5ForConditionalGeneration(nn.Module, DeployModelMixin, CudaGraphMixin):
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
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr

        # build preprocessor
        self.input_processor = Qwen3_5InputProcessor(self.config)

        # build model
        self.model = Qwen3_5Model(config, dtype=dtype, device=device)
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
        attn_metadata: Any,
        state_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        mrope_position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        vis_cu_seqlens: torch.Tensor | None = None,
        vis_pos_emb: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        pos_embeds: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        **kwargs,
    ):
        """Model forward, return logits."""
        hidden_states = self.model(
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
            image_mask=image_mask,
            pos_embeds=pos_embeds,
            grid_thw=grid_thw,
        )
        return hidden_states

    def get_logits(self, hidden_states: torch.Tensor):
        """Compute logits of the model output."""
        return self.lm_head(hidden_states)

    def get_input_embeddings(self):
        """Get input embeddings."""
        return self.model.get_input_embeddings()

    def prepare_inputs_for_generation(
        self,
        past_key_values: List[List[torch.Tensor]],
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

        # vlm inputs
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
                vis_pos_emb = self.model.visual.rot_pos_emb(grid_thw)
                pos_embeds = self.model.visual.fast_pos_embed_interpolate(grid_thw)
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
            past_key_values=new_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            state_ids=context.state_offsets,
            # vl inputs
            mrope_position_ids=mrope_position_ids,
            pixel_values=pixel_values,
            vis_cu_seqlens=vis_cu_seqlens,
            vis_pos_emb=vis_pos_emb,
            image_mask=image_mask,
            grid_thw=grid_thw,
            pos_embeds=pos_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights."""

        def __skip_layers(name):
            """We might change the number of layers so we can debug the model
            with less gpus."""
            import re
            if '.layers.' not in name:
                return False
            matches = re.findall(r'\.layers\.(\d+)\.', name)
            layer_id = int(matches[0])
            return layer_id >= self.config.text_config.num_hidden_layers

        # modify from vllm
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ('.qkv_proj', '.q_proj', 'q'),
            ('.qkv_proj', '.k_proj', 'k'),
            ('.qkv_proj', '.v_proj', 'v'),
            ('.gate_up_proj', '.gate_proj', 0),
            ('.gate_up_proj', '.up_proj', 1),
            ('.in_proj_zba', '.in_proj_z', 'z'),
            ('.in_proj_zba', '.in_proj_b', 'b'),
            ('.in_proj_zba', '.in_proj_a', 'a'),
        ]

        rms_norm_keys = ['model.norm', '.input_layernorm', '.post_attention_layernorm', '.q_norm', '.k_norm']

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:

            if __skip_layers(name):
                continue

            if 'mtp.' in name:
                continue
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
                    # vl attention
                    param = params_dict[name]
                    q, k, v = param.weight_spliter(loaded_weight)
                    load_weight(param, q, shard_id='q')
                    load_weight(param, k, shard_id='k')
                    load_weight(param, v, shard_id='v')
                else:
                    for rms_norm_key in rms_norm_keys:
                        if rms_norm_key in name and 'weight' in name:
                            loaded_weight = loaded_weight + 1
                            break
                    param = params_dict[name]
                    load_weight(param, loaded_weight)

    def make_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Make cudagraph buffers from forward inputs."""

        max_batchs = graph_meta.max_batchs
        device = graph_meta.device
        max_tokens = graph_meta.max_tokens

        input_buffers = super().make_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        mrope_position_ids = kwargs.get('mrope_position_ids', None)
        if mrope_position_ids is not None:
            input_buffers['mrope_position_ids'] = mrope_position_ids.new_zeros(3, max_tokens)

        state_ids = torch.full((max_batchs, ), -1, dtype=torch.long, device=device)
        input_buffers['state_ids'] = state_ids
        return input_buffers

    def fill_buffers_cudagraph(self, graph_meta: CudaGraphMeta, **kwargs):
        """Fill cudagraph buffers from forward inputs."""
        input_buffers = graph_meta.input_buffers

        new_inputs = super().fill_buffers_cudagraph(graph_meta=graph_meta, **kwargs)
        state_ids = kwargs['state_ids']
        input_buffers['state_ids'].fill_(-1)
        input_buffers['state_ids'][:state_ids.size(0)].copy_(state_ids)
        new_inputs['state_ids'] = input_buffers['state_ids']

        input_ids = kwargs.get('input_ids')
        num_tokens = input_ids.size(-1)
        new_batch_size = graph_meta.max_batchs

        is_decoding = graph_meta.is_decoding
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

    def update_model_metas(self, past_key_values: List[List[torch.Tensor]], inputs_embeds: torch.Tensor | None,
                           context: StepContext):
        """Update model meta."""
        if context.is_decoding:
            return self._update_model_meta_decoding(context)
        else:
            return self._update_model_meta_prefilling(context)

    def get_input_processor(self) -> BaseModelInputProcessor:
        """Get input processor."""
        return self.input_processor
