# Copyright (c) OpenMMLab. All rights reserved.
#
# This file contains code adapted from moonshotai/Kimi-K2.6
# ``modeling_kimi_k25.py``. The upstream implementation is based in part on
# LLaVA and DeepSeek-V3.
#
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc.
#
# Code derived from LLaVA and DeepSeek-V3 is licensed under the Apache License,
# Version 2.0. Other upstream portions are licensed under the MIT License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .utils.model import vlm_model

_MISSING = object()


def _config_value(config: Any, *names: str, default: Any = _MISSING) -> Any:
    """Read the first available config attribute or mapping key."""
    for name in names:
        if isinstance(config, dict) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    if default is not _MISSING:
        return default
    joined = '`, `'.join(names)
    raise AttributeError(
        f'Kimi-K2.6 vision config must define one of `{joined}`.')


def _factory_kwargs(dtype: torch.dtype | None,
                    device: torch.device | str | None) -> dict[str, Any]:
    """Build kwargs accepted by PyTorch parameterized modules."""
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs['dtype'] = dtype
    if device is not None:
        kwargs['device'] = device
    return kwargs


def _normalize_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, Sequence) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise ValueError(
            f'`{name}` must be an int or a length-2 sequence, got {value!r}.')
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f'`{name}` values must be positive, got {pair}.')
    return pair


def _grid_shapes(grid_thws: torch.Tensor) -> list[tuple[int, int, int]]:
    """Validate and materialize packed vision grid shapes."""
    if not isinstance(grid_thws, torch.Tensor):
        raise TypeError(
            f'`grid_thws` must be a torch.Tensor, got {type(grid_thws)}.')
    if grid_thws.ndim != 2 or grid_thws.shape[1] != 3:
        raise ValueError(
            f'`grid_thws` must have shape [N, 3], got {tuple(grid_thws.shape)}.'
        )
    if grid_thws.shape[0] == 0:
        raise ValueError('`grid_thws` must contain at least one image grid.')

    shapes = [
        tuple(int(value) for value in shape) for shape in grid_thws.tolist()
    ]
    if any(t <= 0 or h <= 0 or w <= 0 for t, h, w in shapes):
        raise ValueError(
            f'All Kimi vision grid dimensions must be positive, got {shapes}.')
    return shapes


def get_1d_sincos_pos_embed_from_grid(embed_dim: int,
                                      pos: np.ndarray) -> np.ndarray:
    """Create the fixed temporal sinusoidal embedding used by MoonViT."""
    if embed_dim % 2 != 0:
        raise ValueError(f'Embedding dimension must be even, got {embed_dim}.')
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)
    phase = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1)


def get_1d_sincos_pos_embed(embed_dim: int, size: int) -> np.ndarray:
    """Create a fixed 1D sinusoidal position table."""
    grid = np.arange(size, dtype=np.float32)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, grid)


def interpolate_2d_pos_embed(weight: torch.Tensor,
                             shape: tuple[int, int],
                             mode: str = 'bicubic') -> torch.Tensor:
    """Interpolate ``[H, W, C]`` learned positions to a new grid."""
    return (F.interpolate(
        weight.permute(2, 0, 1).unsqueeze(0),
        size=shape,
        mode=mode,
    ).squeeze(0).permute(1, 2, 0).flatten(end_dim=1))


class Learnable2DInterpPosEmbDividedFixed(nn.Module):
    """Learned 2D positions plus a fixed temporal position embedding."""

    def __init__(
        self,
        height: int,
        width: int,
        num_frames: int,
        dim: int,
        interpolation_mode: str = 'bicubic',
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if height <= 0 or width <= 0 or num_frames <= 0:
            raise ValueError(
                f'Position table sizes must be positive, got height={height}, width={width}, frames={num_frames}.'
            )
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.dim = dim
        self.interpolation_mode = interpolation_mode

        factory_kwargs = _factory_kwargs(dtype, device)
        self.weight = nn.Parameter(
            torch.empty(height, width, dim, **factory_kwargs))
        time_weight = torch.from_numpy(get_1d_sincos_pos_embed(
            dim, num_frames)).unsqueeze(1)
        time_dtype = dtype if dtype is not None else torch.float32
        self.register_buffer(
            'time_weight',
            time_weight.to(device=device, dtype=time_dtype),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, x: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        shapes = _grid_shapes(grid_thws)
        expected_tokens = sum(t * h * w for t, h, w in shapes)
        if x.ndim != 2 or x.shape[0] != expected_tokens or x.shape[
                1] != self.dim:
            raise ValueError(
                f'Position input must have shape [{expected_tokens}, {self.dim}], got {tuple(x.shape)}.'
            )

        pos_embs = []
        for t, h, w in shapes:
            if t > self.num_frames:
                raise ValueError(
                    f'Grid temporal size {t} exceeds position table size {self.num_frames}.'
                )
            if (h, w) == self.weight.shape[:-1]:
                pos_emb_2d = self.weight.flatten(end_dim=1)
            else:
                pos_emb_2d = interpolate_2d_pos_embed(
                    self.weight,
                    shape=(h, w),
                    mode=self.interpolation_mode,
                )

            if t == 1:
                pos_emb_3d = pos_emb_2d
            else:
                pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat(
                    t, 1, 1) + self.time_weight[:t]
            pos_embs.append(pos_emb_3d.reshape(-1, self.dim))

        return x + torch.cat(pos_embs, dim=0).to(x)


# Preserve the upstream spelling for reviewers comparing against the remote
# implementation. It is an alias, so it does not change checkpoint names.
Learnable2DInterpPosEmbDivided_fixed = Learnable2DInterpPosEmbDividedFixed


class MoonVision3dPatchEmbed(nn.Module):
    """Embed the processor's already-extracted image patches."""

    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: int | Sequence[int] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
        pos_emb_time: int = 4,
        pos_emb_type: str = 'divided_fixed',
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.patch_size = _normalize_pair(patch_size, 'patch_size')
        factory_kwargs = _factory_kwargs(dtype, device)
        self.proj = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            **factory_kwargs,
        )

        if pos_emb_type != 'divided_fixed':
            raise NotImplementedError(
                f'Unsupported Kimi position embedding type: {pos_emb_type}.')
        self.pos_emb = Learnable2DInterpPosEmbDividedFixed(
            height=pos_emb_height,
            width=pos_emb_width,
            num_frames=pos_emb_time,
            dim=out_dim,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        shapes = _grid_shapes(grid_thws)
        expected_patches = sum(t * h * w for t, h, w in shapes)
        expected_shape = (expected_patches, self.proj.in_channels,
                          *self.patch_size)
        if tuple(x.shape) != expected_shape:
            raise ValueError(
                f'Packed Kimi patches must have shape {expected_shape}, got {tuple(x.shape)}.'
            )

        target_dtype = self.proj.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(dtype=target_dtype)
        x = self.proj(x).flatten(1)
        return self.pos_emb(x, grid_thws)


class Rope2DPosEmbRepeated(nn.Module):
    """MoonViT 2D RoPE shared by every attention head and temporal frame."""

    def __init__(
        self,
        dim: int,
        max_height: int = 512,
        max_width: int = 512,
        theta_base: float = 10000.0,
    ):
        super().__init__()
        if dim % 4 != 0:
            raise ValueError(
                f'2D RoPE head dimension must be divisible by 4, got {dim}.')
        if max_height <= 0 or max_width <= 0:
            raise ValueError(
                f'2D RoPE limits must be positive, got {(max_height, max_width)}.'
            )
        self.dim = dim
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.register_buffer('freqs_cis', None, persistent=False)

    def extra_repr(self):
        return (f'dim={self.dim}, max_height={self.max_height}, '
                f'max_width={self.max_width}, theta_base={self.theta_base}')

    def _precompute_freqs_cis(self, device: torch.device) -> torch.Tensor:
        num_positions = self.max_height * self.max_width
        flat_pos = torch.arange(num_positions,
                                dtype=torch.float32,
                                device=device)
        x_pos = flat_pos % self.max_width
        y_pos = torch.div(flat_pos, self.max_width, rounding_mode='floor')
        dim_range = torch.arange(0,
                                 self.dim,
                                 4,
                                 dtype=torch.float32,
                                 device=device)[:self.dim // 4]
        freqs = 1.0 / self.theta_base**(dim_range / self.dim)
        x_freqs = torch.outer(x_pos, freqs)
        y_freqs = torch.outer(y_pos, freqs)
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
        return torch.stack((x_cis, y_cis),
                           dim=-1).reshape(self.max_height, self.max_width,
                                           self.dim // 2)

    def get_freqs_cis(self, grid_thws: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
        shapes = _grid_shapes(grid_thws)
        if any(h > self.max_height or w > self.max_width
               for _, h, w in shapes):
            raise ValueError(
                f'Vision grid {shapes} exceeds 2D RoPE limits {(self.max_height, self.max_width)}.'
            )
        if self.freqs_cis is None or self.freqs_cis.device != device:
            self.freqs_cis = self._precompute_freqs_cis(device)

        return torch.cat(
            [
                self.freqs_cis[:h, :w].reshape(-1, self.dim // 2).repeat(t, 1)
                for t, h, w in shapes
            ],
            dim=0,
        )


def _validate_rope_inputs(x: torch.Tensor, freqs_cis: torch.Tensor):
    if x.ndim != freqs_cis.ndim + 1:
        raise ValueError(
            f'RoPE tensor ranks are incompatible: {tuple(x.shape)} and {tuple(freqs_cis.shape)}.'
        )
    if x.shape[:-2] != freqs_cis.shape[:-1]:
        raise ValueError(
            f'RoPE leading shapes are incompatible: {tuple(x.shape)} and {tuple(freqs_cis.shape)}.'
        )
    if x.shape[-1] != 2 * freqs_cis.shape[-1]:
        raise ValueError(
            f'RoPE head dimension is incompatible: {tuple(x.shape)} and {tuple(freqs_cis.shape)}.'
        )
    if freqs_cis.dtype != torch.complex64:
        raise TypeError(
            f'MoonViT RoPE frequencies must be complex64, got {freqs_cis.dtype}.'
        )


def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the official adjacent-pair complex 2D rotary embedding."""
    _validate_rope_inputs(xq, freqs_cis)
    _validate_rope_inputs(xk, freqs_cis)

    freqs_cis = freqs_cis.unsqueeze(-2)
    xq_complex = torch.view_as_complex(xq.float().reshape(
        *xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(
        *xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(-2)
    return xq_out.to(xq), xk_out.to(xk)


class MLP2(nn.Module):
    """Two-layer MoonViT feed-forward network."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: nn.Module,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if len(dims) != 3:
            raise ValueError(
                f'MoonViT MLP requires three dimensions, got {dims}.')
        factory_kwargs = _factory_kwargs(dtype, device)
        self.fc0 = nn.Linear(dims[0], dims[1], bias=bias, **factory_kwargs)
        self.fc1 = nn.Linear(dims[1], dims[2], bias=bias, **factory_kwargs)
        self.activation = activation

        for layer in (self.fc0, self.fc1):
            nn.init.trunc_normal_(layer.weight,
                                  std=math.sqrt(2 / layer.in_features))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc1(self.activation(self.fc0(x)))


def packed_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor | Sequence[int],
) -> torch.Tensor:
    """Run non-causal SDPA independently for every packed image clip.

    MoonViT is replicated on every tensor-parallel rank. Using PyTorch SDPA here intentionally avoids LMDeploy's TP-
    aware attention head partitioning. On CUDA, PyTorch can dispatch BF16 inputs to its fused SDPA kernels.
    """
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            f'Packed Q/K/V shapes must match, got {query.shape}, {key.shape}, and {value.shape}.'
        )
    if query.ndim != 3:
        raise ValueError(
            f'Packed Q/K/V must have shape [tokens, heads, dim], got {tuple(query.shape)}.'
        )
    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                f'`cu_seqlens` must contain packed boundaries, got {tuple(cu_seqlens.shape)}.'
            )
        boundaries = [int(value) for value in cu_seqlens.tolist()]
    else:
        boundaries = [int(value) for value in cu_seqlens]
        if len(boundaries) < 2:
            raise ValueError(
                f'`cu_seqlens` must contain packed boundaries, got {boundaries}.'
            )
    if boundaries[0] != 0 or boundaries[-1] != query.shape[0]:
        raise ValueError(
            f'Packed boundaries {boundaries} do not cover {query.shape[0]} query tokens.'
        )
    if any(end <= start for start, end in zip(boundaries, boundaries[1:])):
        raise ValueError(
            f'Packed sequence lengths must be positive, got boundaries {boundaries}.'
        )

    outputs = []
    for start, end in zip(boundaries, boundaries[1:]):
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = key[start:end].transpose(0, 1).unsqueeze(0)
        v = value[start:end].transpose(0, 1).unsqueeze(0)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0).flatten(start_dim=-2)


class MoonViTEncoderLayer(nn.Module):
    """A replicated MoonViT transformer block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        activation: nn.Module | None = None,
        attn_bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'Vision hidden size {hidden_dim} must be divisible by {num_heads} heads.'
            )
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = hidden_dim // num_heads
        factory_kwargs = _factory_kwargs(dtype, device)

        self.norm0 = nn.LayerNorm(hidden_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(hidden_dim, **factory_kwargs)
        self.mlp = MLP2(
            [hidden_dim, mlp_dim, hidden_dim],
            activation or nn.GELU(approximate='tanh'),
            dtype=dtype,
            device=device,
        )
        self.wqkv = nn.Linear(hidden_dim,
                              hidden_dim * 3,
                              bias=attn_bias,
                              **factory_kwargs)
        self.wo = nn.Linear(hidden_dim,
                            hidden_dim,
                            bias=attn_bias,
                            **factory_kwargs)

    def attention_qkvpacked(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor | Sequence[int],
        max_seqlen: int | torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        del max_seqlen
        xqkv = self.wqkv(x)
        xqkv = xqkv.view(
            x.shape[0],
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        xq, xk, xv = torch.unbind(xqkv, dim=1)
        xq, xk = apply_rope(xq, xk, rope_freqs_cis)
        return self.wo(
            packed_scaled_dot_product_attention(xq, xk, xv, cu_seqlens))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor | Sequence[int],
        max_seqlen: int | torch.Tensor,
        rope_freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        hidden_states = self.attention_qkvpacked(
            hidden_states,
            cu_seqlens,
            max_seqlen,
            rope_freqs_cis,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MoonViT3dEncoder(nn.Module):
    """Packed MoonViT encoder with clip-local non-causal attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict[str, Any],
        video_attn_type: str = 'spatial_temporal',
        rope_max_height: int = 512,
        rope_max_width: int = 512,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if video_attn_type != 'spatial_temporal':
            raise NotImplementedError(
                f'Unsupported MoonViT attention type: {video_attn_type}.')
        self.video_attn_type = video_attn_type
        self.rope_2d = Rope2DPosEmbRepeated(
            block_cfg['hidden_dim'] // block_cfg['num_heads'],
            max_height=rope_max_height,
            max_width=rope_max_width,
        )
        self.blocks = nn.ModuleList([
            MoonViTEncoderLayer(
                **block_cfg,
                dtype=dtype,
                device=device,
            ) for _ in range(num_layers)
        ])
        self.final_layernorm = nn.LayerNorm(hidden_dim,
                                            **_factory_kwargs(dtype, device))

    def forward(self, hidden_states: torch.Tensor,
                grid_thws: torch.Tensor) -> torch.Tensor:
        shapes = _grid_shapes(grid_thws)
        expected_tokens = sum(t * h * w for t, h, w in shapes)
        if hidden_states.ndim != 2 or hidden_states.shape[0] != expected_tokens:
            raise ValueError(
                f'Packed hidden states must contain {expected_tokens} tokens, got {tuple(hidden_states.shape)}.'
            )

        rope_freqs_cis = self.rope_2d.get_freqs_cis(grid_thws,
                                                    hidden_states.device)
        lengths = [t * h * w for t, h, w in shapes]
        cu_seqlens = [0]
        for length in lengths:
            cu_seqlens.append(cu_seqlens[-1] + length)
        max_seqlen = max(lengths)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens,
                max_seqlen,
                rope_freqs_cis,
            )
        return self.final_layernorm(hidden_states)


def tpool_patch_merger(
        x: torch.Tensor,
        grid_thws: torch.Tensor,
        merge_kernel_size: int | Sequence[int] = (2, 2),
) -> list[torch.Tensor]:
    """Temporal-mean then arrange each spatial 2x2 block for projection."""
    shapes = _grid_shapes(grid_thws)
    kernel_height, kernel_width = _normalize_pair(merge_kernel_size,
                                                  'merge_kernel_size')
    expected_tokens = sum(t * h * w for t, h, w in shapes)
    if x.ndim != 2 or x.shape[0] != expected_tokens:
        raise ValueError(
            f'Merger input must have {expected_tokens} rows, got {tuple(x.shape)}.'
        )

    outputs = []
    offset = 0
    hidden_dim = x.shape[-1]
    for t, h, w in shapes:
        if h % kernel_height != 0 or w % kernel_width != 0:
            raise ValueError(
                f'Vision grid {(t, h, w)} is not divisible by merge kernel '
                f'{(kernel_height, kernel_width)}.')

        seq = x[offset:offset + t * h * w]
        new_height = h // kernel_height
        new_width = w // kernel_width
        seq = seq.view(
            t,
            new_height,
            kernel_height,
            new_width,
            kernel_width,
            hidden_dim,
        )
        seq = seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        outputs.append(
            seq.view(new_height * new_width, kernel_height * kernel_width,
                     hidden_dim))
        offset += t * h * w
    return outputs


class MoonViT3dModel(nn.Module):
    """Native replicated Kimi-K2.6 MoonViT3D vision tower."""

    model_type = 'moonvit3d'

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.config = config
        self.merge_kernel_size = _normalize_pair(
            _config_value(config, 'merge_kernel_size'), 'merge_kernel_size')
        self.patch_size = _normalize_pair(_config_value(config, 'patch_size'),
                                          'patch_size')
        self.merge_type = _config_value(config, 'merge_type')

        hidden_size = int(
            _config_value(config, 'hidden_size', 'vt_hidden_size'))
        num_heads = int(
            _config_value(config, 'num_attention_heads',
                          'vt_num_attention_heads'))
        num_layers = int(
            _config_value(config, 'num_hidden_layers', 'vt_num_hidden_layers'))
        intermediate_size = int(
            _config_value(config, 'intermediate_size', 'vt_intermediate_size'))
        rope_max_height = int(
            _config_value(config, 'rope_max_height', default=512))
        rope_max_width = int(
            _config_value(config, 'rope_max_width', default=512))

        self.patch_embed = MoonVision3dPatchEmbed(
            out_dim=hidden_size,
            patch_size=self.patch_size,
            pos_emb_height=int(_config_value(config, 'init_pos_emb_height')),
            pos_emb_width=int(_config_value(config, 'init_pos_emb_width')),
            pos_emb_time=int(_config_value(config, 'init_pos_emb_time')),
            pos_emb_type=_config_value(config, 'pos_emb_type'),
            dtype=dtype,
            device=device,
        )
        self.encoder = MoonViT3dEncoder(
            hidden_dim=hidden_size,
            num_layers=num_layers,
            block_cfg={
                'num_heads': num_heads,
                'hidden_dim': hidden_size,
                'mlp_dim': intermediate_size,
                'activation': nn.GELU(approximate='tanh'),
                'attn_bias': True,
            },
            video_attn_type=_config_value(config, 'video_attn_type'),
            rope_max_height=rope_max_height,
            rope_max_width=rope_max_width,
            dtype=dtype,
            device=device,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(self, pixel_values: torch.Tensor,
                grid_thws: torch.Tensor) -> list[torch.Tensor]:
        if self.merge_type != 'sd2_tpool':
            raise NotImplementedError(
                f'Unsupported Kimi vision merge type: {self.merge_type}.')
        hidden_states = self.patch_embed(pixel_values, grid_thws)
        hidden_states = self.encoder(hidden_states, grid_thws)
        return tpool_patch_merger(hidden_states, grid_thws,
                                  self.merge_kernel_size)


class PatchMergerMLP(nn.Module):
    """Kimi patch-merger projector with checkpoint-compatible names."""

    def __init__(
        self,
        config: Any,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        merge_height, merge_width = _normalize_pair(
            _config_value(config, 'merge_kernel_size'),
            'merge_kernel_size',
        )
        mm_hidden_size = int(
            _config_value(config, 'mm_hidden_size', 'hidden_size',
                          'vt_hidden_size'))
        text_hidden_size = int(_config_value(config, 'text_hidden_size'))
        self.hidden_size = mm_hidden_size * merge_height * merge_width
        self.mm_hidden_size = mm_hidden_size
        self.merge_area = merge_height * merge_width

        factory_kwargs = _factory_kwargs(dtype, device)
        self.pre_norm = nn.LayerNorm(
            mm_hidden_size,
            eps=float(_config_value(config, 'projector_ln_eps', default=1e-5)),
            **factory_kwargs,
        )
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, **factory_kwargs),
            nn.GELU(),
            nn.Linear(self.hidden_size, text_hidden_size, **factory_kwargs),
        )

    def _project_item(self, item: torch.Tensor) -> torch.Tensor:
        if item.ndim != 3 or item.shape[-2:] != (self.merge_area,
                                                 self.mm_hidden_size):
            raise ValueError(
                'Each Kimi projector item must have shape '
                f'[tokens, {self.merge_area}, {self.mm_hidden_size}], got {tuple(item.shape)}.'
            )
        return self.proj(
            self.pre_norm(item).reshape(item.shape[0], self.hidden_size))

    def forward(self, x: list[torch.Tensor] | tuple[torch.Tensor, ...]
                | torch.Tensor):
        if isinstance(x, (list, tuple)):
            return [self._project_item(item) for item in x]
        if x.ndim < 3 or x.shape[-2:] != (self.merge_area,
                                          self.mm_hidden_size):
            raise ValueError(
                'Kimi projector tensor must end in '
                f'[{self.merge_area}, {self.mm_hidden_size}], got {tuple(x.shape)}.'
            )
        leading_shape = x.shape[:-2]
        projected = self.proj(self.pre_norm(x).reshape(-1, self.hidden_size))
        return projected.reshape(*leading_shape, projected.shape[-1])


# Upstream-compatible alias used by checkpoint/reference comparisons.
MoonViT3dPretrainedModel = MoonViT3dModel

# The wrapper should use these constructors. They preserve the lightweight
# language-only path through LMDeploy's standard dummy-module mechanism.
KimiK25VisionTower = vlm_model(MoonViT3dModel)
KimiK25MultiModalProjector = vlm_model(PatchMergerMLP)

__all__ = [
    'KimiK25MultiModalProjector',
    'KimiK25VisionTower',
    'Learnable2DInterpPosEmbDividedFixed',
    'Learnable2DInterpPosEmbDivided_fixed',
    'MoonViT3dEncoder',
    'MoonViT3dModel',
    'MoonViT3dPretrainedModel',
    'MoonViTEncoderLayer',
    'MoonVision3dPatchEmbed',
    'PatchMergerMLP',
    'Rope2DPosEmbRepeated',
    'apply_rope',
    'interpolate_2d_pos_embed',
    'packed_scaled_dot_product_attention',
    'tpool_patch_merger',
]
