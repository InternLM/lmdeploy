# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/google-research/timesfm for LMDeploy inference.
import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lmdeploy.pytorch.nn import LayerNorm
from lmdeploy.pytorch.nn.linear import build_colwise_linear, build_rowwise_linear

TS_GEN_TOKEN_ID = 123456


class RMSNorm(nn.Module):
    """RMS normalization."""

    def __init__(
        self,
        num_features: int,
        *,
        epsilon: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(num_features, dtype=dtype, device=device))
        self.num_features = num_features
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        var = torch.mean(torch.square(inputs), dim=-1, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        normed_inputs = normed_inputs * self.scale
        return normed_inputs


class ResidualBlock(nn.Module):
    """Residual block with two linear layers and a linear residual
    connection."""

    def __init__(
        self,
        input_dims,
        hidden_dims,
        output_dims,
        use_bias,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.hidden_layer = build_colwise_linear(
            in_features=input_dims,
            out_features=hidden_dims,
            bias=use_bias,
            dtype=dtype,
            device=device,
        )
        self.output_layer = build_rowwise_linear(
            in_features=hidden_dims,
            out_features=output_dims,
            bias=use_bias,
            dtype=dtype,
            device=device,
        )
        self.residual_layer = build_rowwise_linear(
            in_features=input_dims,
            out_features=output_dims,
            bias=use_bias,
            dtype=dtype,
            device=device,
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.activation(self.hidden_layer(x))) + self.residual_layer(x)

@dataclasses.dataclass(frozen=False)
class DecodeCache:
    """Cache for decoding."""

    next_index: torch.Tensor
    num_masked: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    kv_mask: torch.Tensor | None = None  # (B, cache_size), True=padded/masked


def update_running_stats(
    n: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    x: torch.Tensor,
    mask: torch.Tensor,
):
    """Updates the running stats."""
    is_legit = torch.logical_not(mask)
    inc_n = torch.sum(is_legit.to(x.dtype), dim=-1)

    inc_mu_numerator = torch.sum(x * is_legit, dim=-1)
    inc_n_safe = torch.where(inc_n == 0, 1.0, inc_n)
    inc_mu = inc_mu_numerator / inc_n_safe
    inc_mu = torch.where(inc_n == 0, 0.0, inc_mu)

    inc_var_numerator = torch.sum(((x - inc_mu.unsqueeze(-1))**2) * is_legit, dim=-1)
    inc_var = inc_var_numerator / inc_n_safe
    inc_var = torch.where(inc_n == 0, 0.0, inc_var)
    inc_sigma = torch.sqrt(inc_var)

    new_n = n + inc_n
    new_n_safe = torch.where(new_n == 0, 1.0, new_n)

    new_mu = (n * mu + inc_mu * inc_n) / new_n_safe
    new_mu = torch.where(new_n == 0, 0.0, new_mu)

    term1 = n * sigma.pow(2)
    term2 = inc_n * inc_sigma.pow(2)
    term3 = n * (mu - new_mu).pow(2)
    term4 = inc_n * (inc_mu - new_mu).pow(2)

    new_var = (term1 + term2 + term3 + term4) / new_n_safe
    new_var = torch.where(new_n == 0, 0.0, new_var)
    new_sigma = torch.sqrt(torch.clamp(new_var, min=0.0))

    return (w := (new_n, new_mu, new_sigma), w)


def revin(
    x: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    reverse: bool = False,
):
    """Reversible instance normalization."""
    if len(mu.shape) == len(x.shape) - 1:
        mu = mu[..., None]
        sigma = sigma[..., None]
    elif len(mu.shape) == len(x.shape) - 2:
        mu = mu[..., None, None]
        sigma = sigma[..., None, None]

    _tolerance = 1e-6
    if reverse:
        return x * sigma + mu
    else:
        return (x - mu) / torch.where(sigma < _tolerance, 1.0, sigma)


def make_attn_mask(
    query_length: int,
    num_all_masked_kv: torch.Tensor,
    query_index_offset: torch.Tensor | None = None,
    kv_length: int = 0,
) -> torch.Tensor:
    """Makes attention mask."""
    if kv_length == 0:
        kv_length = query_length

    q_index = torch.arange(query_length, device=num_all_masked_kv.device)[None, None, :, None]
    if query_index_offset is not None:
        q_index = q_index + query_index_offset[:, None, None, None]
    kv_index = torch.arange(kv_length, device=num_all_masked_kv.device)[None, None, None, :]
    return torch.logical_and(
        q_index >= kv_index,
        kv_index >= num_all_masked_kv[:, None, None, None],
    )


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding."""

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
    ):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale

    def forward(
        self,
        inputs: torch.Tensor,
        position: torch.Tensor | None = None,
    ):
        """Generates a JTensor of sinusoids with different frequencies."""
        if self.embedding_dims != inputs.shape[-1]:
            raise ValueError('The embedding dims of the rotary position embedding'
                             'must match the hidden dimension of the inputs.')
        half_embedding_dim = self.embedding_dims // 2
        fraction = (2 * torch.arange(0, half_embedding_dim, device=inputs.device) / self.embedding_dims)
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale)**fraction).to(inputs.device)
        if position is None:
            seq_length = inputs.shape[1]
            position = torch.arange(seq_length, dtype=torch.float32, device=inputs.device)[None, :]

        if len(inputs.shape) == 4:
            position = position[..., None, None]
            timescale = timescale[None, None, None, :]
        elif len(inputs.shape) == 3:
            position = position[..., None]
            timescale = timescale[None, None, :]
        else:
            raise ValueError('Inputs must be of rank 3 or 4.')

        sinusoid_inp = position / timescale
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        first_half, second_half = torch.chunk(inputs, 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat([first_part, second_part], dim=-1)


def _torch_dot_product_attention(query, key, value, mask=None):
    """Same (unscaled) attention as _dot_product_attention, fused kernel."""
    safe_mask = mask
    fully_masked_rows = None
    if mask is not None:
        # Avoid NaNs for fully masked query rows while keeping fused SDPA.
        fully_masked_rows = ~mask.any(dim=-1, keepdim=True)
        if fully_masked_rows.any():
            dummy_key_mask = torch.zeros_like(mask)
            dummy_key_mask[..., 0] = True
            safe_mask = mask | (fully_masked_rows & dummy_key_mask)

    attention_dtype = value.dtype
    if query.dtype != attention_dtype:
        query = query.to(attention_dtype)
    if key.dtype != attention_dtype:
        key = key.to(attention_dtype)

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    output = F.scaled_dot_product_attention(query, key, value, attn_mask=safe_mask, scale=1.0)

    if fully_masked_rows is not None and fully_masked_rows.any():
        output = output.masked_fill(fully_masked_rows, 0.0)

    output = output.permute(0, 2, 1, 3)
    return output


class PerDimScale(nn.Module):
    """Per-dimension scaling."""

    def __init__(
        self,
        num_dims: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_dims = num_dims
        self.per_dim_scale = nn.Parameter(torch.zeros(num_dims, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_factor = (1.442695041 / math.sqrt(self.num_dims) * F.softplus(self.per_dim_scale))
        return x * scale_factor


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads

        if self.in_features % self.num_heads != 0:
            raise ValueError(f"Memory dimension ({self.in_features}) must be divisible by "
                             f"'num_heads' heads ({self.num_heads}).")

        self.qkv_proj = build_colwise_linear(
            self.in_features,
            3 * self.in_features,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.out = build_rowwise_linear(
            self.in_features,
            self.in_features,
            bias=False,
            dtype=dtype,
            device=device,
        )

        self.query_ln = RMSNorm(self.head_dim, dtype=dtype, device=device)
        self.key_ln = RMSNorm(self.head_dim, dtype=dtype, device=device)
        self.rotary_position_embedding = RotaryPositionalEmbedding(embedding_dims=self.head_dim)
        self.per_dim_scale = PerDimScale(num_dims=self.head_dim, dtype=dtype, device=device)

    def forward(
        self,
        inputs_q: torch.Tensor,
        *,
        decode_cache: DecodeCache | None = None,
        patch_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecodeCache | None]:
        b, n_patches, _ = inputs_q.shape
        if patch_mask is None:
            patch_mask = torch.zeros(b, n_patches, dtype=torch.bool, device=inputs_q.device)

        qkv = self.qkv_proj(inputs_q)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        query = query.view(b, n_patches, self.num_heads, self.head_dim)
        key = key.view(b, n_patches, self.num_heads, self.head_dim)
        value = value.view(b, n_patches, self.num_heads, self.head_dim)

        if decode_cache is None:
            # make_attn_mask expects a left-padding count.
            is_valid = ~patch_mask
            has_valid = is_valid.any(dim=-1)
            first_valid = torch.argmax(is_valid.to(torch.int32), dim=-1)
            seq_len = torch.tensor(patch_mask.shape[-1], dtype=torch.int32, device=patch_mask.device)
            num_masked = torch.where(has_valid, first_valid, seq_len)
            next_index = torch.zeros_like(num_masked, dtype=torch.int32)
        else:
            is_valid = ~patch_mask
            has_valid = is_valid.any(dim=-1)
            first_valid = torch.argmax(is_valid.to(torch.int32), dim=-1)
            seq_len = torch.tensor(patch_mask.shape[-1], dtype=torch.int32, device=patch_mask.device)
            leading_masked = torch.where(has_valid, first_valid, seq_len)
            num_masked = leading_masked + decode_cache.num_masked
            next_index = decode_cache.next_index.clone()

        position = (torch.arange(n_patches, device=inputs_q.device)[None, :] + next_index[:, None] -
                    num_masked[:, None])
        query = self.rotary_position_embedding(query, position)
        key = self.rotary_position_embedding(key, position)

        query = self.query_ln(query)
        key = self.key_ln(key)
        query = self.per_dim_scale(query)

        if decode_cache is not None:
            _, decode_cache_size, _, _ = decode_cache.value.shape

            start = decode_cache.next_index[0]
            end = start + n_patches

            decode_cache.key[:, start:end] = key
            decode_cache.value[:, start:end] = value

            if decode_cache.kv_mask is None:
                decode_cache.kv_mask = torch.ones(
                    b,
                    decode_cache_size,
                    dtype=torch.bool,
                    device=patch_mask.device,
                )
            decode_cache.kv_mask[:, start:end] = patch_mask

            key = decode_cache.key
            value = decode_cache.value
            decode_cache.next_index += n_patches
            decode_cache.num_masked = num_masked
            attn_mask = make_attn_mask(
                query_length=n_patches,
                num_all_masked_kv=num_masked,
                query_index_offset=next_index,
                kv_length=decode_cache_size,
            )
            kv_valid = ~decode_cache.kv_mask  # (B, decode_cache_size)
            attn_mask = attn_mask & kv_valid[:, None, None, :]
        else:
            attn_mask = make_attn_mask(query_length=n_patches, num_all_masked_kv=num_masked)
            kv_valid = ~patch_mask  # (B, n_patches), True=valid
            attn_mask = attn_mask & kv_valid[:, None, None, :]

        x = _torch_dot_product_attention(query, key, value, mask=attn_mask)

        x = x.reshape(b, n_patches, self.in_features)
        out = self.out(x)
        return out, decode_cache


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention from a query stream into a static KV context.

    Q comes from the transformer's residual stream; K/V come from an external context tensor (e.g. Q-former output). No
    RoPE, no decode_cache: the KV context is treated as a static set of tokens at every layer / decode step.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        *,
        kv_features: int | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.kv_features = kv_features if kv_features is not None else in_features
        self.head_dim = in_features // num_heads

        if self.in_features % self.num_heads != 0:
            raise ValueError(f"Memory dimension ({self.in_features}) must be divisible by"
                             f" 'num_heads' heads ({self.num_heads}).")

        self.q_proj = build_colwise_linear(
            self.in_features,
            self.in_features,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_proj = build_colwise_linear(
            self.kv_features,
            self.in_features,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.v_proj = build_colwise_linear(
            self.kv_features,
            self.in_features,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.out = build_rowwise_linear(
            self.in_features,
            self.in_features,
            bias=False,
            dtype=dtype,
            device=device,
        )

        self.query_ln = RMSNorm(self.head_dim, dtype=dtype, device=device)
        self.key_ln = RMSNorm(self.head_dim, dtype=dtype, device=device)

        self.per_dim_scale = PerDimScale(num_dims=self.head_dim, dtype=dtype, device=device)

    def forward(
        self,
        inputs_q: torch.Tensor,
        *,
        kv: torch.Tensor,
        kv_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n_q, _ = inputs_q.shape
        n_kv = kv.shape[1]

        if kv.dtype != inputs_q.dtype:
            kv = kv.to(dtype=inputs_q.dtype)

        query = self.q_proj(inputs_q).view(b, n_q, self.num_heads, self.head_dim)
        key = self.k_proj(kv).view(b, n_kv, self.num_heads, self.head_dim)
        value = self.v_proj(kv).view(b, n_kv, self.num_heads, self.head_dim)

        query = self.query_ln(query)
        key = self.key_ln(key)
        query = self.per_dim_scale(query)

        if kv_mask is None:
            attn_mask = torch.ones(
                b,
                1,
                n_q,
                n_kv,
                dtype=torch.bool,
                device=inputs_q.device,
            )
        else:
            kv_valid = ~kv_mask  # True = valid
            attn_mask = kv_valid[:, None, None, :].expand(b, 1, n_q, n_kv)

        x = _torch_dot_product_attention(query, key, value, mask=attn_mask)
        x = x.reshape(b, n_q, self.in_features)
        return self.out(x)


class Transformer(nn.Module):
    """Classic Transformer used in Forecaster."""

    def __init__(
        self,
        model_dims: int,
        hidden_dims: int,
        num_heads: int,
        *,
        cross_attn_kv_dim: int = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.pre_attn_ln = RMSNorm(model_dims, dtype=dtype, device=device)
        self.post_attn_ln = RMSNorm(model_dims, dtype=dtype, device=device)

        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            in_features=model_dims,
            dtype=dtype,
            device=device,
        )

        _cross_kv_dim = cross_attn_kv_dim or model_dims
        self.cross_attn_ln = RMSNorm(model_dims, dtype=dtype, device=device)
        self.cross_attn = MultiHeadCrossAttention(
            num_heads=num_heads,
            in_features=model_dims,
            kv_features=_cross_kv_dim,
            dtype=dtype,
            device=device,
        )
        self.cross_attn_gate = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

        self.pre_ff_ln = RMSNorm(model_dims, dtype=dtype, device=device)
        self.post_ff_ln = RMSNorm(model_dims, dtype=dtype, device=device)

        self.ff0 = build_colwise_linear(
            in_features=model_dims,
            out_features=hidden_dims,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.ff1 = build_rowwise_linear(
            in_features=hidden_dims,
            out_features=model_dims,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.activation = nn.SiLU()

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor,
        decode_cache: DecodeCache | None = None,
        cross_kv: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, DecodeCache | None]:
        attn_output, decode_cache = self.attn(
            inputs_q=self.pre_attn_ln(input_embeddings),
            decode_cache=decode_cache,
            patch_mask=patch_mask,
        )
        attn_output = self.post_attn_ln(attn_output) + input_embeddings

        if cross_kv is not None:
            cross_out = self.cross_attn(
                self.cross_attn_ln(attn_output),
                kv=cross_kv,
            )
            cross_out = torch.tanh(self.cross_attn_gate) * cross_out
            attn_output = attn_output + cross_out

        output_embeddings = (self.post_ff_ln(self.ff1(self.activation(self.ff0(self.pre_ff_ln(attn_output))))) +
                             attn_output)
        return output_embeddings, decode_cache


def strip_leading_nans(arr):
    """Removes contiguous NaN values from the beginning of a NumPy array."""
    isnan = np.isnan(arr)
    first_valid_index = np.argmax(~isnan)
    return arr[first_valid_index:]


def linear_interpolation(arr):
    """Linear interpolation to fill NaN values in a 1D numpy array."""
    nans = np.isnan(arr)
    if not np.any(nans):
        return arr

    def x(z):
        return z.nonzero()[0]

    nans_indices = x(nans)
    non_nans_indices = x(~nans)
    non_nans_values = arr[~nans]

    try:
        arr[nans] = np.interp(nans_indices, non_nans_indices, non_nans_values)
    except ValueError:
        if non_nans_values:
            mu = np.nanmean(arr)
        else:
            mu = 0.0
        arr = np.where(np.isfinite(arr), arr, mu)
    return arr


class ForecasterBackbone(nn.Module):
    """Forecaster 2.5 with 200M parameters."""

    context_limit = 16384
    _quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    _num_layers = 20

    _tokenizer_kwargs = dict(input_dims=64, hidden_dims=1280, output_dims=1280, use_bias=True)
    _output_point_kwargs = dict(input_dims=1280, hidden_dims=1280, output_dims=1280, use_bias=False)
    _output_quantile_kwargs = dict(
        input_dims=1280,
        hidden_dims=1280,
        output_dims=10240,
        use_bias=False,
    )

    _xf_kwargs = dict(
        model_dims=1280,
        hidden_dims=1280,
        num_heads=16,
    )

    def __init__(
        self,
        cross_attn_kv_dim: int = 0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()

        self.p = 32
        self.o = 128
        self.os = 1024
        self.m = self.o // self.p
        self.x = self._num_layers
        self.h = self._xf_kwargs['num_heads']
        self.md = self._xf_kwargs['model_dims']
        self.hd = self.md // self.h
        self.q = len(self._quantiles) + 1
        self.aridx = 5

        xf_kwargs = dict(self._xf_kwargs)
        xf_kwargs['cross_attn_kv_dim'] = int(cross_attn_kv_dim) or self.md

        self.tokenizer = ResidualBlock(**self._tokenizer_kwargs, dtype=dtype, device=device)
        self.stacked_xf = nn.ModuleList([Transformer(**xf_kwargs, dtype=dtype, device=device) for _ in range(self.x)])
        self.output_projection_point = ResidualBlock(**self._output_point_kwargs, dtype=dtype, device=device)
        self.output_projection_quantiles = ResidualBlock(
            **self._output_quantile_kwargs,
            dtype=dtype,
            device=device,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        decode_caches: list | None = None,
        cross_kv: torch.Tensor | None = None,
    ):
        """Forward pass for the history-only path with cross-attention
        injection."""
        input_dtype = inputs.dtype
        model_dtype = next(self.parameters()).dtype
        if inputs.dtype != model_dtype:
            inputs = inputs.to(dtype=model_dtype)

        tokenizer_inputs = torch.cat([inputs, masks.to(inputs.dtype)], dim=-1)
        input_embeddings = self.tokenizer(tokenizer_inputs)

        if cross_kv is not None:
            cross_kv = cross_kv.to(device=input_embeddings.device, dtype=input_embeddings.dtype)

        if decode_caches is None:
            decode_caches = [None] * self.x
        new_decode_caches = []

        output_embeddings = input_embeddings
        token_masks = masks[..., -1]

        for i, layer in enumerate(self.stacked_xf):
            output_embeddings, new_cache = layer(
                output_embeddings,
                token_masks,
                decode_caches[i],
                cross_kv=cross_kv,
            )
            new_decode_caches.append(new_cache)

        output_ts = self.output_projection_point(output_embeddings)
        output_quantile_spread = self.output_projection_quantiles(output_embeddings)

        if output_ts.dtype != input_dtype:
            input_embeddings = input_embeddings.to(dtype=input_dtype)
            output_embeddings = output_embeddings.to(dtype=input_dtype)
            output_ts = output_ts.to(dtype=input_dtype)
            output_quantile_spread = output_quantile_spread.to(dtype=input_dtype)

        return (
            input_embeddings,
            output_embeddings,
            output_ts,
            output_quantile_spread,
        ), new_decode_caches

    def decode(
        self,
        horizon: int,
        inputs,
        masks,
        cross_kv: torch.Tensor | None = None,
    ):
        """Decodes the time series."""
        with torch.no_grad():
            batch_size, context = inputs.shape[0], inputs.shape[1]
            num_decode_steps = (horizon - 1) // self.o
            num_input_patches = context // self.p
            decode_cache_size = num_input_patches + num_decode_steps * self.m
            cache_dtype = next(self.parameters()).dtype

            patched_inputs = torch.reshape(inputs, (batch_size, -1, self.p))
            patched_masks = torch.reshape(masks, (batch_size, -1, self.p))

            n = torch.zeros(batch_size, device=inputs.device)
            mu = torch.zeros(batch_size, device=inputs.device)
            sigma = torch.zeros(batch_size, device=inputs.device)
            patch_mu = []
            patch_sigma = []
            for i in range(num_input_patches):
                (n, mu, sigma), _ = update_running_stats(n, mu, sigma, patched_inputs[:, i], patched_masks[:, i])
                patch_mu.append(mu)
                patch_sigma.append(sigma)
            last_n, last_mu, last_sigma = n, mu, sigma
            context_mu = torch.stack(patch_mu, dim=1)
            context_sigma = torch.stack(patch_sigma, dim=1)

            decode_caches = [
                DecodeCache(
                    next_index=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
                    num_masked=torch.zeros(batch_size, dtype=torch.int32, device=inputs.device),
                    key=torch.zeros(
                        batch_size,
                        decode_cache_size,
                        self.h,
                        self.hd,
                        device=inputs.device,
                        dtype=cache_dtype,
                    ),
                    value=torch.zeros(
                        batch_size,
                        decode_cache_size,
                        self.h,
                        self.hd,
                        device=inputs.device,
                        dtype=cache_dtype,
                    ),
                ) for _ in range(self.x)
            ]

            normed_inputs = revin(patched_inputs, context_mu, context_sigma, reverse=False)
            normed_inputs = torch.where(patched_masks, 0.0, normed_inputs)
            (_, _, normed_outputs, normed_quantile_spread), decode_caches = self(
                normed_inputs,
                patched_masks,
                decode_caches,
                cross_kv=cross_kv,
            )
            renormed_outputs = torch.reshape(
                revin(normed_outputs, context_mu, context_sigma, reverse=True),
                (batch_size, -1, self.o, self.q),
            )
            renormed_quantile_spread = torch.reshape(
                revin(normed_quantile_spread, context_mu, context_sigma, reverse=True),
                (batch_size, -1, self.os, self.q),
            )[:, -1, ...]

            ar_outputs = []
            last_renormed_output = renormed_outputs[:, -1, :, self.aridx]

            for _ in range(num_decode_steps):
                new_patched_input = torch.reshape(last_renormed_output, (batch_size, self.m, self.p))
                new_mask = torch.zeros_like(new_patched_input, dtype=torch.bool)

                n, mu, sigma = last_n, last_mu, last_sigma
                new_mus, new_sigmas = [], []
                for i in range(self.m):
                    (n, mu, sigma), _ = update_running_stats(n, mu, sigma, new_patched_input[:, i], new_mask[:, i])
                    new_mus.append(mu)
                    new_sigmas.append(sigma)
                last_n, last_mu, last_sigma = n, mu, sigma
                new_mu = torch.stack(new_mus, dim=1)
                new_sigma = torch.stack(new_sigmas, dim=1)

                new_normed_input = revin(new_patched_input, new_mu, new_sigma, reverse=False)
                (_, _, new_normed_output, _), decode_caches = self(
                    new_normed_input,
                    new_mask,
                    decode_caches,
                    cross_kv=cross_kv,
                )

                new_renormed_output = torch.reshape(
                    revin(new_normed_output, new_mu, new_sigma, reverse=True),
                    (batch_size, self.m, self.o, self.q),
                )
                ar_outputs.append(new_renormed_output[:, -1, ...])
                last_renormed_output = new_renormed_output[:, -1, :, self.aridx]

            if num_decode_steps > 0:
                ar_renormed_outputs = torch.stack(ar_outputs, dim=1)
            else:
                ar_renormed_outputs = None

        return renormed_outputs, renormed_quantile_spread, ar_renormed_outputs


class QFormerAttention(nn.Module):
    """Batch-first attention used by QFormer blocks.

    The load hook accepts PyTorch packed attention state dict keys
    (``in_proj_weight`` / ``in_proj_bias``) and splits them into LMDeploy q/k/v
    projections.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = build_colwise_linear(
            embed_dim,
            embed_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.k_proj = build_colwise_linear(
            embed_dim,
            embed_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.v_proj = build_colwise_linear(
            embed_dim,
            embed_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.out_proj = build_rowwise_linear(
            embed_dim,
            embed_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.register_load_state_dict_pre_hook(self._load_mha_state_dict_hook)

    def _load_mha_state_dict_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        in_proj_weight_key = prefix + 'in_proj_weight'
        if in_proj_weight_key in state_dict:
            q_weight, k_weight, v_weight = state_dict.pop(in_proj_weight_key).chunk(3, dim=0)
            state_dict[prefix + 'q_proj.weight'] = q_weight
            state_dict[prefix + 'k_proj.weight'] = k_weight
            state_dict[prefix + 'v_proj.weight'] = v_weight

        in_proj_bias_key = prefix + 'in_proj_bias'
        if in_proj_bias_key in state_dict:
            q_bias, k_bias, v_bias = state_dict.pop(in_proj_bias_key).chunk(3, dim=0)
            state_dict[prefix + 'q_proj.bias'] = q_bias
            state_dict[prefix + 'k_proj.bias'] = k_bias
            state_dict[prefix + 'v_proj.bias'] = v_bias

    def _shape(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = ~key_padding_mask.to(dtype=torch.bool, device=query.device)
            attn_mask = attn_mask[:, None, None, :]

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(query.shape[0], query.shape[1], self.embed_dim)
        return self.out_proj(attn_output)


class _QFormerBlock(nn.Module):
    """One Q-former block: self-attn over queries, cross-attn into source, FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_mult: float = 4.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.self_attn_ln = LayerNorm(hidden_dim, eps=1e-5, dtype=dtype, device=device)
        self.self_attn = QFormerAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dtype=dtype,
            device=device,
        )

        self.cross_attn_ln_q = LayerNorm(hidden_dim, eps=1e-5, dtype=dtype, device=device)
        self.cross_attn_ln_kv = LayerNorm(hidden_dim, eps=1e-5, dtype=dtype, device=device)
        self.cross_attn = QFormerAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dtype=dtype,
            device=device,
        )

        ffn_hidden = int(round(hidden_dim * ffn_mult))
        self.ffn_ln = LayerNorm(hidden_dim, eps=1e-5, dtype=dtype, device=device)
        self.ffn = nn.Sequential(
            build_colwise_linear(
                hidden_dim,
                ffn_hidden,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.GELU(),
            nn.Identity(),
            build_rowwise_linear(
                ffn_hidden,
                hidden_dim,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.Identity(),
        )

    def forward(
        self,
        queries: Tensor,
        kv: Tensor,
        kv_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        q_norm = self.self_attn_ln(queries)
        sa_out = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + sa_out

        q_norm = self.cross_attn_ln_q(queries)
        kv_norm = self.cross_attn_ln_kv(kv)
        ca_out = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            key_padding_mask=kv_key_padding_mask,
        )
        queries = queries + ca_out

        queries = queries + self.ffn(self.ffn_ln(queries))
        return queries


class QFormer(nn.Module):
    """BLIP-2-style Q-former: learned queries cross-attend to a source sequence.

    Args:
        in_dim: Source feature dim (e.g. ts encoder hidden, or LLM hidden).
        out_dim: Output (query) hidden dim. Should match the consumer module dim.
        num_query_tokens: Number of learned query tokens.
        num_heads: Attention heads inside the Q-former.
        num_layers: Number of stacked Q-former blocks.
        ffn_mult: FFN expansion ratio.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_query_tokens: int,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_mult: float = 4.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if num_query_tokens <= 0:
            raise ValueError(f"num_query_tokens must be positive, got {num_query_tokens}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim ({out_dim}) must be divisible by num_heads ({num_heads})")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_query_tokens = num_query_tokens

        self.input_proj = build_colwise_linear(
            in_dim,
            out_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        self.input_ln = LayerNorm(out_dim, eps=1e-5, dtype=dtype, device=device)

        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, out_dim, dtype=dtype, device=device))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        self.blocks = nn.ModuleList([
            _QFormerBlock(
                hidden_dim=out_dim,
                num_heads=num_heads,
                ffn_mult=ffn_mult,
                dtype=dtype,
                device=device,
            ) for _ in range(num_layers)
        ])

        self.output_ln = LayerNorm(out_dim, eps=1e-5, dtype=dtype, device=device)

    def forward(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compress `src` into `num_query_tokens` learned query vectors.

        Args:
            src: (B, T, in_dim) source token embeddings.
            src_key_padding_mask: (B, T) bool, True = padded position (ignored).

        Returns:
            (B, num_query_tokens, out_dim) query outputs.
        """
        if src.dim() != 3:
            raise ValueError(f"src must be (B, T, D), got shape {tuple(src.shape)}")
        if src.size(-1) != self.in_dim:
            raise ValueError(f"src last dim ({src.size(-1)}) does not match in_dim ({self.in_dim})")

        kv = self.input_ln(self.input_proj(src))

        if src_key_padding_mask is not None:
            if src_key_padding_mask.shape != src.shape[:2]:
                raise ValueError('src_key_padding_mask shape must equal src.shape[:2]:'
                                 f" {tuple(src_key_padding_mask.shape)} != {tuple(src.shape[:2])}")
            kpm = src_key_padding_mask.to(dtype=torch.bool, device=kv.device)
            # Avoid fully padded attention rows.
            all_padded = kpm.all(dim=1)
            if all_padded.any():
                kpm = kpm.clone()
                kpm[all_padded, 0] = False
        else:
            kpm = None

        queries = self.query_tokens.expand(src.size(0), -1, -1).to(dtype=kv.dtype)
        for block in self.blocks:
            queries = block(queries, kv, kv_key_padding_mask=kpm)

        return self.output_ln(queries)


class Aligner(nn.Module):
    """Aligns precomputed LLM / TS-encoder embeddings into Forecaster's cross-
    attn KV space, and predicts the forecast horizon.

    Two per-modality Q-formers compress the raw hidden-state sequences into a fixed number of query tokens; their
    concatenation is the static KV stream the Forecaster transformer cross-attends to. A small linear *prediction-length
    head* sits on the LLM Q-former's compressed output and regresses log1p(horizon).
    """

    def __init__(
        self,
        config: 'TSForecasterConfig',
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        qformer_kwargs = dict(
            out_dim=config.qformer_hidden_dim,
            num_query_tokens=config.qformer_num_query_tokens,
            num_heads=config.qformer_num_heads,
            num_layers=config.qformer_num_layers,
            dtype=dtype,
            device=device,
        )
        self.ts_qformer = QFormer(in_dim=config.d_ts_encoder, **qformer_kwargs)
        self.llm_qformer = QFormer(in_dim=config.d_llm, **qformer_kwargs)

        self.horizon_head = nn.Sequential(
            LayerNorm(config.qformer_hidden_dim, eps=1e-5, dtype=dtype, device=device),
            build_colwise_linear(
                config.qformer_hidden_dim,
                config.qformer_hidden_dim,
                bias=True,
                dtype=dtype,
                device=device,
            ),
            nn.SiLU(),
            build_rowwise_linear(
                config.qformer_hidden_dim,
                1,
                bias=True,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(
        self,
        llm_embedding_input: Tensor,
        ts_encoder_embedding_input: Tensor,
        llm_embedding_mask: Tensor | None = None,
        ts_encoder_embedding_mask: Tensor | None = None,
    ):
        """Compress both modalities into the Forecaster cross-attention KV
        stream.

        Args:
            llm_embedding_input: (B, T_llm, d_llm) precomputed LLM hidden states.
            ts_encoder_embedding_input: (B, T_ts, d_ts_encoder) precomputed TS-encoder
                hidden states.
            llm_embedding_mask / ts_encoder_embedding_mask: optional (B, T) bool masks,
                True = valid token. Default: all valid.

        Returns:
            ctx: (B, Q_ts + Q_llm, qformer_hidden_dim) cross-attention KV stream.
            llm_chunk: (B, Q_llm, qformer_hidden_dim) compressed LLM tokens.
        """
        ts_param = next(self.ts_qformer.parameters())
        ts_hidden = ts_encoder_embedding_input.to(device=ts_param.device, dtype=ts_param.dtype)
        if ts_encoder_embedding_mask is None:
            ts_pad = torch.zeros(ts_hidden.shape[:2], dtype=torch.bool, device=ts_hidden.device)
        else:
            ts_pad = (~ts_encoder_embedding_mask.to(dtype=torch.bool)).to(device=ts_hidden.device)
        ts_chunk = self.ts_qformer(ts_hidden, src_key_padding_mask=ts_pad)

        llm_param = next(self.llm_qformer.parameters())
        llm_hidden = llm_embedding_input.to(device=llm_param.device, dtype=llm_param.dtype)
        if llm_embedding_mask is None:
            llm_pad = torch.zeros(llm_hidden.shape[:2], dtype=torch.bool, device=llm_hidden.device)
        else:
            llm_pad = (~llm_embedding_mask.to(dtype=torch.bool)).to(device=llm_hidden.device)
        llm_chunk = self.llm_qformer(llm_hidden, src_key_padding_mask=llm_pad)

        ctx = torch.cat([ts_chunk, llm_chunk], dim=1)
        return ctx, llm_chunk

    def predict_horizon(
        self,
        llm_chunk: Tensor,
    ) -> Tensor:
        """Predict per-sample forecast horizon in log1p-space from llm_chunk.

        Args:
            llm_chunk: (B, Q, qformer_hidden_dim) LLM Q-former output.

        Returns:
            (B,) float32 tensor of predicted log1p(horizon).
        """
        if llm_chunk.dim() != 3:
            raise ValueError(f"Expected llm_chunk with shape (B, Q, D), got {tuple(llm_chunk.shape)}")
        head_param = next(self.horizon_head.parameters())
        chunk = llm_chunk.to(device=head_param.device, dtype=head_param.dtype)
        pooled = chunk.mean(dim=1)
        return self.horizon_head(pooled).squeeze(-1).to(dtype=torch.float32)

    @staticmethod
    def decode_horizon(
        pred_log_horizon: Tensor,
        min_h: int = 1,
        max_h: int | None = None,
    ) -> Tensor:
        """Convert log1p-space horizon prediction back to a positive integer
        tensor."""
        horizon = torch.expm1(pred_log_horizon).round().clamp_min(float(min_h))
        if max_h is not None:
            horizon = horizon.clamp_max(float(max_h))
        return horizon.long()


class TSForecasterConfig:
    """Configuration for :class:`TSForecaster`.

    The Forecaster backbone dims (model_dims=1280, patch_len=32, 20 layers, ...) are
    fixed by Forecaster 2.5 200M and are not configurable here.
    """

    model_type = 'timeomni_v2_forecaster'

    TIMESFM_MODEL_DIMS = 1280

    def __init__(
        self,
        d_llm: int = 2560,
        d_ts_encoder: int = 1024,
        qformer_hidden_dim: int = 0,
        qformer_num_query_tokens: int = 32,
        qformer_num_heads: int = 8,
        qformer_num_layers: int = 2,
        horizon_max_length: int = 0,
        max_context: int = 2048,
        max_horizon: int = 1024,
        normalize_inputs: bool = True,
        force_flip_invariance: bool = True,
        infer_is_positive: bool = True,
        fix_quantile_crossing: bool = True,
    ):
        self.d_llm = int(d_llm)
        self.d_ts_encoder = int(d_ts_encoder)

        self.qformer_hidden_dim = int(qformer_hidden_dim) or self.TIMESFM_MODEL_DIMS
        self.qformer_num_query_tokens = int(qformer_num_query_tokens)
        self.qformer_num_heads = int(qformer_num_heads)
        self.qformer_num_layers = int(qformer_num_layers)

        self.horizon_max_length = int(horizon_max_length)

        self.cross_attn_kv_dim = self.qformer_hidden_dim

        self.max_context = int(max_context)
        self.max_horizon = int(max_horizon)
        self.normalize_inputs = bool(normalize_inputs)
        self.force_flip_invariance = bool(force_flip_invariance)
        self.infer_is_positive = bool(infer_is_positive)
        self.fix_quantile_crossing = bool(fix_quantile_crossing)



@dataclass
class TSForecasterOutput:
    """Output of :class:`TSForecaster`.

    Attributes:
        point_forecast: list of (horizon_i, C_i) per-sample median forecasts.
        quantile_forecast: list of (horizon_i, C_i, 10) per-sample quantile
            forecasts (quantile order: [median, 0.1, 0.2, ..., 0.9]).
        predicted_horizon: (B,) long tensor of horizons predicted by the horizon
            head, or None when a horizon override was given.
    """

    point_forecast: list[torch.Tensor] | None = None
    quantile_forecast: list[torch.Tensor] | None = None
    predicted_horizon: torch.Tensor | None = None


class TSForecaster(nn.Module):
    """Standalone TimeOmni_v2 forecaster (cross-attention Forecaster head).

    Inputs (see :meth:`forward`): the raw multi-channel ``history``, plus the two
    precomputed embedding streams ``llm_embedding_input`` and
    ``ts_encoder_embedding_input``. The LLM and TS encoder themselves are NOT part
    of this model.
    """

    def __init__(
        self,
        config: TSForecasterConfig,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        self.aligner = Aligner(config, dtype=dtype, device=device)
        self.forecaster = ForecasterBackbone(
            cross_attn_kv_dim=config.cross_attn_kv_dim,
            dtype=dtype,
            device=device,
        )

        if config.max_context % self.forecaster.p != 0:
            config.max_context = math.ceil(config.max_context / self.forecaster.p) * self.forecaster.p
        if config.max_horizon % self.forecaster.o != 0:
            config.max_horizon = math.ceil(config.max_horizon / self.forecaster.o) * self.forecaster.o
        if config.max_context + config.max_horizon > self.forecaster.context_limit:
            raise ValueError('Context + horizon must be less than the context limit.'
                             f" {config.max_context} + {config.max_horizon} > {self.forecaster.context_limit}.")
        if config.max_horizon > self.forecaster.os:
            raise ValueError(f"Continuous quantile head is not supported for horizons > {self.forecaster.os}.")

        self._horizon_max_length = (int(config.horizon_max_length) or int(config.max_horizon))

    def _forecaster_compiled_decode(
        self,
        horizon,
        inputs,
        masks,
        cross_kv=None,
    ):
        """Single-series decode with revin / flip-invariance / quantile post-
        proc."""
        fc = self.config
        module = self.forecaster
        if horizon > fc.max_horizon:
            raise ValueError(f"Horizon must be less than the max horizon. {horizon} > {fc.max_horizon}.")

        with torch.no_grad():
            target_device = next(module.parameters()).device
            inputs = torch.as_tensor(inputs, dtype=torch.float32, device=target_device)
            masks = torch.as_tensor(masks, dtype=torch.bool, device=target_device)
            batch_size = inputs.shape[0]

            cross_kv_t = None
            if cross_kv is not None:
                cross_kv_t = torch.as_tensor(cross_kv, dtype=torch.float32, device=target_device)
                if cross_kv_t.shape[0] != batch_size:
                    raise ValueError('Batch size mismatch between cross_kv and inputs:'
                                     f" {cross_kv_t.shape[0]} != {batch_size}")

            if fc.infer_is_positive:
                is_positive = torch.all(inputs >= 0, dim=-1, keepdim=True)
            else:
                is_positive = None

            if fc.normalize_inputs:
                mu = torch.mean(inputs, dim=-1, keepdim=True)
                sigma = torch.std(inputs, dim=-1, keepdim=True)
                inputs = revin(inputs, mu, sigma, reverse=False)
            else:
                mu, sigma = None, None

            pf_outputs, quantile_spreads, ar_outputs = module.decode(
                horizon,
                inputs,
                masks,
                cross_kv=cross_kv_t,
            )
            to_cat = [pf_outputs[:, -1, ...]]
            if ar_outputs is not None:
                to_cat.append(ar_outputs.reshape(batch_size, -1, module.q))
            full_forecast = torch.cat(to_cat, dim=1)

            def flip_quantile_fn(x):
                return torch.cat([x[..., :1], torch.flip(x[..., 1:], dims=(-1, ))], dim=-1)

            if fc.force_flip_invariance:
                flipped_pf_outputs, flipped_quantile_spreads, flipped_ar_outputs = module.decode(
                    horizon,
                    -inputs,
                    masks,
                    cross_kv=cross_kv_t,
                )
                flipped_quantile_spreads = flip_quantile_fn(flipped_quantile_spreads)
                flipped_pf_outputs = flip_quantile_fn(flipped_pf_outputs)
                to_cat = [flipped_pf_outputs[:, -1, ...]]
                if flipped_ar_outputs is not None:
                    to_cat.append(flipped_ar_outputs.reshape(batch_size, -1, module.q))
                flipped_full_forecast = torch.cat(to_cat, dim=1)
                quantile_spreads = (quantile_spreads - flipped_quantile_spreads) / 2
                pf_outputs = (pf_outputs - flipped_pf_outputs) / 2
                full_forecast = (full_forecast - flipped_full_forecast) / 2

            for quantile_index in [1, 2, 3, 4, 6, 7, 8, 9]:
                full_forecast[:, :horizon,
                              quantile_index] = (quantile_spreads[:, :horizon, quantile_index] -
                                                 quantile_spreads[:, :horizon, 5] + full_forecast[:, :horizon, 5])
            full_forecast = full_forecast[:, :horizon, :]

            if fc.fix_quantile_crossing:
                for i in [4, 3, 2, 1]:
                    full_forecast[:, :, i] = torch.where(
                        full_forecast[:, :, i] < full_forecast[:, :, i + 1],
                        full_forecast[:, :, i],
                        full_forecast[:, :, i + 1],
                    )
                for i in [6, 7, 8, 9]:
                    full_forecast[:, :, i] = torch.where(
                        full_forecast[:, :, i] > full_forecast[:, :, i - 1],
                        full_forecast[:, :, i],
                        full_forecast[:, :, i - 1],
                    )

            if fc.normalize_inputs:
                full_forecast = revin(full_forecast, mu, sigma, reverse=True)

            if is_positive is not None:
                full_forecast = torch.where(
                    is_positive[..., None],
                    torch.maximum(full_forecast, torch.zeros_like(full_forecast)),
                    full_forecast,
                )

        return full_forecast[..., 5], full_forecast

    def _forecaster_forecast(
        self,
        horizon: int,
        inputs: list,
        cross_kv: list | None = None,
    ):
        """Batched forecast over a list of 1-D series (ported from
        Forecaster_2p5)."""
        max_context = self.config.max_context
        num_inputs = len(inputs)
        if cross_kv is not None and len(cross_kv) != num_inputs:
            raise ValueError(f"cross_kv length must match inputs length: {len(cross_kv)} != {num_inputs}")

        output_points = []
        output_quantiles = []
        for input_index, each_input in enumerate(inputs):
            value = torch.as_tensor(each_input, dtype=torch.float32).reshape(-1)
            if value.numel() > 0:
                value_np = linear_interpolation(strip_leading_nans(value.detach().cpu().numpy()))
                value = torch.from_numpy(value_np).to(dtype=torch.float32)

            if value.numel() > max_context:
                value = value[-max_context:]

            context = max(
                self.forecaster.p,
                math.ceil(value.numel() / self.forecaster.p) * self.forecaster.p,
            )
            context = min(context, max_context)

            value_len = value.numel()
            if value_len >= context:
                value = value[-context:]
                mask = torch.zeros(context, dtype=torch.bool)
            else:
                pad_len = context - value_len
                value = torch.cat([torch.zeros(pad_len, dtype=torch.float32), value], dim=0)
                mask = torch.cat(
                    [torch.ones(pad_len, dtype=torch.bool),
                     torch.zeros(value_len, dtype=torch.bool)],
                    dim=0,
                )

            values_t = value.unsqueeze(0)
            masks_t = mask.unsqueeze(0)
            cross_kv_t = None
            if cross_kv is not None:
                cross_kv_t = torch.as_tensor(
                    cross_kv[input_index],
                    dtype=torch.float32,
                    device=value.device,
                ).unsqueeze(0)

            point_forecast, quantile_forecast = self._forecaster_compiled_decode(
                horizon,
                values_t,
                masks_t,
                cross_kv=cross_kv_t,
            )
            output_points.append(point_forecast)
            output_quantiles.append(quantile_forecast)

        output_points = torch.cat(output_points, dim=0)
        output_quantiles = torch.cat(output_quantiles, dim=0)
        return output_points[:num_inputs], output_quantiles[:num_inputs]

    @staticmethod
    def _flatten_channelwise(history: list[Tensor], ctx: Tensor):
        """Split each (T, C) history into C single-channel series; replicate
        the per-sample cross-attn context once per channel (Forecaster
        forecasts a single univariate series at a time)."""
        channel_counts = []
        flattened_inputs = []
        flattened_prefix = []
        for sample_idx, ts in enumerate(history):
            if ts.dim() != 2:
                raise ValueError(f"Each history series must be 2D (T, C), got shape {tuple(ts.shape)}")
            channels = ts.shape[1]
            channel_counts.append(channels)
            for channel_idx in range(channels):
                flattened_inputs.append(ts[:, channel_idx])
                flattened_prefix.append(ctx[sample_idx])
        return channel_counts, flattened_inputs, flattened_prefix

    def forward(
        self,
        history: list[Tensor],
        llm_embedding_input: Tensor,
        ts_encoder_embedding_input: Tensor,
        llm_embedding_mask: Tensor | None = None,
        ts_encoder_embedding_mask: Tensor | None = None,
        override_horizon: int | Sequence[int] | None = None,
    ) -> TSForecasterOutput:
        """Forecast multi-channel time series.

        Args:
            history: list of B per-sample 2-D tensors (T_i, C_i), the numeric
                forecast context fed to Forecaster. T and C may differ across samples.
            llm_embedding_input: (B, T_llm, d_llm) precomputed LLM hidden states.
            ts_encoder_embedding_input: (B, T_ts, d_ts_encoder) precomputed TS-encoder
                hidden states.
            llm_embedding_mask / ts_encoder_embedding_mask: optional (B, T) bool masks
                (True = valid). Default: all valid.
            override_horizon: optional forecast horizon. An int applies to every
                sample; a sequence gives one horizon per sample. When omitted, the
                horizon head predicts it.
        Returns:
            :class:`TSForecasterOutput` with per-sample ``point_forecast`` /
            ``quantile_forecast`` lists and the predicted horizon.
        """
        if not isinstance(history, (list, tuple)):
            raise ValueError('history must be a list/tuple of (T, C) tensors, one per sample')
        batch_size = len(history)
        if llm_embedding_input.size(0) != batch_size or ts_encoder_embedding_input.size(0) != batch_size:
            raise ValueError('Batch size mismatch: len(history)='
                             f"{batch_size}, llm_embedding_input={llm_embedding_input.size(0)},"
                             f" ts_encoder_embedding_input={ts_encoder_embedding_input.size(0)}")

        ctx, llm_chunk = self.aligner(
            llm_embedding_input,
            ts_encoder_embedding_input,
            llm_embedding_mask=llm_embedding_mask,
            ts_encoder_embedding_mask=ts_encoder_embedding_mask,
        )

        predicted_horizon = None
        if override_horizon is not None:
            if isinstance(override_horizon, int):
                per_sample_horizons = [int(override_horizon)] * batch_size
            else:
                per_sample_horizons = [int(h) for h in override_horizon]
        else:
            pred_log_horizon = self.aligner.predict_horizon(llm_chunk)
            predicted_horizon = self.aligner.decode_horizon(
                pred_log_horizon,
                min_h=1,
                max_h=self._horizon_max_length,
            )
            per_sample_horizons = [int(v) for v in predicted_horizon.tolist()]

        if len(per_sample_horizons) != batch_size:
            raise ValueError(f"per-sample horizon count ({len(per_sample_horizons)}) does not match"
                             f" batch size ({batch_size})")
        # Decode once at max horizon, then slice per sample.
        horizon = max(per_sample_horizons) if per_sample_horizons else 1

        channel_counts, flattened_inputs, flattened_prefix = (self._flatten_channelwise(history, ctx))
        flattened_prefix = [p.to(dtype=torch.float32) for p in flattened_prefix]

        point_flat, quantile_flat = self._forecaster_forecast(
            horizon=horizon,
            inputs=flattened_inputs,
            cross_kv=flattened_prefix,
        )

        point_outputs = []
        quantile_outputs = []
        cursor = 0
        for sample_idx, channels in enumerate(channel_counts):
            sample_horizon = max(1, min(int(per_sample_horizons[sample_idx]), horizon))
            channel_point = []
            channel_quantile = []
            for _ in range(channels):
                channel_point.append(point_flat[cursor, :sample_horizon])
                channel_quantile.append(quantile_flat[cursor, :sample_horizon, :])
                cursor += 1
            point_outputs.append(torch.stack(channel_point, dim=1))
            quantile_outputs.append(torch.stack(channel_quantile, dim=1))

        return TSForecasterOutput(
            point_forecast=point_outputs,
            quantile_forecast=quantile_outputs,
            predicted_horizon=predicted_horizon,
        )


def maybe_forecast_on_ts_gen(
    forecaster: TSForecaster,
    next_token_ids: torch.Tensor,
    *,
    history: list[Tensor],
    llm_embedding_input: Tensor,
    ts_encoder_embedding_input: Tensor,
    llm_embedding_mask: Tensor | None = None,
    ts_encoder_embedding_mask: Tensor | None = None,
    override_horizon: int | Sequence[int] | None = None,
    ts_gen_token_id: int = TS_GEN_TOKEN_ID,
) -> tuple[TSForecasterOutput | None, torch.Tensor]:
    """Run the forecast branch when the generated token is the TS trigger.

    This is a model-local helper for the first integration pass. The token id is test-only until the real model config
    exposes a checkpoint-specific value.
    """
    stop_mask = next_token_ids.reshape(-1).eq(ts_gen_token_id)
    if not stop_mask.any():
        return None, stop_mask

    output = forecaster(
        history=history,
        llm_embedding_input=llm_embedding_input,
        ts_encoder_embedding_input=ts_encoder_embedding_input,
        llm_embedding_mask=llm_embedding_mask,
        ts_encoder_embedding_mask=ts_encoder_embedding_mask,
        override_horizon=override_horizon,
    )
    return output, stop_mask


__all__ = [
    'TS_GEN_TOKEN_ID',
    'TSForecasterConfig',
    'TSForecaster',
    'TSForecasterOutput',
    'Aligner',
    'ForecasterBackbone',
    'QFormer',
    'maybe_forecast_on_ts_gen',
]
