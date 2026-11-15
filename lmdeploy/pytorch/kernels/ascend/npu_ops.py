# Copyright (c) 2024, DeepLink. All rights reserved.
import math

import torch
import torch_npu

from lmdeploy.pytorch.kernels.ascend.types import (
    MoECommType,
    MoeMetadata,
    Optional,
    Sequence,
    Tensor,
)

from . import moe
from .attention import decode_attention, decode_attention_mla
from .utils import SocVersion

__all__ = [
    'add_rms_norm',
    'apply_rotary_pos_emb',
    'apply_rotary_pos_emb_interleaved',
    'lightning_indexer',
    'sparse_flash_attention',
    'prefill_attention',
    'incre_flash_attention',
    'fill_kv_cache',
    'paged_decode_attention',
    'paged_prefill_attention',
    'rms_norm',
    'moe_gating_topk_softmax',
    'get_cache_len',
    'weight_quant_matmul',
    'fused_moe',
    'fused_moe_w8a8',
    'linear',
    'rms_norm_w8a8',
    'add_rms_norm_w8a8',
    'dynamic_quant',
    'linear_w8a8',
    'linear_w8a8_static',
]


def rms_norm_w8a8(
    hidden_states: Tensor, weight: Tensor, epsilon: float, quant_dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    hidden_states = hidden_states.contiguous()
    output = torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]
    x, scale = torch.ops.npu.npu_dynamic_quant(output, dst_type=quant_dtype)
    return x, scale


def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    hidden_states = hidden_states.contiguous()
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    x, scale = torch.ops.npu.npu_dynamic_quant(normed_hidden_states, dst_type=quant_dtype)
    return x, scale, added_hidden_states


def dynamic_quant(hidden_states: Tensor, quant_dtype: torch.dtype, quant_granularity: str) -> tuple[Tensor, Tensor]:
    assert quant_granularity == 'PER_TOKEN'
    x, scale = torch.ops.npu.npu_dynamic_quant(hidden_states, dst_type=quant_dtype)
    return x, scale


def linear_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    rms_scale: torch.Tensor,
    linear_scale: torch.Tensor,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    bias: Tensor,
) -> Tensor:

    out_dtype = torch.bfloat16 if out_dtype == torch.float16 else out_dtype
    output_shape = (*hidden_states.shape[:-1], weight.shape[0])
    hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
    linear_scale = linear_scale.reshape(-1)
    rms_scale = rms_scale.reshape(-1)

    output = torch.ops.npu.npu_quant_matmul(
        hidden_states,
        weight.t(),
        linear_scale,
        pertoken_scale=rms_scale,
        bias=bias,
        output_dtype=out_dtype,
    )
    return output.reshape(output_shape)


def linear_w8a8_static(
    hidden_states: Tensor,
    weight: Tensor,
    input_scale: Tensor,
    input_offset: Tensor,
    deq_scale: Tensor,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    quant_bias: Optional[Tensor],
) -> Tensor:
    """ModelSlim static per-tensor activation / per-channel weight W8A8."""
    out_dtype = torch.bfloat16 if out_dtype == torch.float16 else out_dtype
    input_size = hidden_states.shape[-1]
    reciprocal_scale = input_scale.reciprocal().reshape(-1).expand(input_size)
    offset = input_offset.to(input_scale.dtype).reshape(-1).expand(input_size)
    quantized = torch.ops.npu.npu_quantize(
        hidden_states,
        reciprocal_scale,
        offset,
        torch.qint8,
        -1,
        False,
    )
    return torch.ops.npu.npu_quant_matmul(
        quantized,
        weight.t(),
        deq_scale.reshape(-1),
        bias=quant_bias,
        output_dtype=out_dtype,
    )


def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    normed_hidden_states, _, added_hidden_states = torch.ops.npu.npu_add_rms_norm(
        hidden_states, residual, weight, epsilon
    )
    return normed_hidden_states, added_hidden_states


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    # rotary pos emb helpers:
    query = query.contiguous().unsqueeze(0)
    key = key.contiguous().unsqueeze(0)
    assert len(query.shape) == 4
    batch, seq_len, _, head_size = query.shape
    rotary_dim = cos.size(-1)

    cos = cos.reshape(batch, seq_len, 1, -1)
    sin = sin.reshape(batch, seq_len, 1, -1)

    def rotate_half_(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb_(q, k, cos, sin):
        return (q * cos) + (rotate_half_(q) * sin), (k * cos) + (rotate_half_(k) * sin)

    # ascend ops currently only support dim 128
    if rotary_dim == head_size:
        if head_size == 128:
            return torch.ops.npu.npu_apply_rotary_pos_emb(query, key, cos, sin, 'BSND')
        else:
            return apply_rotary_pos_emb_(query, key, cos, sin)
    elif rotary_dim < head_size:
        q_rot = query[..., :rotary_dim].contiguous()
        q_pass = query[..., rotary_dim:]
        k_rot = key[..., :rotary_dim].contiguous()
        k_pass = key[..., rotary_dim:]

        q_rot, k_rot = torch.ops.npu.npu_apply_rotary_pos_emb(q_rot, k_rot, cos, sin, 'BSND')

        q = torch.cat((q_rot, q_pass), dim=-1)
        k = torch.cat((k_rot, k_pass), dim=-1)
        return q, k
    else:
        raise RuntimeError('apply_rotary_pos_emb does not support rotary_dim > head_size')


def apply_rotary_pos_emb_interleaved(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    return_native_layout: bool = True,
) -> Tensor:
    """Apply adjacent-pair RoPE with the native Ascend interleave operator.

    Return the vendor's native front/back-half layout by default. Set return_native_layout=False to restore adjacent
    pairs.
    """
    if x.ndim != 4 or cos.ndim != 4 or sin.ndim != 4:
        raise ValueError(
            f"npu_interleave_rope expects 4D x/cos/sin tensors, got {x.ndim}D, {cos.ndim}D and {sin.ndim}D"
        )
    output = torch_npu.npu_interleave_rope(x.contiguous(), cos.contiguous(), sin.contiguous())
    if return_native_layout:
        return output

    # The native op returns rotated even and odd elements in separate halves.
    output_even, output_odd = output.chunk(2, dim=-1)
    return torch.stack((output_even, output_odd), dim=-1).flatten(-2)


def lightning_indexer(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    actual_seq_lengths_query: Optional[Tensor],
    actual_seq_lengths_key: Optional[Tensor],
    block_table: Optional[Tensor],
    sparse_count: int,
) -> Tensor:
    """BF16 Lightning Indexer backed by the native torch-npu operator."""
    indices, _ = torch_npu.npu_lightning_indexer(
        query=query.contiguous(),
        key=key,
        weights=weights.contiguous(),
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        layout_query='TND',
        layout_key='PA_BSND',
        sparse_count=sparse_count,
        sparse_mode=3,
    )
    return indices


def sparse_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    block_table: Optional[Tensor],
    actual_seq_lengths_query: Optional[Tensor],
    kv_seqlens: Optional[Tensor],
    query_rope: Optional[Tensor],
    key_rope: Optional[Tensor],
) -> Tensor:
    """BF16 sparse flash attention backed by the native torch-npu operator."""
    if query.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"sparse_flash_attention expects BF16/FP16 query, got {query.dtype}")

    output, _, _ = torch_npu.npu_sparse_flash_attention(
        query=query.contiguous(),
        key=key,
        value=value,
        sparse_indices=sparse_indices.contiguous(),
        scale_value=scale_value,
        block_table=block_table,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=kv_seqlens,
        query_rope=None if query_rope is None else query_rope.contiguous(),
        key_rope=key_rope,
        sparse_block_size=1,
        layout_query='TND',
        layout_kv='PA_BSND',
        sparse_mode=3,
        attention_mode=2,
    )
    return output


def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    q_seq_len: Tensor,
    max_q_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    actual_seq_lengths_cpu: Optional[Tensor],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError('paged_decode_attention does not support alibi_slopes yet')

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attn_output = attn_output.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    if len(attn_mask):
        mask = attn_mask[0]
    else:
        # Handle qwenvl vision part flash-attention
        is_tnd = query.dim() == 3 and query.shape[-2] == num_q_heads and key.shape[-2] == num_kv_heads
        input_layout = 'TND' if is_tnd else 'BSH'
        if is_tnd and actual_seq_lengths_cpu is None:
            raise ValueError('actual_seq_lengths_cpu is required for TND prefill attention')
        if not is_tnd:
            actual_seq_lengths_cpu = None
        fia_kwargs = {}
        if is_tnd and query.shape[-1] > value.shape[-1]:
            nope_dim = value.shape[-1]
            fia_kwargs['query_rope'] = query[..., nope_dim:].contiguous()
            fia_kwargs['key_rope'] = key[..., nope_dim:].contiguous()
            query = query[..., :nope_dim].contiguous()
            key = key[..., :nope_dim].contiguous()
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            input_layout=input_layout,
            actual_seq_lengths=actual_seq_lengths_cpu,
            actual_seq_lengths_kv=actual_seq_lengths_cpu,
            scale=scale_value,
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            sparse_mode=0,
            **fia_kwargs,
        )
        attn_output.copy_(output)
        return attn_output
    if SocVersion.is_Ascend910():
        # The backend supplies the fixed split-fuse causal mask required by
        # sparse mode 3 for both standard attention and MLA.
        fia_kwargs = {}
        if query.shape[-1] > value.shape[-1]:
            # MLA concatenates the NOPE and ROPE parts in Q/K. FIA accepts
            # large MLA head dimensions only when the ROPE part is separate.
            nope_dim = value.shape[-1]
            fia_kwargs['query_rope'] = query[..., nope_dim:].contiguous()
            fia_kwargs['key_rope'] = key[..., nope_dim:].contiguous()
            query = query[..., :nope_dim].contiguous()
            key = key[..., :nope_dim].contiguous()

        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=mask,
            input_layout='TND',
            actual_seq_lengths=actual_seq_lengths_cpu,
            actual_seq_lengths_kv=actual_seq_lengths_cpu,
            scale=scale_value,
            num_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            sparse_mode=3,
            **fia_kwargs,
        )
        attn_output.copy_(output)
    else:
        raise ValueError(f"LMDeploy Ascend doesn't support {SocVersion.device_name()} device currently.")
    return attn_output


def incre_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    num_heads: int,
    input_layout: str,
    softmax_scale: Optional[float],
) -> Tensor:
    attn_output = torch.ops.npu.npu_incre_flash_attention(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout=input_layout,
        scale_value=softmax_scale,
    )
    return attn_output


# atb._npu_reshape_and_cache has a performace advantage of about 3% compared to npu.npu_scatter_nd_update_,
# but atb._npu_reshape_and_cache will report an error when slot_indices is an empty tensor.
"""
def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]],
    v_scales_zeros: Sequence[Optional[Tensor]],
    quant_bits: int,
) -> tuple[Tensor, Tensor]:
    _, head, dim = key.shape
    block_num, block_size = key_cache.shape[:2]
    block_total = block_num * block_size

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()
    kv_indices = kv_indices.view(-1, 1)

    if quant_bits == 8:

        def quant_int8(x, x_scale, x_offset):
            quantized = (
                ((x / x_scale) - x_offset).round().clamp(-128, 127).to(torch.int8)
            )
            return quantized

        key = quant_int8(key, k_scales_zeros[0], k_scales_zeros[1])
        value = quant_int8(value, v_scales_zeros[0], v_scales_zeros[1])

    is_mla = key.shape[-1] != value.shape[-1]
    if is_mla:
        key_cache_reshaped = key_cache.view(block_total, head, dim)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    else:
        key_cache_reshaped = key_cache.view(block_total, head, dim)
        value_cache_reshaped = value_cache.view(block_total, head, dim)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
        torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value)
    return key_cache, value_cache
"""


def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]],
    v_scales_zeros: Sequence[Optional[Tensor]],
    quant_bits: int,
) -> tuple[Tensor, Tensor]:
    split_mla_cache = (
        key_cache.dim() == 4
        and value_cache.dim() == 4
        and key_cache.size(-1) > 0
        and value_cache.size(-1) > 0
        and key.size(-1) == key_cache.size(-1) + value_cache.size(-1)
        and value.size(-1) == value_cache.size(-1)
    )
    if split_mla_cache:
        # key_cache -> rope_cache, value_cache -> nope_cache
        rope_head_size = key_cache.size(-1)
        key_rope = key[..., -rope_head_size:].contiguous()
        value_nope = value.contiguous()
        key_cache_reshaped = torch.flatten(key_cache, start_dim=0, end_dim=1)
        value_cache_reshaped = torch.flatten(value_cache, start_dim=0, end_dim=1)
        kv_indices = kv_indices.view(-1, 1)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key_rope)
        torch.ops.npu.npu_scatter_nd_update_(value_cache_reshaped, kv_indices, value_nope)
        return key_cache, value_cache

    # only support contiguous k,v
    key = key.contiguous()
    value = value.contiguous()

    if quant_bits == 8:

        def quant_int8(x, x_scale, x_offset):
            quantized = ((x / x_scale) - x_offset).round().clamp(-128, 127).to(torch.int8)
            return quantized

        key = quant_int8(key, k_scales_zeros[0], k_scales_zeros[1])
        value = quant_int8(value, v_scales_zeros[0], v_scales_zeros[1])

    is_mla = key.shape[-1] != value.shape[-1]
    if is_mla:
        assert len(key_cache.shape) == 4
        key_cache_reshaped = torch.flatten(key_cache, start_dim=0, end_dim=1)
        kv_indices = kv_indices.view(-1, 1)
        torch.ops.npu.npu_scatter_nd_update_(key_cache_reshaped, kv_indices, key)
    else:
        torch.ops.atb._npu_reshape_and_cache(
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
            slot_indices=kv_indices.to(torch.int32),
        )
    return key_cache, value_cache


def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> tuple[Tensor, Tensor]:
    key_cache = torch.cat([key_cache, key_state], dim=1)
    value_cache = torch.cat([value_cache, value_state], dim=1)
    return key_cache, value_cache


def get_cache_len(cache: Tensor):
    return cache.shape[1]


def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Optional[Tensor],
    block_size: int,
    actual_q_seqlens_cpu: Tensor,
    kv_seqlens_cpu: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError('paged_decode_attention does not support alibi_slopes yet')
    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    query = query.contiguous()
    attn_output = attn_output.contiguous()
    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    key_headsize, value_headsize = key_cache.shape[-1], value_cache.shape[-1]
    if key_headsize == value_headsize:
        return decode_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            scale_value=scale_value,
            block_table=block_table,
            block_size=block_size,
            actual_q_seqlens_cpu=actual_q_seqlens_cpu,
            kv_seqlens_cpu=kv_seqlens_cpu,
            softmax_scale=softmax_scale,
            attn_output=attn_output,
        )
    else:
        return decode_attention_mla(
            query=query,
            key_cache=key_cache,
            num_kv_heads=num_kv_heads,
            num_q_heads=num_q_heads,
            scale_value=scale_value,
            block_table=block_table,
            kv_seqlens_cpu=kv_seqlens_cpu,
            mla_vheadsize=value_headsize,
            attn_output=attn_output,
        )


def paged_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    actual_q_seqlens_cpu: Tensor,
    kv_seqlens_cpu: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float],
    alibi_slopes: Optional[Sequence[float]],
    attn_output: Optional[Tensor],
    kv_scales: Optional[Tensor],
    kv_zeros: Optional[Tensor],
    quant_bits: Optional[int],
    head_size_v: Optional[int] = None,
) -> Tensor:
    if alibi_slopes is not None:
        raise RuntimeError('paged_decode_attention does not support alibi_slopes yet')

    if isinstance(block_table, torch.Tensor) and block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)

    scale_value = softmax_scale if softmax_scale else 1.0 / math.sqrt(query.shape[-1])
    query = query.contiguous()

    # lmdeploy's DeepSeek MLA path absorbs W_UK into Q. Its paged cache is
    # therefore [latent K/V, RoPE K], while value_cache is a view of the
    # latent part. Follow vllm-ascend's multi-token paged attention path:
    # feed the latent and RoPE components separately to FIA v2, and use the
    # split-fuse causal mask with sparse mode 3.
    is_mla = key_cache.shape[-1] != value_cache.shape[-1]
    if is_mla:
        num_tokens = query.shape[0]
        mla_vheadsize = head_size_v or value_cache.shape[-1]
        if query.shape[-1] <= mla_vheadsize:
            raise RuntimeError('MLA paged prefill expects query to contain both latent and RoPE parts')

        q_nope = query[..., :mla_vheadsize].contiguous()
        q_rope = query[..., mla_vheadsize:].contiguous()

        # FIA v2 expects paged KV cache in
        # [block, kv_head, block_size, dim].
        key_cache = key_cache.permute(0, 2, 1, 3)
        value_cache = value_cache.permute(0, 2, 1, 3)
        k_nope = key_cache[..., :mla_vheadsize]
        k_rope = key_cache[..., mla_vheadsize:]
        v_nope = value_cache[..., :mla_vheadsize]
        mask = attn_mask[0] if len(attn_mask) else None

        output, _ = torch_npu.npu_fused_infer_attention_score_v2(
            q_nope,
            k_nope,
            v_nope,
            query_rope=q_rope,
            key_rope=k_rope,
            num_query_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
            input_layout='TND_NTD',
            atten_mask=mask,
            sparse_mode=3,
            softmax_scale=scale_value,
            block_table=block_table,
            block_size=block_size,
            actual_seq_qlen=actual_q_seqlens_cpu,
            actual_seq_kvlen=kv_seqlens_cpu,
        )

        # TND_NTD returns [num_heads, num_tokens, value_head_size].
        output = output[:, :num_tokens].transpose(0, 1)
        if attn_output is not None:
            attn_output.copy_(output)
            return attn_output
        return output

    block_num = key_cache.size(0)
    key_cache = key_cache.view(block_num, block_size, -1)
    value_cache = value_cache.view(block_num, block_size, -1)

    output, _ = torch.ops.npu.npu_fused_infer_attention_score(
        query=query,
        key=key_cache,
        value=value_cache,
        atten_mask=attn_mask[0],
        block_table=block_table,
        input_layout='TND',
        block_size=block_size,
        actual_seq_lengths=actual_q_seqlens_cpu,
        actual_seq_lengths_kv=kv_seqlens_cpu,
        num_key_value_heads=num_kv_heads,
        num_heads=num_q_heads,
        scale=scale_value,
        sparse_mode=3,
    )

    if attn_output is not None:
        attn_output.copy_(output)
        return attn_output
    return output


def rms_norm(hidden_states: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    hidden_states = hidden_states.contiguous()
    return torch.ops.npu.npu_rms_norm(hidden_states, weight, epsilon)[0]


def silu_and_mul(input_tensor: Tensor, dim: int) -> Tensor:
    if SocVersion.is_Ascend910():
        return torch.ops.npu.npu_swiglu(input_tensor, dim)
    elif SocVersion.is_Ascend310P():
        gate_cache, up_cache = input_tensor.chunk(2, dim)
        return torch.ops.npu.npu_silu(gate_cache) * up_cache


def moe_gating_topk_softmax(
    router_logits: Tensor,
    topk: int,
    moe_metadata: MoeMetadata,
) -> tuple[Tensor, Tensor]:
    if moe_metadata.ep_size > 1:
        if moe_metadata.pad_size > 0:
            router_logits = torch.nn.functional.pad(router_logits, (0, 0, 0, moe_metadata.pad_size))
        if moe_metadata.tp_size > 1:
            split_router_logits = torch.tensor_split(router_logits, moe_metadata.tp_size, dim=0)
            router_logits = split_router_logits[moe_metadata.tp_rank]
    if moe_metadata.router_n_groups > 0:
        routing_weights, selected_idx, _ = torch.ops.npu.npu_moe_gating_top_k(
            router_logits,
            k=topk,
            bias=None,
            k_group=moe_metadata.router_n_groups,
            group_count=moe_metadata.router_n_groups,
            group_select_mode=0,  # 0: the maximum in the group; 1: topk2.sum
            renorm=0,  # 0: softmax->topk; 1: topk->softmax
            norm_type=0,  # 0: softmax; 1: sigmoid
            out_flag=False,
            routed_scaling_factor=1,
            eps=1e-20,
        )
    else:
        routing_weights, selected_idx, _ = torch.ops.npu.npu_moe_gating_top_k_softmax(router_logits, None, topk)
    return routing_weights, selected_idx


# Quantification of W4A16 is currently supported and tested.
def weight_quant_matmul(
    x: Tensor,
    qweight: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    all_reduce: Optional[bool] = False,
    group_size: Optional[int] = 0,
) -> Tensor:
    offset = None if (offset is None or offset.numel() == 0) else offset
    return torch.ops.npu.npu_weight_quant_batchmatmul(
        x,
        qweight,
        scale,
        antiquant_offset=offset,
        antiquant_group_size=group_size,
        bias=bias,
    )


def fused_moe(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    down_weights: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    moe_metadata: MoeMetadata,
    chunked_moe_layout=None,
) -> Tensor:
    topk_ids = topk_ids.to(torch.int32)
    (
        hidden_states,
        num_tokens,
        paded_num_tokens,
        x_active_mask,
        topk_ids,
        topk_weights,
    ) = moe.moe_prepare(
        hidden_states,
        moe_metadata.x_active_mask,
        moe_metadata.pad_size,
        moe_metadata.tp_size,
        moe_metadata.ep_size,
        moe_metadata.tp_rank,
        moe_metadata.moe_comm_type,
        topk_ids,
        topk_weights,
    )

    if moe_metadata.moe_comm_type == MoECommType.MC2:
        moe_output = moe.fused_moe_mc2(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            renormalize,
            moe_metadata.ep_size,
            moe_metadata.ep_rank,
            moe_metadata.moe_group_name,
            x_active_mask,
            chunked_moe_layout,
        )
    elif moe_metadata.moe_comm_type == MoECommType.ALLTOALL:
        moe_output = moe.fused_moe_all2all(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            renormalize,
            moe_metadata.ep_size,
            moe_metadata.ep_rank,
            moe_metadata.ep_group,
            moe_metadata.expert_ids_per_ep_rank,
            chunked_moe_layout,
        )
    # TODO: fused_moe_allgather
    else:
        moe_output = moe.fused_moe_naive(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            topk,
            renormalize,
            chunked_moe_layout,
        )

    moe_output = moe.moe_finalize(
        moe_output,
        num_tokens,
        paded_num_tokens,
        moe_metadata.ep_size,
        moe_metadata.tp_size,
        moe_metadata.tp_group,
    )

    return moe_output


def fused_moe_w8a8(
    hidden_states: Tensor,
    gate_up_weights: Tensor,
    gate_up_scales: Tensor,
    down_weights: Tensor,
    down_scales: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    topk: int,
    renormalize: bool,
    moe_metadata: MoeMetadata,
) -> Tensor:
    """Dynamic W8A8 MoE using only public torch_npu operators."""
    topk_ids = topk_ids.to(torch.int32)
    (
        hidden_states,
        num_tokens,
        paded_num_tokens,
        x_active_mask,
        topk_ids,
        topk_weights,
    ) = moe.moe_prepare(
        hidden_states,
        moe_metadata.x_active_mask,
        moe_metadata.pad_size,
        moe_metadata.tp_size,
        moe_metadata.ep_size,
        moe_metadata.tp_rank,
        moe_metadata.moe_comm_type,
        topk_ids,
        topk_weights,
    )

    if moe_metadata.moe_comm_type == MoECommType.MC2:
        moe_output = moe.fused_moe_mc2(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            renormalize,
            moe_metadata.ep_size,
            moe_metadata.ep_rank,
            moe_metadata.moe_group_name,
            x_active_mask,
            None,
            gate_up_scales,
            down_scales,
        )
    elif moe_metadata.moe_comm_type == MoECommType.ALLTOALL:
        moe_output = moe.fused_moe_all2all(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            renormalize,
            moe_metadata.ep_size,
            moe_metadata.ep_rank,
            moe_metadata.ep_group,
            moe_metadata.expert_ids_per_ep_rank,
            None,
            gate_up_scales,
            down_scales,
        )
    else:
        moe_output = moe.fused_moe_naive(
            hidden_states,
            gate_up_weights,
            down_weights,
            topk_weights,
            topk_ids,
            topk,
            renormalize,
            None,
            gate_up_scales,
            down_scales,
        )

    return moe.moe_finalize(
        moe_output,
        num_tokens,
        paded_num_tokens,
        moe_metadata.ep_size,
        moe_metadata.tp_size,
        moe_metadata.tp_group,
    )


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    all_reduce: Optional[bool],
    group: Optional[str],
) -> Tensor:
    if all_reduce:
        assert group is None or group == '', 'In eager mode, only use default_pg'
        group = torch.distributed.distributed_c10d._world.default_pg
        hcomm_info = group._get_backend(x.device).get_hccl_comm_name(x.device.index)
        out = torch.ops.npu.npu_mm_all_reduce_base(
            x.contiguous(),
            weight.transpose(0, 1),
            hcomm_info,
            reduce_op='sum',
            bias=bias,
        )
    else:
        # on 310p, the weight is transposed to nz format in llm part on graph mode,
        # but in vl part, eager mode is used.
        # we need to reshape it back to nd.
        if len(weight.shape) == 4 and weight.shape[0] == 1 and weight.shape[1] * weight.shape[3] == x.shape[-1]:
            weight = weight.permute(0, 2, 1, 3)
            weight = weight.reshape(weight.shape[1], -1)
        out = torch.nn.functional.linear(x, weight, bias)
    return out


def transdata(
    hidden_states: Tensor,
    transdata_type: int,
):
    raise NotImplementedError('transdata in eager mode is not implemented yet!')
