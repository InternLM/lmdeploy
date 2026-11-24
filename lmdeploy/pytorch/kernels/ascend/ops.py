# Copyright (c) 2024, DeepLink. All rights reserved.
"""Ascend operator argument normalization, called directly by native
wrappers."""

import torch

from . import npu_ops
from .types import MoeMetadata, Optional, Sequence, Tensor, dynamic_quant_scale_type, linear_w8a8_scale_type


def add_rms_norm(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    """Add the residual connection and then apply Root Mean Square (RMS)
    Normalization.

    Args:
        hidden_states (Tensor): The input tensor to be normalized.
        residual (Tensor): The residual tensor to be added to the hidden states.
        weight (Tensor): The weight tensor used for normalization.
        epsilon (float): A small constant added to the root mean square to prevent division by zero.

    Returns:
        tuple[Tensor, Tensor]:
            - The normalized output tensor.
            - The added result of the residual connection.
    """
    return npu_ops.add_rms_norm(hidden_states, residual, weight, epsilon)


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    cos: Optional[Tensor],
    sin: Optional[Tensor],
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to the query and key tensors.

    Rotary position embedding is a method of embedding positional information into
    self-attention computations without increasing the model size.

    Args:
        query (Tensor): The query tensor to apply the rotary position embeddings to.
        key (Tensor): The key tensor to apply the rotary position embeddings to.
        cos (Optional[Tensor]): The cosine component of the rotary position embeddings.
        sin (Optional[Tensor]): The sine component of the rotary position embeddings.

    Returns:
        tuple[Tensor, Tensor]:
            - The query tensor with the rotary position embeddings applied.
            - The key tensor with the rotary position embeddings applied.
    """
    return npu_ops.apply_rotary_pos_emb(
        query,
        key,
        cos,
        sin,
    )


def apply_rotary_pos_emb_interleaved(x: Tensor, cos: Tensor, sin: Tensor, return_native_layout: bool = True) -> Tensor:
    """Apply complex RoPE to adjacent element pairs.

    By default, the output keeps the native vendor front/back-half layout. Set return_native_layout=False to restore the
    adjacent-pair layout. The cos and sin tables use the front/back-half layout required by the vendor implementation.
    """
    return npu_ops.apply_rotary_pos_emb_interleaved(x, cos, sin, return_native_layout)


def lightning_indexer(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_key: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    sparse_count: int = 2048,
) -> Tensor:
    """Select causal sparse-attention token indices from a paged key cache."""
    return npu_ops.lightning_indexer(
        query,
        key,
        weights,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        block_table,
        sparse_count,
    )


def sparse_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    block_table: Optional[Tensor] = None,
    actual_seq_lengths_query: Optional[Tensor] = None,
    kv_seqlens: Optional[Tensor] = None,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
) -> Tensor:
    """Run sparse flash attention using logical token indices."""
    return npu_ops.sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        block_table,
        actual_seq_lengths_query,
        kv_seqlens,
        query_rope,
        key_rope,
    )


def prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    max_q_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    softmax_scale: Optional[float] = None,
    alibi_slopes: Optional[Sequence[float]] = None,
    attn_output: Optional[Tensor] = None,
    actual_seq_lengths_cpu: Optional[Tensor] = None,
) -> Tensor:
    """Computes the multi-head attention over the query, key, and value
    tensors. This interface is used for prefilling stage in LLM inference
    without paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.
        q_seq_len (Tensor): The length of each query sequence.
        kv_seq_len (Tensor): The length of each key/value sequence.
        max_q_seq_len (int): The maximum length of any query sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        attn_mask (Sequence[Optional[Tensor]]): A sequence of optional attention masks, one for each batch.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.
        actual_seq_lengths_cpu (Optional[Tensor]): Precomputed cumulative end
            position of each query sequence on CPU. It is required for TND
            attention.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return npu_ops.prefill_attention(
        query,
        key,
        value,
        q_seq_len,
        max_q_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        softmax_scale,
        alibi_slopes,
        attn_output,
        actual_seq_lengths_cpu,
    )


def incre_flash_attention(
    query: Tensor, key: Tensor, value: Tensor, num_heads: int, input_layout: str, softmax_scale: Optional[float] = None
) -> Tensor:
    """Computes the multi-head attention over the query, key, and value
    tensors. This interface is used for prefilling stage in LLM inference
    without paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key (Tensor): The key tensor.
        value (Tensor): The value tensor.
        num_heads (int): The number of query heads.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.

    Returns:
        Tensor: The computed attention output tensor.
    """
    return npu_ops.incre_flash_attention(
        query,
        key,
        value,
        num_heads,
        input_layout,
        softmax_scale,
    )


def fill_kv_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    kv_indices: Tensor,
    k_scales_zeros: Sequence[Optional[Tensor]] = tuple(),
    v_scales_zeros: Sequence[Optional[Tensor]] = tuple(),
    quant_bits: int = 0,
) -> tuple[Tensor, Tensor]:
    """Fills the key-value cache with the provided key and value tensors.

    Args:
        key (Tensor): The key tensor to be stored in the cache.
        value (Tensor): The value tensor to be stored in the cache.
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.
        kv_indices (Tensor): The indices specifying where to store the key and value in the cache.
        k_scales_zeros (Sequence[Optional[Tensor]]): The scales and zeros used to quantify key.
        v_scales_zeros (Sequence[Optional[Tensor]]): The scales and zeros used to quantify value.
        quant_bits (int): The bits which k/v is quantized into.

    Returns:
        tuple[Tensor, Tensor]:
            - The updated key cache tensor.
            - The updated value cache tensor.
    """
    return npu_ops.fill_kv_cache(
        key,
        value,
        key_cache,
        value_cache,
        kv_indices,
        k_scales_zeros,
        v_scales_zeros,
        quant_bits,
    )


def paged_decode_attention(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size_v: Optional[int] = 0,
    softmax_scale: Optional[float] = None,
    alibi_slopes: Optional[Sequence[float]] = None,
    attn_output: Optional[Tensor] = None,
    kv_scales: Optional[Tensor] = None,
    kv_zeros: Optional[Tensor] = None,
    quant_bits: Optional[int] = 0,
) -> Tensor:
    """Computes the multi-head attention over the query, key, and value
    tensors. This interface is used for decoding stage in LLM inference with
    paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key_cache (Tensor): The cacheed key tensor.
        value_cache (Tensor): The cached value tensor.
        block_table (Tensor): A tensor that maps each position in the query sequence to the corresponding
                              block in the key/value cache.
        block_size (int): The size of each block in the input sequence.
        q_seq_len (Tensor): The length of query sequence.
        kv_seq_len (Tensor): The length of each key/value sequence.
        max_kv_seq_len (int): The maximum length of any key/value sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        head_size_v (int): The number of value head size.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.
        kv_scales (Optional[Tensor]): The quantization factors for key and value.
        kv_zeros (Optional[Tensor]): The quantization offset for key and value.
        quant_bits (Optional[int]): The bits which k/v is quantized into.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return npu_ops.paged_decode_attention(
        query,
        key_cache,
        value_cache,
        block_table,
        block_size,
        q_seq_len,
        kv_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        softmax_scale,
        alibi_slopes,
        attn_output,
        kv_scales,
        kv_zeros,
        quant_bits,
    )


def paged_prefill_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    block_size: int,
    q_seq_len: Tensor,
    kv_seq_len: Tensor,
    max_q_seq_len: int,
    max_kv_seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: Sequence[Optional[Tensor]],
    head_size_v: Optional[int] = 0,
    softmax_scale: Optional[float] = None,
    alibi_slopes: Optional[Sequence[float]] = None,
    attn_output: Optional[Tensor] = None,
    kv_scales: Tensor = None,
    kv_zeros: Tensor = None,
    quant_bits: int = 0,
) -> Tensor:
    """Computes the multi-head attention over the query, key, and value
    tensors. This interface is used for prefilling stage in LLM inference with
    paged kv-cache.

    Args:
        query (Tensor): The query tensor.
        key_cache (Tensor): The cacheed key tensor.
        value_cache (Tensor): The cached value tensor.
        block_table (Tensor): A tensor that maps each position in the query sequence to the corresponding
                              block in the key/value cache.
        block_size (int): The size of each block in the input sequence.
        q_seq_len (Tensor): The length of each query sequence.
        kv_seq_len (Tensor): The length of each key/value sequence.
        max_q_seq_len (int): The maximum length of any query sequence.
        max_kv_seq_len (int): The maximum length of any key/value sequence.
        num_q_heads (int): The number of query heads.
        num_kv_heads (int): The number of key/value heads.
        attn_mask (Sequence[Optional[Tensor]]): A sequence of optional attention masks, one for each batch.
        head_size_v (int): The number of value head size.
        softmax_scale (Optional[float]): The scale factor to apply to the attention logits before the softmax.
        alibi_slopes (Optional[Sequence[float]]): The slopes for the ALiBi attention bias, one for each head.
        attn_output (Optional[Tensor]): The computed attention output tensor.
        kv_scales (Optional[Tensor]): The quantization factors for key and value.
        kv_zeros (Optional[Tensor]): The quantization offset for key and value.
        quant_bits (Optional[int]): The bits which k/v is quantized into.

    Returns:
        Tensor: The computed attention output tensor, alias of attn_output.
    """
    return npu_ops.paged_prefill_attention(
        query,
        key,
        value,
        key_cache,
        value_cache,
        block_table,
        block_size,
        q_seq_len,
        kv_seq_len,
        max_q_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        attn_mask,
        softmax_scale,
        alibi_slopes,
        attn_output,
        kv_scales,
        kv_zeros,
        quant_bits,
        head_size_v,
    )


def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    """Apply Root Mean Square (RMS) Normalization to the input.

    Args:
        hidden_states (Tensor): The input tensor to be normalized.
        weight (Tensor): The weight tensor used for normalization.
        epsilon (float): A small constant added to the root mean square to prevent division by zero.

    Returns:
        Tensor: The normalized output tensor.
    """
    return npu_ops.rms_norm(hidden_states, weight, epsilon)


def silu_and_mul(input_tensor: Tensor, dim: int = -1) -> Tensor:
    """Apply silu activation on the first half part of input tensor along dim,
    and then do elementwise mul between the activated result and second half
    part of input tensor along dim.

    Args:
        input_tensor (Tensor): The input tensor to be apply silu and mul activation.
        dim (int): The axis that we would split input tensor into two parts.

    Returns:
        Tensor: The activated output tensor.
    """
    return npu_ops.silu_and_mul(input_tensor, dim)


def moe_gating_topk_softmax(
    router_logits: Tensor,
    topk: int,
    moe_metadata: MoeMetadata,
) -> tuple[Tensor, Tensor]:
    """Given router_logits of experts, it computes the probability
    distributions of experts and then selecting topk values and their
    corresponding indices.

    Args:
        router_logits (Tensor): The input router logits of probability.
        topk (int): The number of top experts to select.
        moe_metadata (MoeMetadata): Metadata containing configuration and context for the MoE layer
                                    required by the backend operation.

    Returns:
        tuple[Tensor, Tensor]:
        - The router weight of selected experts.
        - The index of selected experts.
    """
    return npu_ops.moe_gating_topk_softmax(router_logits, topk, moe_metadata)


def fill_contiguous_kvcache(
    key_cache: Tensor, value_cache: Tensor, key_state: Tensor, value_state: Tensor
) -> tuple[Tensor, Tensor]:
    """Fills the key-value cache with the provided key and value tensors in
    contiguous way.

    Args:
        key (Tensor): The key tensor to be stored in the cache.
        value (Tensor): The value tensor to be stored in the cache.
        key_cache (Tensor): The existing key cache tensor.
        value_cache (Tensor): The existing value cache tensor.

    Returns:
        tuple[Tensor, Tensor]:
            - The updated key cache tensor.
            - The updated value cache tensor.
    """
    return npu_ops.fill_contiguous_kvcache(key_cache, value_cache, key_state, value_state)


def get_cache_len(cache: Tensor) -> int:
    """Get the seq length of input cache tensor.

    Args:
        cache (Tensor): The input cache tensor.

    Returns:
        int: the required length
    """
    return npu_ops.get_cache_len(cache)


def weight_quant_matmul(
    x1: Tensor,
    x2: Tensor,
    scale: Tensor,
    offset: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    group_size: Optional[int] = 0,
) -> Tensor:
    """Complete a matrix multiplication computation with quantized scenarios as
    inputs.

    Args:
        x1 (Tensor): The input tensor.
        x2 (Tensor): The quantized weight tensor.
        scale (Tensor): The antiquant scale tensor of quantized weight.
        offset (Optional[Tensor]): An optional antiquant offset tensor of quantized weight.
        bias (Optional[Tensor]): An optional bias tensor of matrix multiplication.
        group_size (Optional[int]): An optional group_size of the quantized weight in the per_group algorithm mode.

    Returns:
        Tensor: The output tensor of the matrix product in the quantisation scenario.
    """
    return npu_ops.weight_quant_matmul(x1, x2, scale, offset, bias, group_size)


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
    """Implement the Fused Mixture of Experts (MoE) model.

    Args:
        hidden_states (Tensor): The hidden_states tensor.
        top_k (int): The number of top K experts selected among multiple experts.
        topk_ids (Tensor): The IDs of the top K selected experts.
        topk_weights (Tensor): The topk_weights tensor corresponds to the weight of experts in topk_ids.
        gate_up_weights (Tensor): The gate_up_weights tensor used to upsample.
        down_weights (Tensor): The down_weights tensor used to downsample.
        renormalize (bool): A boolean flag to indicate whether to renormalize the output.
        moe_metadata (MoeMetadata): Metadata containing configuration and context for the MoE layer
                                    required by the backend operation.

    Returns:
        Tensor: The output tensor of the Fused Mixture of Experts (MoE) model.
    """
    fused_moe_impl = npu_ops.fused_moe
    args = (
        hidden_states,
        gate_up_weights,
        down_weights,
        topk_weights,
        topk_ids,
        topk,
        renormalize,
        moe_metadata,
    )
    if chunked_moe_layout is None:
        return fused_moe_impl(*args)
    return fused_moe_impl(*args, chunked_moe_layout)


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
    """Run dynamic W8A8 MoE through the vendor's non-fused fallback.

    Unlike :func:`fused_moe`, this entry point carries the per-channel weight
    scales required by quantized grouped matmuls.  It deliberately has no
    ``chunked_moe_layout`` argument: GLM-5.2 has fewer than the Ascend grouped
    matmul limit of 1024 (local) experts.
    """
    return npu_ops.fused_moe_w8a8(
        hidden_states,
        gate_up_weights,
        gate_up_scales,
        down_weights,
        down_scales,
        topk_weights,
        topk_ids,
        topk,
        renormalize,
        moe_metadata,
    )


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    all_reduce: Optional[bool] = False,
    group: Optional[str] = '',
) -> Tensor:
    """Complete a linear computation.

    Args:
        x1 (Tensor): The first input tensor of linear computation.
        x2 (Tensor): The second input tensor of linear computation.
        bias (Optional[Tensor]): An optional bias tensor of linear computation.
        all_reduce (Optional[bool]): An optional bool describes whether or not allreduce is required.

    Returns:
        Tensor: The output tensor of linear computation.
    """
    return npu_ops.linear(x, weight, bias, all_reduce, group)


def dynamic_quant(
    x: Tensor, quant_dtype: torch.dtype, quant_granularity: str = None
) -> tuple[Tensor, dynamic_quant_scale_type]:
    """Perform dynamic quantization on a tensor.

    Args:
        x (Tensor): The input tensor to be quantized.
        quant_dtype (torch.dtype): The data type to which the tensor should be quantized.
        quant_granularity (str, optional): The granularity of quantization. Defaults to "PER_TOKEN".
            Options include:
            - "PER_TOKEN": Quantize each element independently.
            - "PER_CHANNEL": Quantize each channel independently.
            - "PER_TENSOR": Quantize the entire tensor as a whole.

    Returns:
        tuple[Tensor, dynamic_quant_scale_type]: A tuple containing:
            - The quantized tensor.
            - The scaling factor used during quantization.
    """
    return npu_ops.dynamic_quant(x, quant_dtype, quant_granularity)


def linear_w8a8(
    a: Tensor,
    b: Tensor,
    rms_scale: linear_w8a8_scale_type,
    linear_scale: linear_w8a8_scale_type,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    bias: Tensor = None,
) -> Tensor:
    """Performs a linear transformation on two quantized input tensors.

    Args:
        a (Tensor): The first quantized input tensor.
        b (Tensor): The second quantized input tensor.
        rms_scale (float): The scaling factor for a.
        linear_scale (float): The scaling factor for b.
        out_dtype (torch.dtype): The target data type for the output tensor.
        quant_dtype (torch.dtype): The data type of the quantized input tensors.
        bias (Tensor): The bias tensor to be added to the output.

    Returns:
        Tensor: The  output tensor after applying the linear transformation.
    """
    return npu_ops.linear_w8a8(a, b, rms_scale, linear_scale, out_dtype, quant_dtype, bias)


def linear_w8a8_static(
    x: Tensor,
    weight: Tensor,
    input_scale: Tensor,
    input_offset: Tensor,
    deq_scale: Tensor,
    out_dtype: torch.dtype,
    quant_dtype: torch.dtype,
    quant_bias: Optional[Tensor] = None,
) -> Tensor:
    """Run ModelSlim static W8A8 quantization followed by quant matmul."""
    return npu_ops.linear_w8a8_static(
        x,
        weight,
        input_scale,
        input_offset,
        deq_scale,
        out_dtype,
        quant_dtype,
        quant_bias,
    )


def rms_norm_w8a8(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Apply RMS normalization to the input tensor and quantizes the result.

    Args:
        hidden_states (Tensor): The input tensor to be normalized and quantized.
        weight (Tensor): The scaling weight applied to the normalized tensor.
        epsilon (float): A value added to the denominator for numerical stability during normalization.
        quant_dtype (torch.dtype): The target data type for the quantized result.

     Returns:
        tuple[Tensor, Tensor]: A tuple containing:
            - The RMS-normalized and quantized tensor.
            - The scaling factor used during quantization.
    """
    return npu_ops.rms_norm_w8a8(hidden_states, weight, epsilon, quant_dtype)


def add_rms_norm_w8a8(
    hidden_states: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply RMS normalization to the input tensor, adds a residual connection,
    and quantizes the result.

    Args:
        hidden_states (Tensor): The input tensor to be normalized and quantized.
        residual (Tensor): The residual tensor to be added to the normalized tensor.
        weight (Tensor): The scaling weight applied to the normalized tensor.
        epsilon (float): A value added to the denominator for numerical stability during normalization.
        quant_dtype (torch.dtype): The target data type for the quantized result.

    Returns:
        tuple[Tensor, Tensor, Tensor]: A tuple containing:
            - The RMS-normalized, residual-added, and quantized tensor.
            - The scaling factor used during quantization.
            - The residual tensor.
    """
    return npu_ops.add_rms_norm_w8a8(hidden_states, residual, weight, epsilon, quant_dtype)
