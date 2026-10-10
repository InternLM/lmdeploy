# Copyright (c) OpenMMLab. All rights reserved.
"""Reusable Torch semantics for the DSA KPool indexer.

KPool has two different kinds of runtime data.  Closed pools are pageable and belong in the named DSA index cache.  The
unfinished per-request tail is sequence state and must be supplied by the caller; this module deliberately does not hide
it in mutable module tensors.

The functions here are a device-agnostic correctness path.  CUDA backends can replace compression, FP8 scoring, and
top-k with fused kernels while retaining these input/output contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lmdeploy.pytorch.consts import DSA_INDEX_SCALE_BYTES, DSA_INDEXER_K_CACHE_NAME, dsa_packed_indexer_k_cache_shape
from lmdeploy.pytorch.engine.cache_engine.schema import BlockCacheBinding, BlockCacheRequest, BlockCacheRequestContext
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.pytorch.nn.linear import build_colwise_linear

KPOOL_PAGE_SIZE = 64
KPOOL_FP8_MAX = 448.0
KPOOL_NORM_EPS = 1e-6
KPOOL_SCALE_FORMAT = 'ue8m0'
KPOOL_INDEXER_PARAMETER_NAMES = (
    'index_kpool_compress_ape',
    'index_kpool_compress_gate',
    'k_norm.weight',
    'k_norm.bias',
    'weights_proj.weight',
    'wk.weight',
    'wq_b.weight',
)


def _validate_pool_geometry(pool_size: int, topk: int | None = None) -> None:
    if pool_size <= 1:
        raise ValueError(f'KPool pool_size must be greater than one, got {pool_size}.')
    if KPOOL_PAGE_SIZE % pool_size:
        raise ValueError(f'KPool pool_size must divide page size {KPOOL_PAGE_SIZE}, got {pool_size}.')
    if topk is not None and (topk <= 0 or topk % pool_size):
        raise ValueError(f'KPool topk must be positive and divisible by pool_size, got topk={topk}, '
                         f'pool_size={pool_size}.')


@dataclass(frozen=True)
class KPoolUpdate:
    """Closed pools and the unfinished tail produced by one request chunk.

    ``closed_group_ids`` are request-local logical pool ids.  ``tail_keys`` and
    ``tail_scores`` start at ``tail_logical_start`` and must be persisted as
    request state before the next chunk or decode token.
    """

    closed_group_ids: Tensor
    closed_keys: Tensor
    closed_scores: Tensor
    tail_keys: Tensor
    tail_scores: Tensor
    tail_logical_start: int


@dataclass(frozen=True)
class KPoolDecodeUpdate:
    """Fixed-shape KPool update used by batched autoregressive decode.

    CUDA Graph decode pads requests to a capture bucket, so this contract
    keeps every tensor batch-shaped and represents inactive rows with
    ``valid_state=False`` instead of materializing dynamic index tensors.
    """

    closed_keys: Tensor
    closed_scores: Tensor
    group_ids: Tensor
    should_close: Tensor
    next_tail_keys: Tensor
    next_tail_scores: Tensor
    safe_state_ids: Tensor
    valid_state: Tensor


class KPoolIndexer(nn.Module):
    """Replicated KPool parameter layer shared by all attention-TP ranks.

    The seven parameter names and dtypes match the GLM-5.3/SGLang checkpoint contract.  Query/key rotary handling and
    cache ownership stay with the model/backend because they depend on model geometry and request metadata.
    """

    def __init__(
        self,
        hidden_size: int,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        q_lora_rank: int,
        index_kpool: int,
        norm_eps: float = KPOOL_NORM_EPS,
        scale_fmt: str | None = KPOOL_SCALE_FORMAT,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        prefix: str = '',
        key_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        _validate_pool_geometry(index_kpool, index_topk)
        if index_head_dim <= 0 or index_head_dim & (index_head_dim - 1):
            raise ValueError(f'KPool index_head_dim must be a positive power of two, got {index_head_dim}.')
        if dtype is None:
            dtype = torch.bfloat16

        self.hidden_size = hidden_size
        self.n_heads = index_n_heads
        self.head_dim = index_head_dim
        self.index_topk = index_topk
        self.q_lora_rank = q_lora_rank
        self.index_kpool = index_kpool
        self.softmax_scale = index_head_dim**-0.5
        self.scale_fmt = scale_fmt
        self._block_cache_binding: BlockCacheBinding | None = None

        def add_prefix(name: str) -> str:
            return f'{prefix}.{name}' if prefix else name

        # SGLang's ReplicatedLinear contract: none of these projections is TP
        # sharded, even when the surrounding MLA attention runs with TP=8.
        self.wq_b = build_colwise_linear(
            q_lora_rank,
            index_n_heads * index_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=None,
            check_dist=False,
            prefix=add_prefix('wq_b'),
        )
        self.wk = build_colwise_linear(
            hidden_size,
            index_head_dim,
            bias=False,
            dtype=dtype,
            device=device,
            is_tp=False,
            quant_config=None,
            check_dist=False,
            prefix=add_prefix('wk'),
        )
        # The checkpoint stores BF16, but SGLang promotes these parameters to
        # FP32 before inference.  LMDeploy's loader performs the same cast.
        self.weights_proj = build_colwise_linear(
            hidden_size,
            index_n_heads,
            bias=False,
            dtype=torch.float32,
            device=device,
            is_tp=False,
            quant_config=None,
            check_dist=False,
            prefix=add_prefix('weights_proj'),
        )
        # The device-agnostic KPool owner keeps a Torch reference default.
        # Models that require a platform-exact provider can inject the same
        # parameter-shaped component without changing checkpoint names.
        self.k_norm = (key_norm if key_norm is not None else nn.LayerNorm(
            index_head_dim, eps=norm_eps, dtype=torch.float32, device=device).requires_grad_(False))
        self.index_kpool_compress_ape = nn.Parameter(
            torch.zeros(index_kpool, index_head_dim, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.index_kpool_compress_gate = nn.Parameter(
            torch.empty(index_head_dim, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )

    def get_block_cache_requests(self, context: BlockCacheRequestContext):
        """Declare the pooled index cache through the shared cache planner."""
        geometry = context.geometry
        if (geometry.logical_block_size != KPOOL_PAGE_SIZE
                or geometry.kernel_block_size != KPOOL_PAGE_SIZE):
            raise ValueError('GLM-5.3 KPool requires logical block_size=64 and kernel_block_size=64.')
        return (BlockCacheRequest(
            name=DSA_INDEXER_K_CACHE_NAME,
            shape=dsa_packed_indexer_k_cache_shape(KPOOL_PAGE_SIZE // self.index_kpool, self.head_dim),
            dtype=torch.uint8,
            per_row_contiguous=True,
        ), )

    def bind_block_cache(self, binding: BlockCacheBinding):
        """Retain the compact consumer row assigned by the cache planner."""
        if binding.cache_name != DSA_INDEXER_K_CACHE_NAME:
            raise ValueError(f'Unexpected KPool cache name: {binding.cache_name}.')
        self._block_cache_binding = binding

    def get_block_cache(self) -> Tensor:
        """Resolve this indexer's row from the live request context."""
        binding = self._block_cache_binding
        if binding is None:
            raise RuntimeError('The KPool index cache has not been bound.')
        caches = get_step_ctx_manager().current_context().block_caches
        if hasattr(caches, 'row'):
            return caches.row(binding.cache_name, binding.consumer_row)
        return caches[binding.cache_name][binding.consumer_row]

    def project_query(self, q_lora: Tensor) -> Tensor:
        """Project the latent query to ``[..., index_n_heads, head_dim]``."""
        query = self.wq_b(q_lora)
        return query.unflatten(-1, (self.n_heads, self.head_dim))

    def project_key(self, hidden_states: Tensor) -> Tensor:
        """Project and normalize one shared index key per token."""
        key = self.wk(hidden_states)
        # Match vLLM's FP32 LayerNorm followed by a cast to the activation dtype.
        return self.k_norm(key.float()).to(key.dtype)

    def project_compress_score(self, hidden_states: Tensor) -> Tensor:
        """Return per-slot, per-dimension compression gates."""
        return F.linear(hidden_states, self.index_kpool_compress_gate)

    def project_head_gate(self, hidden_states: Tensor) -> Tensor:
        """Return FP32 head gates including SGLang's ``num_heads**-0.5``."""
        return self.weights_proj(hidden_states.float()) * self.n_heads**-0.5

    def quantize_fp8(self, values: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize index vectors with the checkpoint's UE8M0 scale rule."""
        return kpool_quantize_fp8(
            values,
            block_size=self.head_dim,
            round_scale=self.scale_fmt is not None,
        )


def kpool_normalized_hadamard(values: Tensor) -> Tensor:
    """Apply the normalized Walsh-Hadamard transform on the last dimension."""
    width = values.size(-1)
    if width <= 0 or width & (width - 1):
        raise ValueError(f'Hadamard width must be a positive power of two, got {width}.')

    output = values.float()
    stride = 1
    while stride < width:
        shape = output.shape[:-1] + (-1, 2, stride)
        paired = output.reshape(shape)
        left, right = paired.unbind(dim=-2)
        output = torch.cat((left + right, left - right), dim=-1).reshape_as(output)
        stride *= 2
    return output * width**-0.5


def kpool_rotate_query(query: Tensor) -> Tensor:
    """Match the BF16-preserving query rotation used before FP8
    quantization."""
    return kpool_normalized_hadamard(query).to(query.dtype)


def kpool_partition_update(
    chunk_keys: Tensor,
    chunk_scores: Tensor,
    history_length: int,
    pool_size: int,
    tail_keys: Tensor | None = None,
    tail_scores: Tensor | None = None,
) -> KPoolUpdate:
    """Assemble arbitrary-length input with a prior tail into closed pools.

    This is the state-free equivalent of SGLang's extend/decode tail ring.  The caller owns persistence of the returned
    tail and supplies it on the next invocation.
    """
    _validate_pool_geometry(pool_size)
    if history_length < 0:
        raise ValueError(f'history_length must be non-negative, got {history_length}.')
    if chunk_keys.ndim != 2 or chunk_scores.shape != chunk_keys.shape:
        raise ValueError('chunk_keys and chunk_scores must have the same [tokens, head_dim] shape.')

    previous_tail_len = history_length % pool_size
    if tail_keys is None:
        tail_keys = chunk_keys[:0]
    if tail_scores is None:
        tail_scores = chunk_scores[:0]
    expected_tail_shape = (previous_tail_len, chunk_keys.size(1))
    if tail_keys.shape != expected_tail_shape or tail_scores.shape != expected_tail_shape:
        raise ValueError('The supplied tail must contain exactly history_length % pool_size rows; '
                         f'expected {expected_tail_shape}, got keys={tuple(tail_keys.shape)}, '
                         f'scores={tuple(tail_scores.shape)}.')
    if tail_keys.device != chunk_keys.device or tail_scores.device != chunk_scores.device:
        raise ValueError('Chunk and tail tensors must be on the same device.')

    all_keys = torch.cat((tail_keys, chunk_keys), dim=0)
    all_scores = torch.cat((tail_scores, chunk_scores), dim=0)
    num_closed = all_keys.size(0) // pool_size
    num_closed_tokens = num_closed * pool_size
    first_group_id = (history_length - previous_tail_len) // pool_size
    closed_group_ids = torch.arange(
        first_group_id,
        first_group_id + num_closed,
        dtype=torch.int64,
        device=chunk_keys.device,
    )
    closed_shape = (num_closed, pool_size, chunk_keys.size(1))
    closed_keys = all_keys[:num_closed_tokens].reshape(closed_shape)
    closed_scores = all_scores[:num_closed_tokens].reshape(closed_shape)
    new_tail_keys = all_keys[num_closed_tokens:]
    new_tail_scores = all_scores[num_closed_tokens:]
    tail_logical_start = history_length + chunk_keys.size(0) - new_tail_keys.size(0)
    return KPoolUpdate(
        closed_group_ids=closed_group_ids,
        closed_keys=closed_keys,
        closed_scores=closed_scores,
        tail_keys=new_tail_keys,
        tail_scores=new_tail_scores,
        tail_logical_start=tail_logical_start,
    )


def kpool_decode_update(
    keys: Tensor,
    scores: Tensor,
    tail_key_state: Tensor,
    tail_score_state: Tensor,
    state_ids: Tensor,
    history_lengths: Tensor,
    pool_size: int,
) -> KPoolDecodeUpdate:
    """Build a fixed-shape, graph-safe update for one-token decode.

    State slot zero is LMDeploy's reserved dummy slot. Invalid CUDA Graph padding rows therefore read and write slot
    zero without affecting a live request. Every returned tensor has a shape determined only by the graph capture
    bucket.
    """
    _validate_pool_geometry(pool_size)
    if keys.ndim != 2 or scores.shape != keys.shape:
        raise ValueError(
            'keys and scores must have the same [batch, head_dim] shape.')
    if tail_key_state.shape != tail_score_state.shape:
        raise ValueError('KPool key and score tail states must match.')
    if tail_key_state.ndim != 3:
        raise ValueError(
            'KPool tail state must have shape [state, pool_size, head_dim].')
    if tail_key_state.shape[1:] != (pool_size, keys.size(1)):
        raise ValueError(
            'KPool tail state geometry does not match decode inputs.')
    batch_size = keys.size(0)
    if state_ids.shape != (batch_size, ) or history_lengths.shape != (
            batch_size, ):
        raise ValueError(
            'state_ids and history_lengths must contain one value per row.')

    safe_state_ids = state_ids.to(torch.int64).clamp_min(0)
    valid_state = state_ids >= 0
    history_lengths = history_lengths.to(torch.int64).clamp_min(0)
    previous_tail_lengths = torch.remainder(history_lengths, pool_size)
    previous_keys = tail_key_state.index_select(0, safe_state_ids)
    previous_scores = tail_score_state.index_select(0, safe_state_ids)

    slots = torch.arange(
        pool_size, dtype=torch.int64, device=keys.device)[None, :, None]
    previous_valid = slots < previous_tail_lengths[:, None, None]
    closed_keys = torch.where(previous_valid, previous_keys,
                              torch.zeros_like(previous_keys))
    closed_scores = torch.where(previous_valid, previous_scores,
                                torch.zeros_like(previous_scores))
    insert_at = previous_tail_lengths[:, None, None].expand(
        -1, 1, keys.size(1))
    closed_keys.scatter_(1, insert_at, keys[:, None, :])
    closed_scores.scatter_(1, insert_at, scores[:, None, :])

    should_close = valid_state & (previous_tail_lengths == pool_size - 1)
    next_tail_keys = torch.where(should_close[:, None, None],
                                 torch.zeros_like(closed_keys), closed_keys)
    next_tail_scores = torch.where(should_close[:, None, None],
                                   torch.zeros_like(closed_scores),
                                   closed_scores)
    # Invalid graph-padding rows preserve the reserved dummy state.
    next_tail_keys = torch.where(valid_state[:, None, None], next_tail_keys,
                                 previous_keys)
    next_tail_scores = torch.where(valid_state[:, None, None],
                                   next_tail_scores, previous_scores)
    group_ids = torch.div(
        history_lengths, pool_size, rounding_mode='floor')
    return KPoolDecodeUpdate(
        closed_keys=closed_keys,
        closed_scores=closed_scores,
        group_ids=group_ids,
        should_close=should_close,
        next_tail_keys=next_tail_keys,
        next_tail_scores=next_tail_scores,
        safe_state_ids=safe_state_ids,
        valid_state=valid_state,
    )


KPoolCompressMode = Literal['extend', 'decode']


def _validate_compress_inputs(closed_keys: Tensor, closed_scores: Tensor,
                              ape: Tensor) -> None:
    if closed_keys.ndim != 3 or closed_scores.shape != closed_keys.shape:
        raise ValueError(
            'closed_keys and closed_scores must have the same '
            '[groups, pool_size, head_dim] shape.')
    if ape.shape != closed_keys.shape[1:]:
        raise ValueError(
            f'ape must have shape {tuple(closed_keys.shape[1:])}, '
            f'got {tuple(ape.shape)}.')


def _finish_compressed_pool(pooled: Tensor) -> Tensor:
    # SGLang rounds both the pooled vector and the rotated vector through BF16.
    pooled = pooled.to(torch.bfloat16).float()
    return kpool_normalized_hadamard(pooled).to(torch.bfloat16)


def kpool_compress_online(closed_keys: Tensor, closed_scores: Tensor,
                          ape: Tensor) -> Tensor:
    """Use the online recurrence from SGLang's extend assembly kernel.

    The explicit slot loop matches SGLang's compression kernel reduction order.  A vectorized max/exp/sum is
    mathematically equivalent but can move values across the FP8 quantization boundary after different FP32 rounds.
    """
    _validate_compress_inputs(closed_keys, closed_scores, ape)

    groups, pool_size, head_dim = closed_keys.shape
    max_score = torch.full(
        (groups, head_dim),
        -float('inf'),
        dtype=torch.float32,
        device=closed_keys.device,
    )
    denominator = torch.zeros_like(max_score)
    accumulator = torch.zeros_like(max_score)
    for slot in range(pool_size):
        score = closed_scores[:, slot].float() + ape[slot].float()
        new_max = torch.maximum(max_score, score)
        rescale = torch.exp(max_score - new_max)
        probability = torch.exp(score - new_max)
        key = closed_keys[:, slot].float()
        denominator = denominator * rescale + probability
        accumulator = accumulator * rescale + key * probability
        max_score = new_max
    return _finish_compressed_pool(accumulator / denominator)


def kpool_compress_two_pass(closed_keys: Tensor, closed_scores: Tensor,
                            ape: Tensor) -> Tensor:
    """Use the max-then-sum order from SGLang's decode close-pool kernel."""
    _validate_compress_inputs(closed_keys, closed_scores, ape)
    groups, pool_size, head_dim = closed_keys.shape
    max_score = torch.full(
        (groups, head_dim),
        -float('inf'),
        dtype=torch.float32,
        device=closed_keys.device,
    )
    for slot in range(pool_size):
        score = closed_scores[:, slot].float() + ape[slot].float()
        max_score = torch.maximum(max_score, score)

    denominator = torch.zeros_like(max_score)
    accumulator = torch.zeros_like(max_score)
    for slot in range(pool_size):
        score = closed_scores[:, slot].float() + ape[slot].float()
        probability = torch.exp(score - max_score)
        key = closed_keys[:, slot].float()
        denominator = denominator + probability
        accumulator = accumulator + key * probability
    return _finish_compressed_pool(accumulator / denominator)


def kpool_compress(
    closed_keys: Tensor,
    closed_scores: Tensor,
    ape: Tensor,
    *,
    mode: KPoolCompressMode,
) -> Tensor:
    """Dispatch the explicit SGLang compression order for a forward mode."""
    if mode == 'extend':
        return kpool_compress_online(closed_keys, closed_scores, ape)
    if mode == 'decode':
        return kpool_compress_two_pass(closed_keys, closed_scores, ape)
    raise ValueError(
        f'KPool compression mode must be extend or decode, got {mode!r}.')


def kpool_quantize_fp8(
    values: Tensor,
    block_size: int = 128,
    round_scale: bool = True,
) -> tuple[Tensor, Tensor]:
    """Block-quantize to E4M3FN using SGLang's scale and clamp semantics."""
    if block_size <= 0 or values.size(-1) % block_size:
        raise ValueError(f'Last dimension must be divisible by block_size, got shape={tuple(values.shape)}, '
                         f'block_size={block_size}.')
    grouped = values.float().unflatten(-1, (-1, block_size))
    absmax = grouped.abs().amax(dim=-1).clamp_min(1e-4)
    scale = absmax / KPOOL_FP8_MAX
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    quantized = (grouped / scale.unsqueeze(-1)).clamp(-KPOOL_FP8_MAX, KPOOL_FP8_MAX)
    quantized = quantized.flatten(-2).to(torch.float8_e4m3fn)
    return quantized, scale.to(torch.float32)


def kpool_score(
    query_fp8: Tensor,
    query_scale: Tensor,
    pooled_key_fp8: Tensor,
    pooled_key_scale: Tensor,
    head_gate: Tensor,
    softmax_scale: float | None = None,
) -> Tensor:
    """Reference FP8 MQA logits for pooled history.

    ``head_gate`` is the output of :meth:`KPoolIndexer.project_head_gate`, so it
    already includes the ``num_heads**-0.5`` factor.  Query and pooled-key
    scales are applied exactly once here.
    """
    if query_fp8.ndim != 3:
        raise ValueError('query_fp8 must have shape [rows, heads, head_dim].')
    if pooled_key_fp8.ndim != 2 or pooled_key_fp8.size(1) != query_fp8.size(2):
        raise ValueError('pooled_key_fp8 must have shape [groups, query_head_dim].')
    rows, heads, head_dim = query_fp8.shape
    if head_gate.shape != (rows, heads):
        raise ValueError(f'head_gate must have shape {(rows, heads)}, got {tuple(head_gate.shape)}.')
    if query_scale.shape == (rows, heads, 1):
        query_scale = query_scale.squeeze(-1)
    if query_scale.shape != (rows, heads):
        raise ValueError(f'query_scale must have shape {(rows, heads)} or {(rows, heads, 1)}, '
                         f'got {tuple(query_scale.shape)}.')
    if pooled_key_scale.shape == (pooled_key_fp8.size(0), 1):
        pooled_key_scale = pooled_key_scale.squeeze(-1)
    if pooled_key_scale.shape != (pooled_key_fp8.size(0), ):
        raise ValueError('pooled_key_scale must contain one scale per pooled key.')
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    query_weight = head_gate.float() * query_scale.float() * softmax_scale
    per_head_logits = torch.einsum(
        'rhd,kd->rhk', query_fp8.float(), pooled_key_fp8.float())
    per_head_logits = per_head_logits.clamp_min_(0)
    logits = torch.einsum('rhk,rh->rk', per_head_logits, query_weight)
    return logits * pooled_key_scale.float().unsqueeze(0)


def kpool_pooled_block_offsets(token_block_offsets: Tensor, pool_size: int,
                               page_size: int = KPOOL_PAGE_SIZE) -> Tensor:
    """Map compact16 owners directly; subsample columns for legacy page64."""
    _validate_pool_geometry(pool_size)
    if token_block_offsets.ndim < 1:
        raise ValueError('token_block_offsets must have at least one dimension.')
    stride = page_size * pool_size // KPOOL_PAGE_SIZE
    if page_size * pool_size % KPOOL_PAGE_SIZE or stride < 1:
        raise ValueError("KPool storage pages must cover whole token pages.")
    if stride == 1:
        return token_block_offsets
    columns = torch.arange(0, token_block_offsets.size(-1), stride, device=token_block_offsets.device)
    return token_block_offsets.index_select(-1, columns)


def kpool_pooled_write_locations(
    token_block_offsets: Tensor,
    group_ids: Tensor,
    pool_size: int,
    page_size: int = KPOOL_PAGE_SIZE,
) -> Tensor:
    """Map request-local logical pool ids to packed physical cache slots."""
    _validate_pool_geometry(pool_size)
    if page_size not in (KPOOL_PAGE_SIZE, KPOOL_PAGE_SIZE // pool_size):
        raise ValueError(f"Unsupported KPool storage page size: {page_size}.")
    if token_block_offsets.ndim != 1 or group_ids.ndim != 1:
        raise ValueError('token_block_offsets and group_ids must both be one-dimensional.')
    group_ids = group_ids.to(torch.int64)
    page_group = torch.div(group_ids, page_size, rounding_mode='floor')
    token_page_column = page_group * (page_size * pool_size // KPOOL_PAGE_SIZE)
    if token_page_column.numel() and int(token_page_column.max()) >= token_block_offsets.numel():
        raise ValueError('token_block_offsets is too short for the requested logical pool ids.')
    physical_page = token_block_offsets.index_select(0, token_page_column)
    return physical_page.to(torch.int64) * page_size + torch.remainder(group_ids, page_size)


def kpool_packed_cache_views(
    packed_cache: Tensor,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Expose FP8 values and FP32 scales from one packed DSA cache row.

    The byte layout intentionally matches the existing DeepGEMM DSA cache:
    each page stores its value rows first, followed by its scales. Compact
    pages hold 16 rows; the legacy layout holds 64.
    """
    if packed_cache.dtype != torch.uint8:
        raise TypeError(
            'Packed KPool cache must be uint8, '
            f'got {packed_cache.dtype}.')
    if packed_cache.dim() != 4 or packed_cache.size(2) != 1:
        raise ValueError(
            'Packed KPool cache must have shape '
            '[num_blocks, entries, 1, head_dim + 4].')
    packed_width = head_dim + DSA_INDEX_SCALE_BYTES
    if packed_cache.size(-1) != packed_width:
        raise ValueError(
            f'Packed KPool cache last dim must be {packed_width}, '
            f'got {packed_cache.size(-1)}.')

    num_blocks, entries_per_block = packed_cache.shape[:2]
    flat = packed_cache.view(num_blocks, -1)
    value_bytes = entries_per_block * head_dim
    scale_bytes = entries_per_block * DSA_INDEX_SCALE_BYTES
    values = flat[:, :value_bytes].view(torch.float8_e4m3fn).view(
        num_blocks, entries_per_block, head_dim)
    scales = flat[:, value_bytes:value_bytes + scale_bytes].view(
        torch.float32).view(num_blocks, entries_per_block, 1)
    return values, scales


def kpool_write_packed_cache(
    packed_cache: Tensor,
    token_block_offsets: Tensor,
    group_ids: Tensor,
    pooled_key_fp8: Tensor,
    pooled_key_scale: Tensor,
    pool_size: int,
) -> None:
    """Write closed request-local pools into their pageable packed cache."""
    if pooled_key_fp8.ndim != 2:
        raise ValueError('pooled_key_fp8 must have shape [groups, head_dim].')
    if pooled_key_scale.shape == (pooled_key_fp8.size(0), ):
        pooled_key_scale = pooled_key_scale.unsqueeze(-1)
    if pooled_key_scale.shape != (pooled_key_fp8.size(0), 1):
        raise ValueError('pooled_key_scale must contain one scale per group.')
    if group_ids.shape != (pooled_key_fp8.size(0), ):
        raise ValueError('group_ids must contain one id per pooled key.')
    values, scales = kpool_packed_cache_views(
        packed_cache, pooled_key_fp8.size(-1))
    locations = kpool_pooled_write_locations(
        token_block_offsets,
        group_ids,
        pool_size,
        page_size=values.size(1),
    )
    pages = torch.div(locations, values.size(1), rounding_mode='floor')
    slots = torch.remainder(locations, values.size(1))
    values[pages, slots] = pooled_key_fp8.to(values.dtype)
    scales[pages, slots] = pooled_key_scale.to(scales.dtype)


def kpool_write_packed_cache_batched(
    packed_cache: Tensor,
    token_block_offsets: Tensor,
    group_ids: Tensor,
    pooled_key_fp8: Tensor,
    pooled_key_scale: Tensor,
    pool_size: int,
    valid: Tensor,
) -> None:
    """Write at most one closed pool per fixed-shape decode row.

    Invalid rows target reserved cache block zero and keep its previous value. This avoids dynamic boolean indexing and
    keeps CUDA Graph addresses and launch geometry stable.
    """
    if pooled_key_fp8.ndim != 2:
        raise ValueError(
            'pooled_key_fp8 must have shape [batch, head_dim].')
    batch_size = pooled_key_fp8.size(0)
    if pooled_key_scale.shape == (batch_size, ):
        pooled_key_scale = pooled_key_scale.unsqueeze(-1)
    if pooled_key_scale.shape != (batch_size, 1):
        raise ValueError('pooled_key_scale must contain one scale per row.')
    if token_block_offsets.ndim != 2 or token_block_offsets.size(0) != batch_size:
        raise ValueError(
            'token_block_offsets must have one page-table row per batch row.')
    if group_ids.shape != (batch_size, ) or valid.shape != (batch_size, ):
        raise ValueError('group_ids and valid must contain one value per row.')

    values, scales = kpool_packed_cache_views(
        packed_cache, pooled_key_fp8.size(-1))
    group_ids = group_ids.to(torch.int64)
    page_group = torch.div(
        group_ids, values.size(1), rounding_mode='floor')
    token_page_column = page_group * (values.size(1) * pool_size // KPOOL_PAGE_SIZE)
    token_page_column = token_page_column.clamp(
        min=0, max=token_block_offsets.size(1) - 1)
    physical_page = token_block_offsets.gather(
        1, token_page_column[:, None]).squeeze(1).to(torch.int64)
    locations = physical_page * values.size(1) + torch.remainder(
        group_ids, values.size(1))
    # Cache block zero is reserved by LMDeploy and is the shared dummy target.
    locations = torch.where(valid, locations, torch.zeros_like(locations))
    pages = torch.div(locations, values.size(1), rounding_mode='floor')
    slots = torch.remainder(locations, values.size(1))
    current_values = values[pages, slots]
    current_scales = scales[pages, slots]
    values[pages, slots] = torch.where(
        valid[:, None], pooled_key_fp8.to(values.dtype), current_values)
    scales[pages, slots] = torch.where(
        valid[:, None], pooled_key_scale.to(scales.dtype), current_scales)


def kpool_read_packed_cache(
    packed_cache: Tensor,
    token_block_offsets: Tensor,
    num_groups: int,
    pool_size: int,
) -> tuple[Tensor, Tensor]:
    """Gather a request's closed pools in logical order from paged storage."""
    if num_groups < 0:
        raise ValueError(f'num_groups must be non-negative, got {num_groups}.')
    head_dim = packed_cache.size(-1) - DSA_INDEX_SCALE_BYTES
    values, scales = kpool_packed_cache_views(packed_cache, head_dim)
    group_ids = torch.arange(
        num_groups, dtype=torch.int64, device=packed_cache.device)
    locations = kpool_pooled_write_locations(
        token_block_offsets,
        group_ids,
        pool_size,
        page_size=values.size(1),
    )
    pages = torch.div(locations, values.size(1), rounding_mode='floor')
    slots = torch.remainder(locations, values.size(1))
    return values[pages, slots], scales[pages, slots]


def kpool_selected_token_counts(seq_lens: Tensor, topk: int, pool_size: int) -> Tensor:
    """Return selected history plus always-selected ragged tail token
    counts."""
    _validate_pool_geometry(pool_size, topk)
    full_pool_tokens = torch.div(seq_lens, pool_size, rounding_mode='floor') * pool_size
    return full_pool_tokens.clamp(max=topk) + seq_lens - full_pool_tokens


def _map_logical_indices(
    logical: Tensor,
    valid: Tensor,
    page_table: Tensor | None,
    topk_offsets: Tensor | None,
) -> Tensor:
    if page_table is not None and topk_offsets is not None:
        raise ValueError('page_table and topk_offsets are mutually exclusive.')
    if page_table is not None:
        if page_table.ndim != 2 or page_table.size(0) != logical.size(0):
            raise ValueError('page_table must have one [logical_token] row per score row.')
        if page_table.size(1) == 0:
            if not valid.is_cuda and bool(valid.any()):
                raise ValueError('A non-empty logical selection cannot use an empty page_table.')
            return torch.full_like(logical, -1, dtype=torch.int32)
        safe = logical.clamp(min=0, max=page_table.size(1) - 1)
        output = page_table.gather(1, safe).to(torch.int32)
    elif topk_offsets is not None:
        if topk_offsets.ndim == 2 and topk_offsets.size(1) == 1:
            topk_offsets = topk_offsets.squeeze(1)
        if topk_offsets.shape != (logical.size(0), ):
            raise ValueError('topk_offsets must contain one value per score row.')
        output = (logical + topk_offsets.to(torch.int64).unsqueeze(1)).to(torch.int32)
    else:
        output = logical.to(torch.int32)
    return torch.where(valid, output, torch.full_like(output, -1))


def kpool_expand_selected_groups(
    selected_groups: Tensor,
    group_lengths: Tensor,
    pool_size: int,
    topk: int,
    *,
    seq_lens: Tensor | None = None,
    page_table: Tensor | None = None,
    topk_offsets: Tensor | None = None,
    page_table_row_index: Tensor | None = None,
    out_rows: int | None = None,
) -> Tensor:
    """Expand preselected pool ids to tokens and append ragged tails.

    The returned width is ``topk`` without ``seq_lens`` and
    ``topk + pool_size - 1`` when the always-selected tail is requested.
    Entries are request-local logical token ids unless ``page_table`` or
    ``topk_offsets`` requests the same transform used by SGLang.  The order of
    ``selected_groups`` is deliberately preserved because sparse-attention
    reduction order is numerically observable.
    """
    _validate_pool_geometry(pool_size, topk)
    group_budget = topk // pool_size
    if selected_groups.ndim != 2 or selected_groups.size(1) != group_budget:
        raise ValueError(
            'selected_groups must have shape '
            f'[rows, {group_budget}], got {tuple(selected_groups.shape)}.')
    rows = selected_groups.size(0)
    if group_lengths.shape != (rows, ):
        raise ValueError(
            'group_lengths must contain one value per selected-groups row.')
    if out_rows is not None and out_rows < rows:
        raise ValueError(f'out_rows must be at least {rows}, got {out_rows}.')
    device = selected_groups.device
    group_lengths = group_lengths.to(device=device, dtype=torch.int64)
    if not group_lengths.is_cuda and bool((group_lengths < 0).any()):
        raise ValueError('group_lengths must be non-negative.')
    selected_groups = selected_groups.to(device=device, dtype=torch.int64)

    page_table_for_rows = page_table
    if page_table_row_index is not None:
        if page_table is None:
            raise ValueError('page_table_row_index requires page_table.')
        if page_table_row_index.shape != (rows, ):
            raise ValueError('page_table_row_index must contain one index per score row.')
        page_table_for_rows = page_table.index_select(0, page_table_row_index.to(page_table.device, torch.int64))

    ranks = torch.arange(group_budget, dtype=torch.int64, device=device)
    selected_valid = ranks.unsqueeze(0) < group_lengths.clamp(
        max=group_budget).unsqueeze(1)
    selected_valid &= selected_groups >= 0
    selected_valid &= selected_groups < group_lengths.unsqueeze(1)

    offsets = torch.arange(pool_size, dtype=torch.int64, device=device)
    logical = (selected_groups.unsqueeze(-1) * pool_size + offsets).reshape(rows, topk)
    expanded_valid = selected_valid.unsqueeze(-1).expand(-1, -1, pool_size).reshape(rows, topk)
    expanded = _map_logical_indices(logical, expanded_valid, page_table_for_rows, topk_offsets)

    if seq_lens is None:
        result = expanded
    else:
        if seq_lens.shape != (rows, ):
            raise ValueError('seq_lens must contain one value per score row.')
        seq_lens = seq_lens.to(device=device, dtype=torch.int64)
        if (not seq_lens.is_cuda and bool((torch.div(
                seq_lens, pool_size, rounding_mode='floor') !=
                                          group_lengths).any())):
            raise ValueError('group_lengths must equal floor(seq_lens / pool_size) when appending KPool tails.')

        output_width = topk + pool_size - 1
        output_columns = torch.arange(
            output_width, dtype=torch.int64, device=device).unsqueeze(0)
        history_width = (group_lengths * pool_size).clamp(max=topk).unsqueeze(1)
        history_valid = output_columns < history_width
        safe_history_columns = output_columns.clamp(max=topk - 1).expand(rows, -1)
        history_values = expanded.gather(1, safe_history_columns)

        tail_offset = output_columns - history_width
        tail_count = torch.remainder(seq_lens, pool_size).unsqueeze(1)
        tail_valid = (tail_offset >= 0) & (tail_offset < tail_count)
        tail_logical = group_lengths.unsqueeze(1) * pool_size + tail_offset
        tail_values = _map_logical_indices(tail_logical, tail_valid, page_table_for_rows, topk_offsets)

        result = torch.full(
            (rows, output_width), -1, dtype=torch.int32, device=device)
        result = torch.where(history_valid, history_values, result)
        result = torch.where(tail_valid, tail_values, result)

    if out_rows is not None and out_rows != rows:
        padded = torch.full((out_rows, result.size(1)), -1, dtype=result.dtype, device=result.device)
        padded[:rows] = result
        return padded
    return result


def kpool_topk(
    logits: Tensor,
    group_lengths: Tensor,
    pool_size: int,
    topk: int,
    *,
    seq_lens: Tensor | None = None,
    row_starts: Tensor | None = None,
    page_table: Tensor | None = None,
    topk_offsets: Tensor | None = None,
    page_table_row_index: Tensor | None = None,
    out_rows: int | None = None,
) -> Tensor:
    """Torch reference for SGLang's pooled-group selection semantics.

    Rows no longer than the group budget bypass score selection and retain
    chronological group order.  Longer rows use an unsorted Torch top-k as a
    set-level CPU reference.  The CUDA production path supplies groups from
    the shared byte-radix selector to :func:`kpool_expand_selected_groups`.
    """
    _validate_pool_geometry(pool_size, topk)
    if logits.ndim != 2 or group_lengths.shape != (logits.size(0), ):
        raise ValueError(
            'logits must be [rows, groups] with one group_lengths value per row.')
    rows, columns = logits.shape
    if row_starts is None:
        row_starts = torch.zeros(
            rows, dtype=torch.int64, device=logits.device)
    else:
        if row_starts.shape != (rows, ):
            raise ValueError('row_starts must contain one value per score row.')
        row_starts = row_starts.to(
            device=logits.device, dtype=torch.int64)
    group_lengths = group_lengths.to(device=logits.device, dtype=torch.int64)
    invalid_windows = ((group_lengths < 0) | (row_starts < 0)
                       | (row_starts + group_lengths > columns))
    if bool(invalid_windows.any()):
        raise ValueError(
            'Each [row_start, row_start + group_length) window must be inside logits.')

    group_budget = topk // pool_size
    chronological = torch.arange(
        group_budget, dtype=torch.int64, device=logits.device).expand(rows, -1)
    selected_groups = torch.where(
        chronological < group_lengths.unsqueeze(1),
        chronological,
        torch.full_like(chronological, -1),
    )

    long_rows = torch.nonzero(
        group_lengths > group_budget, as_tuple=False).flatten()
    if long_rows.numel():
        column_ids = torch.arange(
            columns, dtype=torch.int64, device=logits.device).unsqueeze(0)
        valid_window = (column_ids >= row_starts.unsqueeze(1)) & (
            column_ids < (row_starts + group_lengths).unsqueeze(1))
        masked_logits = logits.float().masked_fill(
            ~valid_window, float('-inf'))
        long_logits = masked_logits.index_select(0, long_rows)
        _, indices = torch.topk(
            long_logits, group_budget, dim=-1, sorted=False)
        selected_groups[long_rows] = indices - row_starts.index_select(
            0, long_rows).unsqueeze(1)

    return kpool_expand_selected_groups(
        selected_groups,
        group_lengths,
        pool_size,
        topk,
        seq_lens=seq_lens,
        page_table=page_table,
        topk_offsets=topk_offsets,
        page_table_row_index=page_table_row_index,
        out_rows=out_rows,
    )
