from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tests.turbomind.linear_attn.benchmark import BenchmarkRequest, BenchmarkTask, diff_metrics
from tests.turbomind.linear_attn.cases import InputTensors, RunCase

HEAD_DIM = 128
CHUNK_SIZE = 64
SUPPORTED_CHUNK_SIZES = (1, 16, 32, 64)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_contiguous(name: str, x: torch.Tensor | None) -> None:
    if x is not None:
        _require(x.is_contiguous(), f'{name} must be contiguous, got stride {tuple(x.stride())}')


def _check_chunk_size(chunk_size: int) -> None:
    _require(chunk_size in SUPPORTED_CHUNK_SIZES,
             f'chunk_size must be one of {SUPPORTED_CHUNK_SIZES}, got {chunk_size}')


def _check_3d(name: str, x: torch.Tensor) -> None:
    _require(x.ndim == 3, f'{name} must be rank 3, got shape {tuple(x.shape)}')


def _check_nonempty_batch(name: str, x: torch.Tensor) -> None:
    _require(x.shape[0] > 0, f'{name}: fixed-mode tensors must have B > 0')


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int) -> None:
    _require(cu_seqlens.dtype in (torch.int32, torch.int64),
             f'cu_seqlens dtype must be int32 or int64, got {cu_seqlens.dtype}')
    _require(cu_seqlens.ndim == 1, f'cu_seqlens must be rank 1, got shape {tuple(cu_seqlens.shape)}')
    _require(cu_seqlens.numel() >= 2, f'cu_seqlens must contain at least 2 entries, got {cu_seqlens.numel()}')
    cu_cpu = cu_seqlens.detach().cpu()
    _require(int(cu_cpu[0]) == 0, f'cu_seqlens must start at 0, got {int(cu_cpu[0])}')
    _require(int(cu_cpu[-1]) == total_tokens,
             f'cu_seqlens must end at total_tokens={total_tokens}, got {int(cu_cpu[-1])}')
    deltas = cu_cpu[1:] - cu_cpu[:-1]
    _require(bool((deltas > 0).all()), f'cu_seqlens must be strictly increasing, got {cu_cpu.tolist()}')


def unpack_varlen(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    _require(x.ndim >= 2, f'x must have rank >= 2, got shape {tuple(x.shape)}')
    _require(x.shape[0] == 1, f'varlen tensors must have leading batch 1, got {x.shape[0]}')
    _validate_cu_seqlens(cu_seqlens, x.shape[1])
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    max_len = max(int(length) for length in lengths)
    out = torch.zeros((len(lengths), max_len, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for batch_idx, length in enumerate(lengths):
        start = int(cu_seqlens[batch_idx])
        end = int(cu_seqlens[batch_idx + 1])
        out[batch_idx, :int(length)] = x[0, start:end]
    return out.contiguous()


def pack_varlen(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    _require(x.ndim >= 2, f'x must have rank >= 2, got shape {tuple(x.shape)}')
    _validate_cu_seqlens(cu_seqlens, int(cu_seqlens[-1]))
    real_batch = cu_seqlens.numel() - 1
    _require(x.shape[0] == real_batch, f'x leading batch must be {real_batch}, got {x.shape[0]}')
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    max_len = max(int(length) for length in lengths)
    _require(x.shape[1] >= max_len, f'dense token length must be at least {max_len}, got {x.shape[1]}')
    total = int(cu_seqlens[-1])
    out = torch.empty((1, total, *x.shape[2:]), dtype=x.dtype, device=x.device)
    for batch_idx in range(real_batch):
        start = int(cu_seqlens[batch_idx])
        end = int(cu_seqlens[batch_idx + 1])
        out[0, start:end] = x[batch_idx, :end - start]
    return out.contiguous()


def _pad_and_reshape(x: torch.Tensor, dim: int, chunk_size: int) -> torch.Tensor:
    length = x.shape[dim]
    pad_size = (chunk_size - length % chunk_size) % chunk_size
    zeros = [0] * (2 * (x.ndim - 1 - dim))
    padded = F.pad(x, (*zeros, 0, pad_size))
    return padded.reshape((*x.shape[:dim], -1, chunk_size, *x.shape[dim + 1:]))


def _sequence_lengths(x: torch.Tensor, cu_seqlens: torch.Tensor | None) -> list[int]:
    if cu_seqlens is None:
        return [x.shape[1]] * x.shape[0]
    _validate_cu_seqlens(cu_seqlens, x.shape[1])
    return [int(v) for v in (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()]


def chunk_local_cumsum(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    _check_chunk_size(chunk_size)
    _check_3d('g', g)
    _check_contiguous('g', g)

    if cu_seqlens is not None:
        _require(g.shape[0] == 1, f'varlen g must have leading batch 1, got {g.shape[0]}')
        dense = unpack_varlen(g.to(torch.float32), cu_seqlens)
    else:
        _check_nonempty_batch('g', g)
        dense = g.to(torch.float32)

    original_tokens = dense.shape[1]
    chunked = _pad_and_reshape(dense, dim=1, chunk_size=chunk_size)
    chunked = chunked.cumsum(dim=2)
    out = chunked.reshape(dense.shape[0], -1, dense.shape[2])[:, :original_tokens].contiguous()

    if cu_seqlens is not None:
        return pack_varlen(out, cu_seqlens)
    return out


def _validate_kv_dims(name: str, x: torch.Tensor, head_dim: int = HEAD_DIM) -> None:
    _require(x.ndim == 4, f'{name} must be rank 4, got shape {tuple(x.shape)}')
    _require(x.shape[-1] == head_dim, f'{name}.shape[-1] must be {head_dim}, got {x.shape[-1]}')


def _expand_qk_to_value_heads(x: torch.Tensor, hv: int, name: str) -> torch.Tensor:
    hq = x.shape[2]
    _require(hv % hq == 0, f'Hv={hv} must be divisible by Hq={hq} for {name}')
    if hq == hv:
        return x
    return x.repeat_interleave(hv // hq, dim=2)


def _trim_or_pack_tokens(x: torch.Tensor, original_tokens: int, cu_seqlens: torch.Tensor | None) -> torch.Tensor:
    x = x.reshape(x.shape[0], -1, *x.shape[3:])[:, :original_tokens].contiguous()
    if cu_seqlens is not None:
        return pack_varlen(x, cu_seqlens)
    return x


def kkt_solve(
    k: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    _check_chunk_size(chunk_size)
    _validate_kv_dims('k', k)
    _check_3d('g_cumsum', g_cumsum)
    _check_3d('beta', beta)
    _check_contiguous('k', k)
    _check_contiguous('g_cumsum', g_cumsum)
    _check_contiguous('beta', beta)
    _require(g_cumsum.shape == beta.shape,
             f'g_cumsum and beta must have identical shape, got {tuple(g_cumsum.shape)} and {tuple(beta.shape)}')
    _require(k.shape[:2] == g_cumsum.shape[:2],
             f'k and g_cumsum must share batch/token dims, got {tuple(k.shape[:2])} and {tuple(g_cumsum.shape[:2])}')

    if cu_seqlens is not None:
        _require(k.shape[0] == 1, f'varlen k must have leading batch 1, got {k.shape[0]}')
        k_dense = unpack_varlen(k.to(torch.float32), cu_seqlens)
        g_dense = unpack_varlen(g_cumsum.to(torch.float32), cu_seqlens)
        beta_dense = unpack_varlen(beta.to(torch.float32), cu_seqlens)
    else:
        _check_nonempty_batch('k', k)
        k_dense = k.to(torch.float32)
        g_dense = g_cumsum.to(torch.float32)
        beta_dense = beta.to(torch.float32)

    original_tokens = k_dense.shape[1]
    hv = g_dense.shape[-1]
    k_dense = _expand_qk_to_value_heads(k_dense, hv, 'k')

    k_chunk = _pad_and_reshape(k_dense, dim=1, chunk_size=chunk_size)
    g_chunk = _pad_and_reshape(g_dense, dim=1, chunk_size=chunk_size)
    beta_chunk = _pad_and_reshape(beta_dense, dim=1, chunk_size=chunk_size)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=k.device))
    decay = torch.exp(g_chunk[:, :, :, None, :] - g_chunk[:, :, None, :, :])
    decay = decay.masked_fill(mask[None, None, :, :, None], 0.0)
    attn = torch.einsum('bnchk,bndhk->bnchd', k_chunk * beta_chunk.unsqueeze(-1), k_chunk)
    attn = attn * decay.swapaxes(-2, -1)

    x = -attn.swapaxes(2, 3).contiguous()
    for row_idx in range(1, chunk_size):
        row = x[..., row_idx, :row_idx].clone()
        sub = x[..., :row_idx, :row_idx].clone()
        x[..., row_idx, :row_idx] = row + (row.unsqueeze(-1) * sub).sum(-2)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=k.device)
    x = x + eye
    A = x.swapaxes(2, 3).contiguous()
    return _trim_or_pack_tokens(A, original_tokens, cu_seqlens)


def kkt_solve_without_cumsum(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = CHUNK_SIZE,
) -> torch.Tensor:
    _check_chunk_size(chunk_size)
    _validate_kv_dims('k', k)
    _check_3d('beta', beta)
    _check_contiguous('k', k)
    _check_contiguous('beta', beta)
    _require(k.shape[:2] == beta.shape[:2],
             f'k and beta must share batch/token dims, got {tuple(k.shape[:2])} and {tuple(beta.shape[:2])}')

    if cu_seqlens is not None:
        _require(k.shape[0] == 1, f'varlen k must have leading batch 1, got {k.shape[0]}')
        k_dense = unpack_varlen(k.to(torch.float32), cu_seqlens)
        beta_dense = unpack_varlen(beta.to(torch.float32), cu_seqlens)
    else:
        _check_nonempty_batch('k', k)
        k_dense = k.to(torch.float32)
        beta_dense = beta.to(torch.float32)

    original_tokens = k_dense.shape[1]
    hv = beta_dense.shape[-1]
    k_dense = _expand_qk_to_value_heads(k_dense, hv, 'k')

    k_chunk = _pad_and_reshape(k_dense, dim=1, chunk_size=chunk_size)
    beta_chunk = _pad_and_reshape(beta_dense, dim=1, chunk_size=chunk_size)

    gram = torch.einsum('bnrhk,bnchk->bnhrc', k_chunk, k_chunk)
    beta_rows = beta_chunk.permute(0, 1, 3, 2).unsqueeze(-1)
    lower = torch.tril(gram * beta_rows, diagonal=-1)
    eye = torch.eye(chunk_size, dtype=torch.float32, device=k.device)
    l = lower + eye

    a = torch.zeros_like(l)
    for row in range(chunk_size):
        for col in range(row):
            acc = -l[:, :, :, row, col]
            for mid in range(col + 1, row):
                acc = acc - l[:, :, :, row, mid] * a[:, :, :, mid, col]
            a[:, :, :, row, col] = acc
        a[:, :, :, row, row] = 1.0

    A = a.permute(0, 1, 3, 2, 4).contiguous()
    return _trim_or_pack_tokens(A, original_tokens, cu_seqlens)


def _fill_last_chunk_g(g_chunk: torch.Tensor, lengths: list[int], chunk_size: int) -> torch.Tensor:
    g_chunk = g_chunk.clone()
    for batch_idx, length in enumerate(lengths):
        last_size = length % chunk_size
        if last_size:
            last_chunk = length // chunk_size
            g_chunk[batch_idx, last_chunk, last_size:] = g_chunk[batch_idx, last_chunk, last_size - 1:last_size]
    return g_chunk


def _pack_chunk_states(h: torch.Tensor, lengths: list[int], chunk_size: int) -> torch.Tensor:
    total_chunks = sum(math.ceil(length / chunk_size) for length in lengths)
    out = torch.empty((1, total_chunks, *h.shape[2:]), dtype=h.dtype, device=h.device)
    offset = 0
    for batch_idx, length in enumerate(lengths):
        chunks = math.ceil(length / chunk_size)
        out[0, offset:offset + chunks] = h[batch_idx, :chunks]
        offset += chunks
    return out.contiguous()


def _dense_inputs_for_forward(q, k, v, A, g_cumsum, beta, cu_seqlens):
    if cu_seqlens is None:
        return (
            q.to(torch.float32),
            k.to(torch.float32),
            v.to(torch.float32),
            A.to(torch.float32),
            g_cumsum.to(torch.float32),
            beta.to(torch.float32),
            [q.shape[1]] * q.shape[0],
        )

    return (
        unpack_varlen(q.to(torch.float32), cu_seqlens),
        unpack_varlen(k.to(torch.float32), cu_seqlens),
        unpack_varlen(v.to(torch.float32), cu_seqlens),
        unpack_varlen(A.to(torch.float32), cu_seqlens),
        unpack_varlen(g_cumsum.to(torch.float32), cu_seqlens),
        unpack_varlen(beta.to(torch.float32), cu_seqlens),
        _sequence_lengths(q, cu_seqlens),
    )


def fused_chunk_gdr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_h: bool = False,
    chunk_size: int = CHUNK_SIZE,
):
    _check_chunk_size(chunk_size)
    _check_contiguous('q', q)
    _check_contiguous('k', k)
    _check_contiguous('v', v)
    _check_contiguous('A', A)
    _check_contiguous('g_cumsum', g_cumsum)
    _check_contiguous('beta', beta)
    _check_contiguous('initial_state', initial_state)
    _validate_kv_dims('q', q)
    _validate_kv_dims('k', k)
    _validate_kv_dims('v', v)
    _require(q.shape[:2] == k.shape[:2] == v.shape[:2],
             f'q, k, and v must share batch/token dims, got q={tuple(q.shape[:2])}, '
             f'k={tuple(k.shape[:2])}, v={tuple(v.shape[:2])}')
    _require(q.shape[2] == k.shape[2],
             f'q and k must share Hq, got q={q.shape[2]}, k={k.shape[2]}')
    _require(A.ndim == 4 and A.shape[-1] == chunk_size,
             f'A must have shape [B,T,Hv,{chunk_size}], got {tuple(A.shape)}')
    _require(v.shape[:3] == g_cumsum.shape,
             f'v and g_cumsum must share [B,T,Hv], got {tuple(v.shape[:3])} and {tuple(g_cumsum.shape)}')
    _require(A.shape[:3] == g_cumsum.shape,
             f'A and g_cumsum must share [B,T,Hv], got {tuple(A.shape[:3])} and {tuple(g_cumsum.shape)}')
    _require(beta.shape == g_cumsum.shape and beta.shape == v.shape[:3],
             f'beta, g_cumsum, and v must share [B,T,Hv], got beta={tuple(beta.shape)}, '
             f'g_cumsum={tuple(g_cumsum.shape)}, v={tuple(v.shape[:3])}')
    if cu_seqlens is None:
        _check_nonempty_batch('q', q)

    input_dtype = q.dtype
    q_dense, k_dense, v_dense, A_dense, g_dense, beta_dense, lengths = _dense_inputs_for_forward(
        q, k, v, A, g_cumsum, beta, cu_seqlens)

    batch, original_tokens, _hq, head_dim = q_dense.shape
    hv = v_dense.shape[2]
    q_dense = _expand_qk_to_value_heads(q_dense, hv, 'q')
    k_dense = _expand_qk_to_value_heads(k_dense, hv, 'k')
    scale = head_dim ** -0.5 if scale is None else scale

    k_chunk = _pad_and_reshape(k_dense, dim=1, chunk_size=chunk_size)
    v_chunk = _pad_and_reshape(v_dense, dim=1, chunk_size=chunk_size)
    A_chunk = _pad_and_reshape(A_dense, dim=1, chunk_size=chunk_size)
    beta_chunk = _pad_and_reshape(beta_dense, dim=1, chunk_size=chunk_size)
    g_chunk = _fill_last_chunk_g(_pad_and_reshape(g_dense, dim=1, chunk_size=chunk_size), lengths, chunk_size)

    k_beta = k_chunk * beta_chunk.unsqueeze(-1) * g_chunk.exp().unsqueeze(-1)
    v_beta = v_chunk * beta_chunk.unsqueeze(-1)
    w = torch.einsum('bnchd,bndhk->bnchk', A_chunk, k_beta)
    u = torch.einsum('bnchd,bndhv->bnchv', A_chunk, v_beta)

    if initial_state is None:
        state = torch.zeros((batch, hv, head_dim, head_dim), dtype=torch.float32, device=q.device)
    else:
        _require(initial_state.shape == (batch, hv, head_dim, head_dim),
                 f'initial_state must have shape {(batch, hv, head_dim, head_dim)}, got {tuple(initial_state.shape)}')
        state = initial_state.to(torch.float32).clone()

    h_chunks = []
    vn_chunks = []
    for chunk_idx in range(k_chunk.shape[1]):
        h_chunks.append(state.clone())
        v_new = u[:, chunk_idx] - torch.einsum('bchk,bhkv->bchv', w[:, chunk_idx], state)
        vn_chunks.append(v_new)
        state = state * g_chunk[:, chunk_idx, -1, :, None, None].exp()
        decay = torch.exp(g_chunk[:, chunk_idx, -1:, :, None] - g_chunk[:, chunk_idx, :, :, None])
        state = state + torch.einsum('bchk,bchv->bhkv', k_chunk[:, chunk_idx] * decay, v_new)

    h_dense = torch.stack(h_chunks, dim=1).contiguous()
    vn_dense = torch.stack(vn_chunks, dim=1).reshape(batch, -1, hv, head_dim)[:, :original_tokens].contiguous()

    q_chunk = _pad_and_reshape(q_dense * scale, dim=1, chunk_size=chunk_size)
    k_out_chunk = _pad_and_reshape(k_dense, dim=1, chunk_size=chunk_size)
    v_out_chunk = _pad_and_reshape(vn_dense, dim=1, chunk_size=chunk_size)
    g_out_chunk = _pad_and_reshape(g_dense, dim=1, chunk_size=chunk_size)

    upper_mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    decay_mask = torch.exp(g_out_chunk[:, :, :, None, :] - g_out_chunk[:, :, None, :, :])
    decay_mask = decay_mask.masked_fill(upper_mask[None, None, :, :, None], 0.0)
    attn = torch.einsum('bnchk,bndhk->bncdh', q_chunk, k_out_chunk) * decay_mask
    attn_inter = torch.einsum('bnchk,bnhkv->bnchv', q_chunk * g_out_chunk.exp().unsqueeze(-1), h_dense)
    o_dense = attn_inter + torch.einsum('bncdh,bndhv->bnchv', attn, v_out_chunk)
    o_dense = o_dense.reshape(batch, -1, hv, head_dim)[:, :original_tokens].contiguous()

    if cu_seqlens is not None:
        o = pack_varlen(o_dense, cu_seqlens)
        h_out = _pack_chunk_states(h_dense, lengths, chunk_size)
    else:
        o = o_dense
        h_out = h_dense

    o = o.to(input_dtype)
    if return_h:
        return o, state.contiguous(), h_out.contiguous()
    return o, state.contiguous()


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    return_intermediates: bool = False,
    chunk_size: int = CHUNK_SIZE,
):
    g_cumsum = chunk_local_cumsum(g, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    A = kkt_solve(k, g_cumsum, beta, cu_seqlens=cu_seqlens, chunk_size=chunk_size)
    if return_intermediates:
        o, final_state, h = fused_chunk_gdr_fwd(
            q, k, v, A, g_cumsum, beta, scale=scale, initial_state=initial_state,
            cu_seqlens=cu_seqlens, return_h=True, chunk_size=chunk_size)
        return o, final_state, g_cumsum, A, h
    return fused_chunk_gdr_fwd(
        q, k, v, A, g_cumsum, beta, scale=scale, initial_state=initial_state,
        cu_seqlens=cu_seqlens, return_h=False, chunk_size=chunk_size)


def is_available() -> bool:
    return True


def make_benchmark_task(run: RunCase, inputs: InputTensors, request: BenchmarkRequest,
                        device: torch.device) -> BenchmarkTask:
    from tests.turbomind.linear_attn.benchmark import base_row

    def execute():
        return chunk_gated_delta_rule_fwd(
            inputs.q,
            inputs.k,
            inputs.v,
            inputs.g,
            inputs.beta,
            initial_state=inputs.h0,
            cu_seqlens=inputs.offsets,
            chunk_size=run.chunk_size,
        )

    def validate():
        torch.cuda.reset_peak_memory_stats(device)
        actual_o, actual_state = execute()
        torch.cuda.synchronize(device)
        return {
            **diff_metrics('output', actual_o, actual_o),
            **diff_metrics('state', actual_state, actual_state),
        }

    return BenchmarkTask(
        base_row(
            run,
            'reference',
            cp_mode=request.cp_mode,
            cp_pattern=request.cp_pattern,
            cp_enabled=False,
        ),
        execute,
        validate=validate,
    )
