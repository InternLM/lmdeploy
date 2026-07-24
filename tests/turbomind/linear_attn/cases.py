from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product

import torch
import torch.nn.functional as F

HEAD_DIM = 128
CHUNK_SIZE = 64


@dataclass(frozen=True, kw_only=True)
class Fixed:
    batch_size: int
    seq_len: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if self.seq_len <= 0:
            raise ValueError('seq_len must be positive')

    @property
    def name(self) -> str:
        return f'b{self.batch_size}_s{self.seq_len}'

    @property
    def total_tokens(self) -> int:
        return self.seq_len

    @property
    def real_batch_size(self) -> int:
        return self.batch_size


@dataclass(frozen=True, kw_only=True)
class Varlen:
    offsets: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.offsets) < 2:
            raise ValueError('varlen offsets must contain at least two entries')
        if self.offsets[0] != 0:
            raise ValueError('varlen offsets must start at 0')
        if any(end < start for start, end in zip(self.offsets, self.offsets[1:])):
            raise ValueError('varlen offsets must be nondecreasing')

    @property
    def name(self) -> str:
        return f'v{len(self.offsets) - 1}_t{self.offsets[-1]}'

    @property
    def total_tokens(self) -> int:
        return self.offsets[-1]

    @property
    def real_batch_size(self) -> int:
        return len(self.offsets) - 1


@dataclass(frozen=True, kw_only=True)
class Heads:
    hq: int
    hv: int

    def __post_init__(self) -> None:
        if self.hq < 1 or self.hv < 1:
            raise ValueError('hq and hv must be positive')
        if self.hv % self.hq != 0:
            raise ValueError(f'hv={self.hv} must be divisible by hq={self.hq}')

    @property
    def name(self) -> str:
        return f'hq{self.hq}_hv{self.hv}'


def dtype_name(input_dtype: torch.dtype) -> str:
    names = {
        torch.bfloat16: 'bf16',
        torch.float16: 'f16',
        torch.float32: 'f32',
    }
    try:
        return names[input_dtype]
    except KeyError as exc:
        raise ValueError(f'unsupported input dtype {input_dtype}') from exc


@dataclass(frozen=True, kw_only=True)
class InputCase:
    layout: Fixed | Varlen
    heads: Heads
    input_dtype: torch.dtype
    has_h0: bool
    seed: int

    def __post_init__(self) -> None:
        dtype_name(self.input_dtype)

    @property
    def name(self) -> str:
        h0 = 'h0' if self.has_h0 else 'noh0'
        return f'{self.layout.name}_{self.heads.name}_{dtype_name(self.input_dtype)}_{h0}'

    @property
    def total_tokens(self) -> int:
        return self.layout.total_tokens

    @property
    def real_batch_size(self) -> int:
        return self.layout.real_batch_size

    @property
    def varlen(self) -> bool:
        return isinstance(self.layout, Varlen)


@dataclass(frozen=True, kw_only=True)
class RunCase:
    input: InputCase
    state_dtype: str
    chunk_size: int

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError('chunk_size must be positive')
        state_torch_dtype(self.state_dtype)

    @property
    def name(self) -> str:
        return f'{self.input.name}_state{self.state_dtype}_chunk{self.chunk_size}'


@dataclass
class InputTensors:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    h0: torch.Tensor | None
    offsets: torch.Tensor | None


@dataclass
class StateBuffer:
    storage: torch.Tensor
    ptrs: torch.Tensor
    tma_descs: torch.Tensor | None = None

    def reset(self, h0: torch.Tensor | None) -> None:
        if h0 is None:
            self.storage.zero_()
        else:
            _validate_h0_shape(h0, self.storage)
            self.storage.copy_(h0.to(self.storage.dtype))


@dataclass
class GroupedStateBuffer:
    blocks: torch.Tensor
    ptrs: torch.Tensor
    hv: int
    heads_per_block: int

    @property
    def num_head_groups(self) -> int:
        return self.blocks.shape[1]

    @property
    def layers_per_block(self) -> int:
        return self.blocks.shape[2]

    def logical(self, layer: int) -> torch.Tensor:
        groups = [self.blocks[:, group, layer] for group in range(self.num_head_groups)]
        return torch.cat(groups, dim=1)[:, :self.hv]

    def set_logical(self, layer: int, state: torch.Tensor) -> None:
        for group in range(self.num_head_groups):
            begin = group * self.heads_per_block
            end = min(begin + self.heads_per_block, self.hv)
            self.blocks[:, group, layer, :end - begin].copy_(
                state[:, begin:end].to(self.blocks.dtype))


def _randn(shape: tuple[int, ...], generator: torch.Generator, device: torch.device | str) -> torch.Tensor:
    return torch.randn(shape, generator=generator, device=device, dtype=torch.float32)


def _pack_dense(dense: torch.Tensor, offsets: tuple[int, ...]) -> torch.Tensor:
    total = offsets[-1]
    out = torch.empty((1, total, *dense.shape[2:]), dtype=dense.dtype, device=dense.device)
    for batch_idx in range(len(offsets) - 1):
        start = offsets[batch_idx]
        end = offsets[batch_idx + 1]
        out[0, start:end] = dense[batch_idx, :end - start]
    return out.contiguous()


def make_input_tensors(case: InputCase, device: torch.device | str = 'cuda') -> InputTensors:
    generator = torch.Generator(device=device)
    generator.manual_seed(case.seed)

    if isinstance(case.layout, Fixed):
        leading_shape = (case.layout.batch_size, case.layout.seq_len)
        offsets = None
    else:
        lengths = [
            case.layout.offsets[idx + 1] - case.layout.offsets[idx]
            for idx in range(len(case.layout.offsets) - 1)
        ]
        leading_shape = (len(lengths), max(lengths))
        offsets = torch.tensor(case.layout.offsets, device=device, dtype=torch.int32)

    q = (_randn((*leading_shape, case.heads.hq, HEAD_DIM), generator, device) * 0.2).to(case.input_dtype)
    k = (_randn((*leading_shape, case.heads.hq, HEAD_DIM), generator, device) * 0.2).to(case.input_dtype)
    q_float = q.float()
    k_float = k.float()
    q = (q_float * torch.rsqrt(q_float.square().sum(-1, keepdim=True) + 1e-6)).to(case.input_dtype)
    k = (k_float * torch.rsqrt(k_float.square().sum(-1, keepdim=True) + 1e-6)).to(case.input_dtype)
    v = (_randn((*leading_shape, case.heads.hv, HEAD_DIM), generator, device) * 0.2).to(case.input_dtype)
    g = F.logsigmoid(_randn((*leading_shape, case.heads.hv), generator, device)) / 16.0
    beta = torch.sigmoid(_randn((*leading_shape, case.heads.hv), generator, device))

    h0 = None
    if case.has_h0:
        h0 = _randn((case.real_batch_size, case.heads.hv, HEAD_DIM, HEAD_DIM), generator, device) * 0.05

    if isinstance(case.layout, Varlen):
        q = _pack_dense(q, case.layout.offsets)
        k = _pack_dense(k, case.layout.offsets)
        v = _pack_dense(v, case.layout.offsets)
        g = _pack_dense(g, case.layout.offsets)
        beta = _pack_dense(beta, case.layout.offsets)

    return InputTensors(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        h0=None if h0 is None else h0.contiguous(),
        offsets=offsets,
    )


def make_packed_qkv_views(inputs: InputTensors) -> InputTensors:
    batch, tokens, hq, head_dim = inputs.q.shape
    hv = inputs.v.shape[2]
    conv_dim = (2 * hq + hv) * head_dim
    packed = torch.empty(
        (batch, tokens, conv_dim),
        dtype=inputs.q.dtype,
        device=inputs.q.device,
    )
    q = torch.as_strided(
        packed,
        (batch, tokens, hq, head_dim),
        (tokens * conv_dim, conv_dim, head_dim, 1),
        storage_offset=0,
    )
    k = torch.as_strided(
        packed,
        (batch, tokens, hq, head_dim),
        (tokens * conv_dim, conv_dim, head_dim, 1),
        storage_offset=hq * head_dim,
    )
    v = torch.as_strided(
        packed,
        (batch, tokens, hv, head_dim),
        (tokens * conv_dim, conv_dim, head_dim, 1),
        storage_offset=2 * hq * head_dim,
    )
    q.copy_(inputs.q)
    k.copy_(inputs.k)
    v.copy_(inputs.v)
    return replace(inputs, q=q, k=k, v=v)


def state_torch_dtype(state_dtype: str) -> torch.dtype:
    if state_dtype == 'bf16':
        return torch.bfloat16
    if state_dtype == 'f16':
        return torch.float16
    if state_dtype == 'f32':
        return torch.float32
    raise ValueError(f'unsupported state dtype {state_dtype}')


def _validate_h0_shape(h0: torch.Tensor, storage: torch.Tensor) -> None:
    if h0.shape != storage.shape:
        raise ValueError(f'h0 shape {tuple(h0.shape)} does not match state storage shape {tuple(storage.shape)}')


def make_state_buffer(h0: torch.Tensor | None, run: RunCase, device: torch.device) -> StateBuffer:
    state_dtype = state_torch_dtype(run.state_dtype)
    shape = (run.input.real_batch_size, run.input.heads.hv, HEAD_DIM, HEAD_DIM)
    storage = torch.empty(shape, device=device, dtype=state_dtype)
    if h0 is None:
        storage.zero_()
    else:
        _validate_h0_shape(h0, storage)
        storage.copy_(h0.to(state_dtype))
    ptrs = torch.tensor([storage[i].data_ptr() for i in range(storage.shape[0])], device=device, dtype=torch.int64)
    return StateBuffer(storage=storage, ptrs=ptrs)


def make_grouped_state_buffer(
    initial_state: torch.Tensor,
    *,
    state_dtype: torch.dtype,
    layers_per_block: int,
    heads_per_block: int,
    layer: int,
) -> GroupedStateBuffer:
    batch, hv, _, _ = initial_state.shape
    groups = (hv + heads_per_block - 1) // heads_per_block
    blocks = torch.full(
        (batch, groups, layers_per_block, heads_per_block, HEAD_DIM, HEAD_DIM),
        0.125,
        dtype=state_dtype,
        device=initial_state.device,
    )
    fixture = GroupedStateBuffer(
        blocks=blocks,
        ptrs=torch.tensor(
            [[blocks[b, g].data_ptr() for g in range(groups)] for b in range(batch)],
            dtype=torch.int64,
            device=initial_state.device,
        ),
        hv=hv,
        heads_per_block=heads_per_block,
    )
    fixture.set_logical(layer, initial_state)
    return fixture


Shape = tuple[Fixed | Varlen, Heads]


def input_cases(
    *,
    layouts: tuple[Fixed | Varlen, ...],
    heads: tuple[Heads, ...],
    input_dtypes: tuple[torch.dtype, ...],
    h0: tuple[bool, ...],
    seed_base: int,
) -> tuple[InputCase, ...]:
    return tuple(
        InputCase(layout=layout, heads=head, input_dtype=input_dtype, has_h0=has_h0, seed=seed_base + idx)
        for idx, (layout, head, input_dtype, has_h0) in enumerate(product(layouts, heads, input_dtypes, h0))
    )


def expand_inputs(
    shapes: tuple[Shape, ...],
    *,
    input_dtypes: tuple[torch.dtype, ...],
    h0: tuple[bool, ...],
    seed_base: int,
) -> tuple[InputCase, ...]:
    return tuple(
        InputCase(layout=layout, heads=heads, input_dtype=input_dtype, has_h0=has_h0, seed=seed_base + idx)
        for idx, ((layout, heads), input_dtype, has_h0) in enumerate(product(shapes, input_dtypes, h0))
    )


def run_cases(
    inputs: tuple[InputCase, ...],
    *,
    state_dtypes: tuple[str, ...],
    chunk_sizes: tuple[int, ...],
) -> tuple[RunCase, ...]:
    return tuple(
        RunCase(input=input_case, state_dtype=state_dtype, chunk_size=chunk_size)
        for input_case, state_dtype, chunk_size in product(inputs, state_dtypes, chunk_sizes)
    )


def input_work(case: InputCase) -> int:
    if case.varlen:
        return case.total_tokens * case.heads.hv
    else:
        return case.real_batch_size * case.total_tokens * case.heads.hv

def case_chunks(run: RunCase) -> int:
    layout = run.input.layout
    if isinstance(layout, Fixed):
        return (layout.seq_len + run.chunk_size - 1) // run.chunk_size
    return sum(
        (layout.offsets[idx + 1] - layout.offsets[idx] + run.chunk_size - 1) // run.chunk_size
        for idx in range(len(layout.offsets) - 1)
    )
