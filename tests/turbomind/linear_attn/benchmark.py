from __future__ import annotations

import argparse
import importlib
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from types import ModuleType, SimpleNamespace

import torch

from tests.turbomind.linear_attn.cases import (
    Fixed,
    Heads,
    InputCase,
    InputTensors,
    RunCase,
    Shape,
    Varlen,
    case_chunks,
    dtype_name,
    expand_inputs,
    input_work,
    make_input_tensors,
    run_cases,
)

VALID_BACKENDS = ('reference', 'turbomind', 'fla', 'flashqla')
VALID_CP_MODES = ('auto', 'off')
VALID_CP_PATTERNS = ('auto', 'warmup', 'fallback', 'alternating')
CP_FALLBACK_SEGMENT_LOG_DECAY = -5.0


def noop() -> None:
    return None


@dataclass(frozen=True)
class BenchmarkRequest:
    backend: str
    state_dtype: str
    chunk_size: int | None
    cp_mode: str
    cp_pattern: str
    validate_outputs: bool
    print_diffs: bool
    l2_flush: bool
    warmup: int
    iters: int


@dataclass
class BenchmarkTask:
    row: dict[str, object]
    run: Callable[[], object]
    prepare: Callable[[], object] = noop
    validate: Callable[[], dict[str, object] | None] = noop


def max_abs_diff(actual: torch.Tensor | None, expected: torch.Tensor | None) -> float:
    if actual is None and expected is None:
        return 0.0
    if actual is None or expected is None:
        return math.inf
    if actual.numel() == 0:
        return 0.0
    return float((actual.float() - expected.float()).abs().max().item())


def mean_abs_diff(actual: torch.Tensor | None, expected: torch.Tensor | None) -> float:
    if actual is None and expected is None:
        return 0.0
    if actual is None or expected is None:
        return math.inf
    if actual.numel() == 0:
        return 0.0
    return float((actual.float() - expected.float()).abs().mean().item())


def diff_metrics(name: str, actual: torch.Tensor | None, expected: torch.Tensor | None) -> dict[str, float]:
    return {
        f'{name}_max_diff': max_abs_diff(actual, expected),
        f'{name}_mean_diff': mean_abs_diff(actual, expected),
    }


def base_row(
    run: RunCase,
    backend: str,
    *,
    cp_mode: str = 'auto',
    cp_pattern: str = 'auto',
    cp_enabled: bool | None = None,
    **extra,
) -> dict[str, object]:
    row = {
        'case': run.name,
        'backend': backend,
        'input_dtype': dtype_name(run.input.input_dtype),
        'h0': run.input.has_h0,
        'state_dtype': run.state_dtype,
        'varlen': run.input.varlen,
        'batch': run.input.real_batch_size,
        'tokens': run.input.total_tokens,
        'hq': run.input.heads.hq,
        'hv': run.input.heads.hv,
        'chunk_size': run.chunk_size,
        'chunks': case_chunks(run),
        'cp_mode': cp_mode,
        'cp_pattern': cp_pattern,
    }
    if cp_enabled is not None:
        row['cp_enabled'] = cp_enabled
    row.update(extra)
    return row


def format_row(row: dict[str, object]) -> str:
    values = dict(row)
    parts = [str(values.pop('case'))]
    for key, value in values.items():
        if key == 'latency_ms':
            value = f'{float(value):.7f}'
        parts.append(f'{key}={value}')
    return ','.join(parts)


def _validate_cp_mode(cp_mode: str) -> None:
    if cp_mode not in VALID_CP_MODES:
        raise ValueError(f'unsupported_cp_mode {cp_mode}')


def _validate_cp_pattern(cp_pattern: str) -> None:
    if cp_pattern not in VALID_CP_PATTERNS:
        raise ValueError(f'unsupported_cp_pattern {cp_pattern}')


def validate_cp_request(cp_mode: str, cp_pattern: str, backend: str) -> None:
    _validate_cp_mode(cp_mode)
    _validate_cp_pattern(cp_pattern)
    if cp_pattern != 'auto' and backend not in ('turbomind', 'flashqla'):
        raise ValueError('cp_pattern_requires_cp_backend')


def enforce_cp_intent(row: dict[str, object], cp_mode: str) -> None:
    _validate_cp_mode(cp_mode)
    if cp_mode == 'auto':
        return
    cp_enabled = bool(row.get('cp_enabled', False))
    if cp_mode == 'off' and cp_enabled:
        raise ValueError('cp_selected')


def apply_cp_pattern(
    case: InputCase,
    inputs: InputTensors,
    cp_pattern: str,
    segment_tokens: int,
) -> InputTensors:
    _validate_cp_pattern(cp_pattern)
    if cp_pattern == 'auto':
        return inputs
    if segment_tokens <= 0:
        raise ValueError('cp_not_selected')

    g = torch.empty_like(inputs.g)
    rng = random.Random(case.seed)
    hv = case.heads.hv

    def fill_sequence(batch: int, sequence_begin: int, sequence_end: int) -> None:
        warmup_count = hv // 2
        if hv % 2:
            warmup_count += rng.getrandbits(1)
        starts_with_warmup = [True] * warmup_count + [False] * (hv - warmup_count)
        rng.shuffle(starts_with_warmup)

        for local_segment, segment_begin in enumerate(range(sequence_begin, sequence_end, segment_tokens)):
            segment_end = min(segment_begin + segment_tokens, sequence_end)
            fallback_value = CP_FALLBACK_SEGMENT_LOG_DECAY / (segment_end - segment_begin)

            if cp_pattern == 'warmup':
                values = [-1.0] * hv
            elif cp_pattern == 'fallback':
                values = [fallback_value] * hv
            else:
                values = [
                    -1.0 if start_warmup != bool(local_segment & 1) else fallback_value
                    for start_warmup in starts_with_warmup
                ]

            head_values = torch.tensor(values, dtype=g.dtype, device=g.device)
            g[batch, segment_begin:segment_end, :] = head_values

    if isinstance(case.layout, Fixed):
        for batch in range(case.layout.batch_size):
            fill_sequence(batch, 0, case.layout.seq_len)
    else:
        for begin, end in zip(case.layout.offsets[:-1], case.layout.offsets[1:]):
            fill_sequence(0, begin, end)

    return replace(inputs, g=g)


DEFAULT_L2_FLUSH_BYTES = 128 * 1024 * 1024


def _device_l2_cache_size(device: torch.device) -> int | None:
    props = torch.cuda.get_device_properties(device)
    for name in ('L2_cache_size', 'l2_cache_size'):
        value = getattr(props, name, None)
        if value:
            return int(value)
    return None


def l2_flush_bytes(device: torch.device) -> int:
    cache_size = _device_l2_cache_size(device)
    if cache_size is None:
        return DEFAULT_L2_FLUSH_BYTES
    return max(1, cache_size * 2)


class L2CacheFlusher:

    def __init__(self, device: torch.device):
        self.bytes = l2_flush_bytes(device)
        self._buffer = torch.empty(self.bytes, device=device, dtype=torch.uint8)
        self._fill_value = 0

    def flush(self) -> None:
        self._buffer.fill_(self._fill_value)
        self._fill_value = 255 if self._fill_value == 0 else 0


def time_task(task: BenchmarkTask, request: BenchmarkRequest, device: torch.device) -> dict[str, object]:
    if request.iters < 0:
        raise ValueError('iters_must_be_non_negative')
    if request.warmup < 0:
        raise ValueError('warmup_must_be_non_negative')

    validation_row = None
    if request.validate_outputs or request.print_diffs:
        validation_row = task.validate()

    if request.iters == 0:
        row = dict(task.row)
        row['latency_ms'] = 0.0
        row['l2_flush_bytes'] = 0
        if request.print_diffs and validation_row is not None:
            row.update(validation_row)
        return row

    for _ in range(request.warmup):
        task.prepare()
        task.run()

    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    flusher = L2CacheFlusher(device) if request.l2_flush else None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    elapsed_ms = 0.0

    for _ in range(request.iters):
        task.prepare()
        torch.cuda.synchronize(device)
        with torch.cuda.stream(stream):
            if flusher is not None:
                flusher.flush()
            start_event.record(stream)
            task.run()
            end_event.record(stream)
        end_event.synchronize()
        elapsed_ms += start_event.elapsed_time(end_event)

    row = dict(task.row)
    row['latency_ms'] = elapsed_ms / max(request.iters, 1)
    row['l2_flush_bytes'] = 0 if flusher is None else flusher.bytes
    if request.print_diffs and validation_row is not None:
        row.update(validation_row)
    return row


VALID_SUITES = ('smoke', 'quick', 'full', 'stress', 'custom')
ALL_HEADS = (
    Heads(hq=16, hv=16),
    Heads(hq=16, hv=32),
    Heads(hq=8, hv=16),
    Heads(hq=16, hv=48),
    Heads(hq=8, hv=24),
    Heads(hq=16, hv=64),
    Heads(hq=8, hv=32),
    Heads(hq=4, hv=16),
    Heads(hq=2, hv=8),
)
PRODUCTION_HEADS = (
    Heads(hq=16, hv=16),
    Heads(hq=16, hv=32),
    Heads(hq=8, hv=16),
    Heads(hq=8, hv=24),
    Heads(hq=4, hv=16),
    Heads(hq=2, hv=8),
)
BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
INPUT_DTYPES = (torch.bfloat16,)
STATE_DTYPES = ('f32', 'bf16')
CHUNK_SIZES = (64,)
MAX_WORK = 1_048_576
HIGH_THROUGHPUT_SEQ_LENS = (256, 512, 1024, 2048, 4096, 8192, 16384)
HIGH_THROUGHPUT_MIN_CONCURRENCY = 128
HIGH_THROUGHPUT_MAX_TOKENS = 65_536
HIGH_THROUGHPUT_MAX_WORK = 2_097_152
HIGH_THROUGHPUT_TARGET_PER_SEQ_LEN = 2
CP_SEQ_LENS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
CP_MIN_EFFECTIVE_CONCURRENCY = 64
CP_TARGET_PER_SEQ_LEN = 2
VARLEN_BATCH_SIZE = 8
VARLEN_MIN_SEQ_LEN = 256
VARLEN_TOTAL_TOKEN_TARGETS = (
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    40960,
    49152,
    57344,
    65536,
    81920,
    98304,
    114688,
    131072,
)
VARLEN_PROFILE_CYCLE = ('skewed', 'staircase')
VARLEN_STRESS_COUNT = 64
VQ_RATIO_RANK = {4: 4, 3: 3, 2: 2, 1: 1}


def shape_concurrency(shape: Shape) -> int:
    layout, heads = shape
    return layout.real_batch_size * heads.hv


def shape_work(shape: Shape) -> int:
    layout, heads = shape
    if isinstance(layout, Varlen):
        return layout.total_tokens * heads.hv
    return layout.real_batch_size * layout.total_tokens * heads.hv


def fixed_layout(shape: Shape) -> Fixed:
    layout, _heads = shape
    if not isinstance(layout, Fixed):
        raise TypeError('shape must use a fixed layout')
    return layout


def unique_shape_concurrency(
    shapes: Iterable[Shape],
    *,
    score: Callable[[Shape], tuple[int, ...]],
) -> tuple[Shape, ...]:
    selected: dict[int, Shape] = {}
    for shape in shapes:
        key = shape_concurrency(shape)
        current = selected.get(key)
        if current is None or score(shape) > score(current):
            selected[key] = shape
    return tuple(selected[key] for key in sorted(selected))


def larger_batch_score(shape: Shape) -> tuple[int]:
    layout, _heads = shape
    return (layout.real_batch_size,)


def recurrent_shape_candidates() -> tuple[Shape, ...]:
    return tuple(
        (Fixed(batch_size=batch_size, seq_len=1), heads)
        for batch_size in BATCH_SIZES
        for heads in ALL_HEADS
    )


def recurrent_inputs() -> tuple[InputCase, ...]:
    shapes = unique_shape_concurrency(
        recurrent_shape_candidates(),
        score=larger_batch_score,
    )
    return expand_inputs(
        shapes,
        input_dtypes=INPUT_DTYPES,
        h0=(False,),
        seed_base=10_000,
    )


def high_throughput_shape_candidates(seq_len: int) -> tuple[Shape, ...]:
    return tuple(
        shape
        for shape in (
            (Fixed(batch_size=batch_size, seq_len=seq_len), heads)
            for batch_size in BATCH_SIZES
            for heads in PRODUCTION_HEADS
        )
        if shape_concurrency(shape) >= HIGH_THROUGHPUT_MIN_CONCURRENCY
        if fixed_layout(shape).batch_size * seq_len <= HIGH_THROUGHPUT_MAX_TOKENS
        if shape_work(shape) <= HIGH_THROUGHPUT_MAX_WORK
    )


def best_high_throughput_shape_per_head(shapes: Iterable[Shape]) -> tuple[Shape, ...]:
    selected: dict[Heads, Shape] = {}
    for shape in shapes:
        layout, heads = shape
        current = selected.get(heads)
        score = (-shape_concurrency(shape), -layout.batch_size * layout.seq_len)
        if current is None or score > (
            -shape_concurrency(current),
            -fixed_layout(current).batch_size * fixed_layout(current).seq_len,
        ):
            selected[heads] = shape
    return tuple(selected[heads] for heads in PRODUCTION_HEADS if heads in selected)


def high_throughput_selection_score(
    shape: Shape,
    head_counts: dict[Heads, int],
) -> tuple[int, int, int]:
    _layout, heads = shape
    return (
        -head_counts.get(heads, 0),
        VQ_RATIO_RANK[heads.hv // heads.hq],
        shape_concurrency(shape),
    )


def high_throughput_inputs() -> tuple[InputCase, ...]:
    selected = []
    head_counts: dict[Heads, int] = {}
    for seq_len in HIGH_THROUGHPUT_SEQ_LENS:
        candidates = list(
            best_high_throughput_shape_per_head(
                high_throughput_shape_candidates(seq_len),
            )
        )
        for _ in range(min(HIGH_THROUGHPUT_TARGET_PER_SEQ_LEN, len(candidates))):
            shape = max(candidates, key=lambda item: high_throughput_selection_score(item, head_counts))
            candidates.remove(shape)
            selected.append(shape)
            _layout, heads = shape
            head_counts[heads] = head_counts.get(heads, 0) + 1

    return expand_inputs(
        tuple(selected),
        input_dtypes=INPUT_DTYPES,
        h0=(False,),
        seed_base=20_000,
    )


def cp_effective_concurrency(shape: Shape) -> int:
    layout = fixed_layout(shape)
    return shape_concurrency(shape) * layout.seq_len // 1024


def cp_shape_candidates(seq_len: int) -> tuple[Shape, ...]:
    return tuple(
        shape
        for shape in (
            (Fixed(batch_size=batch_size, seq_len=seq_len), heads)
            for batch_size in BATCH_SIZES
            for heads in PRODUCTION_HEADS
        )
        if shape_concurrency(shape) <= 32
        if cp_effective_concurrency(shape) >= CP_MIN_EFFECTIVE_CONCURRENCY
        if shape_work(shape) <= MAX_WORK
    )


def best_cp_shape_per_head(shapes: Iterable[Shape]) -> tuple[Shape, ...]:
    selected: dict[Heads, Shape] = {}
    for shape in shapes:
        _layout, heads = shape
        current = selected.get(heads)
        score = (shape_concurrency(shape), fixed_layout(shape).batch_size)
        if current is None or score > (shape_concurrency(current), fixed_layout(current).batch_size):
            selected[heads] = shape
    return tuple(selected[heads] for heads in PRODUCTION_HEADS if heads in selected)


def cp_selection_score(shape: Shape, head_counts: dict[Heads, int]) -> tuple[int, int, int, int]:
    _layout, heads = shape
    return (
        -head_counts.get(heads, 0),
        VQ_RATIO_RANK[heads.hv // heads.hq],
        shape_concurrency(shape),
        fixed_layout(shape).batch_size,
    )


def cp_inputs() -> tuple[InputCase, ...]:
    selected = []
    head_counts: dict[Heads, int] = {}
    for seq_len in CP_SEQ_LENS:
        candidates = list(best_cp_shape_per_head(cp_shape_candidates(seq_len)))
        for _ in range(min(CP_TARGET_PER_SEQ_LEN, len(candidates))):
            shape = max(candidates, key=lambda item: cp_selection_score(item, head_counts))
            candidates.remove(shape)
            selected.append(shape)
            _layout, heads = shape
            head_counts[heads] = head_counts.get(heads, 0) + 1

    return expand_inputs(
        tuple(selected),
        input_dtypes=INPUT_DTYPES,
        h0=(False,),
        seed_base=30_000,
    )


def offsets_from_lengths(lengths: tuple[int, ...]) -> tuple[int, ...]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return tuple(offsets)


def skewed_varlen_lengths(total_tokens: int) -> tuple[int, ...]:
    base_total = VARLEN_BATCH_SIZE * VARLEN_MIN_SEQ_LEN
    if total_tokens < base_total:
        raise ValueError(f'total_tokens={total_tokens} is too small for varlen batch size')

    extra = total_tokens - base_total
    weights = (64, 32, 16, 8, 4, 2, 1, 1)
    increments = [extra * weight // sum(weights) for weight in weights]
    increments[0] += extra - sum(increments)
    return tuple(VARLEN_MIN_SEQ_LEN + increment for increment in increments)


def staircase_varlen_lengths(total_tokens: int) -> tuple[int, ...]:
    base_total = VARLEN_BATCH_SIZE * VARLEN_MIN_SEQ_LEN
    if total_tokens < base_total:
        raise ValueError(f'total_tokens={total_tokens} is too small for varlen batch size')

    extra = total_tokens - base_total
    weights = (1, 2, 3, 4, 5, 6, 7, 8)
    increments = [extra * weight // sum(weights) for weight in weights]
    increments[-1] += extra - sum(increments)
    return tuple(VARLEN_MIN_SEQ_LEN + increment for increment in increments)


def varlen_profile(index: int) -> str:
    return VARLEN_PROFILE_CYCLE[index % len(VARLEN_PROFILE_CYCLE)]


def varlen_lengths(total_tokens: int, profile: str) -> tuple[int, ...]:
    if profile == 'skewed':
        return skewed_varlen_lengths(total_tokens)
    if profile == 'staircase':
        return staircase_varlen_lengths(total_tokens)
    raise ValueError(f'unsupported varlen profile {profile}')


def valid_varlen_heads(total_tokens: int) -> tuple[Heads, ...]:
    return tuple(heads for heads in PRODUCTION_HEADS if total_tokens * heads.hv <= MAX_WORK)


def varlen_selection_score(
    heads: Heads,
    *,
    head_counts: dict[Heads, int],
    hv_counts: dict[int, int],
) -> tuple[int, int, int, int]:
    return (
        -head_counts.get(heads, 0),
        VQ_RATIO_RANK[heads.hv // heads.hq],
        -hv_counts.get(heads.hv, 0),
        heads.hv,
    )


def varlen_shapes(total_token_targets: tuple[int, ...]) -> tuple[Shape, ...]:
    selected: dict[int, Shape] = {}
    head_counts: dict[Heads, int] = {}
    hv_counts: dict[int, int] = {}

    indexed_targets = tuple(enumerate(total_token_targets))
    for index, total_tokens in sorted(indexed_targets, key=lambda item: item[1], reverse=True):
        valid_heads = valid_varlen_heads(total_tokens)
        if not valid_heads:
            raise ValueError(f'no production heads satisfy workload cap for total_tokens={total_tokens}')

        heads = max(
            valid_heads,
            key=lambda item: varlen_selection_score(item, head_counts=head_counts, hv_counts=hv_counts),
        )
        profile = varlen_profile(index)
        layout = Varlen(offsets=offsets_from_lengths(varlen_lengths(total_tokens, profile)))
        selected[index] = (layout, heads)
        head_counts[heads] = head_counts.get(heads, 0) + 1
        hv_counts[heads.hv] = hv_counts.get(heads.hv, 0) + 1

    return tuple(selected[index] for index, _total_tokens in indexed_targets)


def stress_varlen_total_token_targets(*, seed: int, count: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    candidates = [
        total_tokens
        for total_tokens in (factor * 1024 for factor in range(3, 129))
        if total_tokens not in VARLEN_TOTAL_TOKEN_TARGETS
    ]
    if count > len(candidates):
        raise ValueError(f'stress varlen count={count} exceeds available extra targets')

    rng.shuffle(candidates)
    return tuple(sorted(candidates[:count]))


def varlen_inputs(*, seed_base: int = 40_000) -> tuple[InputCase, ...]:
    return expand_inputs(
        varlen_shapes(VARLEN_TOTAL_TOKEN_TARGETS),
        input_dtypes=INPUT_DTYPES,
        h0=(False,),
        seed_base=seed_base,
    )


def stress_varlen_inputs(
    *,
    seed: int,
    count: int = VARLEN_STRESS_COUNT,
    seed_base: int = 50_000,
) -> tuple[InputCase, ...]:
    return expand_inputs(
        varlen_shapes(stress_varlen_total_token_targets(seed=seed, count=count)),
        input_dtypes=INPUT_DTYPES,
        h0=(False,),
        seed_base=seed_base,
    )


def h0_coverage_inputs() -> tuple[InputCase, ...]:
    shapes = (
        (Fixed(batch_size=2, seq_len=2048), Heads(hq=4, hv=16)),
        (
            Varlen(offsets=(0, 512, 1536, 2048, 4096)),
            Heads(hq=8, hv=16),
        ),
    )
    return expand_inputs(
        shapes,
        input_dtypes=INPUT_DTYPES,
        h0=(True,),
        seed_base=60_000,
    )


@dataclass(frozen=True)
class CaseFamilies:
    recurrent: tuple[InputCase, ...]
    high_throughput: tuple[InputCase, ...]
    cp: tuple[InputCase, ...]
    varlen: tuple[InputCase, ...]
    h0_coverage: tuple[InputCase, ...]
    stress_varlen: tuple[InputCase, ...]


@dataclass(frozen=True)
class SuiteRequest:
    suite: str
    state_dtypes: tuple[str, ...] = ('f32',)
    include_h0: bool = False
    chunk_sizes: tuple[int, ...] = CHUNK_SIZES


def parse_state_dtypes(value: str) -> tuple[str, ...]:
    if value == 'all':
        return STATE_DTYPES
    if value in STATE_DTYPES:
        return (value,)
    raise ValueError(f'unsupported state dtype {value}')


def custom_run(args) -> RunCase:
    if args.seq_len is None or args.hq is None or args.hv is None:
        raise ValueError('custom benchmark requires --seq-len, --hq, --hv')
    if args.seq_len < 1 or args.hq < 1 or args.hv < 1 or args.batch_size < 1:
        raise ValueError('custom benchmark requires positive --seq-len, --hq, --hv, and --batch-size')
    case = InputCase(
        layout=Fixed(batch_size=args.batch_size, seq_len=args.seq_len),
        heads=Heads(hq=args.hq, hv=args.hv),
        input_dtype=torch.bfloat16,
        has_h0=False,
        seed=9001,
    )
    return RunCase(
        input=case,
        state_dtype=args.state_dtype,
        chunk_size=resolve_benchmark_chunk_size(case, args.chunk_size, args.device),
    )


def custom_runs(args) -> tuple[RunCase, ...]:
    runs = []
    for state_dtype in parse_state_dtypes(args.state_dtype):
        run_args = dict(vars(args))
        run_args['state_dtype'] = state_dtype
        runs.append(custom_run(SimpleNamespace(**run_args)))
    return tuple(runs)


def make_families() -> CaseFamilies:
    return CaseFamilies(
        recurrent=recurrent_inputs(),
        high_throughput=high_throughput_inputs(),
        cp=cp_inputs(),
        varlen=varlen_inputs(seed_base=40_000),
        h0_coverage=h0_coverage_inputs(),
        stress_varlen=stress_varlen_inputs(seed=4200, seed_base=50_000),
    )


def smallest(inputs: tuple[InputCase, ...]) -> InputCase:
    return min(inputs, key=lambda case: (input_work(case), case.real_batch_size, case.name))


def smoke_inputs(families: CaseFamilies) -> tuple[InputCase, ...]:
    return (
        smallest(families.recurrent),
        smallest(families.high_throughput),
        smallest(families.cp),
        smallest(families.varlen),
    )


def quick_inputs(families: CaseFamilies) -> tuple[InputCase, ...]:
    return tuple(
        case for case in (
            *families.recurrent,
            *families.high_throughput,
            *families.cp,
            *families.varlen,
        )
        if case.total_tokens <= 16384
        if case.real_batch_size <= 64
    )


def full_inputs(families: CaseFamilies) -> tuple[InputCase, ...]:
    return (
        *families.recurrent,
        *families.high_throughput,
        *families.cp,
        *families.varlen,
    )


def stress_inputs(families: CaseFamilies) -> tuple[InputCase, ...]:
    return (
        *full_inputs(families),
        *families.stress_varlen,
    )


def suite_inputs(request: SuiteRequest, families: CaseFamilies) -> tuple[InputCase, ...]:
    if request.suite == 'smoke':
        inputs = smoke_inputs(families)
    elif request.suite == 'quick':
        inputs = quick_inputs(families)
    elif request.suite == 'full':
        inputs = full_inputs(families)
    elif request.suite == 'stress':
        inputs = stress_inputs(families)
    elif request.suite == 'custom':
        raise ValueError('custom suite requires explicit inputs')
    else:
        raise ValueError(f'unsupported suite {request.suite}')

    if request.include_h0:
        inputs = (*inputs, *families.h0_coverage)
    return inputs


def select_runs(request: SuiteRequest, families: CaseFamilies) -> tuple[RunCase, ...]:
    return run_cases(
        suite_inputs(request, families),
        state_dtypes=request.state_dtypes,
        chunk_sizes=request.chunk_sizes,
    )


BACKEND_MODULES = {
    'reference': 'tests.turbomind.linear_attn.reference',
    'turbomind': 'tests.turbomind.linear_attn.turbomind_gated_delta_rule',
    'fla': 'tests.turbomind.linear_attn.fla_gated_delta_rule',
    'flashqla': 'tests.turbomind.linear_attn.flashqla_gated_delta_rule',
}


BACKEND_UNSUPPORTED_REASONS = (
    'turbomind_requires_bf16_qkv',
    'flashqla_requires_chunk32',
    'flashqla_requires_chunk64',
    'fla_requires_chunk64',
    'cp_selected',
    'cp_pattern_requires_cp_backend',
    'cp_not_selected',
    'cp_pattern_requires_multiple_segments',
)


def get_backend_module(name: str) -> ModuleType:
    try:
        module_name = BACKEND_MODULES[name]
    except KeyError as exc:
        raise ValueError(f'unsupported backend {name}') from exc
    return importlib.import_module(module_name)


def unavailable_row(run: RunCase, request: BenchmarkRequest) -> dict[str, object]:
    return base_row(
        run,
        request.backend,
        cp_mode=request.cp_mode,
        cp_pattern=request.cp_pattern,
        status='unavailable',
    )


def unsupported_row(run: RunCase, request: BenchmarkRequest, reason: str) -> dict[str, object]:
    return base_row(
        run,
        request.backend,
        cp_mode=request.cp_mode,
        cp_pattern=request.cp_pattern,
        status='unsupported',
        reason=reason,
    )


def run_case(run: RunCase, request: BenchmarkRequest, device: torch.device) -> dict[str, object]:
    try:
        validate_cp_request(request.cp_mode, request.cp_pattern, request.backend)
    except ValueError as exc:
        reason = str(exc)
        if reason not in BACKEND_UNSUPPORTED_REASONS:
            raise
        return unsupported_row(run, request, reason)

    backend = get_backend_module(request.backend)
    if not backend.is_available():
        return unavailable_row(run, request)

    try:
        validate = getattr(backend, 'validate_benchmark_case', None)
        if validate is not None:
            validate(run, request)
    except ValueError as exc:
        reason = str(exc)
        if reason not in BACKEND_UNSUPPORTED_REASONS:
            raise
        return unsupported_row(run, request, reason)

    inputs = make_input_tensors(run.input, device=device)
    try:
        task = backend.make_benchmark_task(run, inputs, request, device)
    except ValueError as exc:
        reason = str(exc)
        if reason not in BACKEND_UNSUPPORTED_REASONS:
            raise
        return unsupported_row(run, request, reason)
    row = time_task(task, request, device)
    try:
        enforce_cp_intent(row, request.cp_mode)
    except ValueError as exc:
        reason = str(exc)
        if reason not in BACKEND_UNSUPPORTED_REASONS:
            raise
        return unsupported_row(run, request, reason)
    return row


CP_UNSUPPORTED_REASONS = BACKEND_UNSUPPORTED_REASONS


def parse_csv(value: str, valid: Iterable[str], name: str) -> list[str]:
    parts = [item.strip() for item in value.split(',')]
    if not parts or any(item == '' for item in parts):
        raise ValueError(f'{name} must not contain empty values')

    valid_set = set(valid)
    invalid = [item for item in parts if item not in valid_set]
    if invalid:
        raise ValueError(f'{name} contains invalid values {invalid}; valid values are {list(valid)}')

    selected = []
    seen = set()
    for item in parts:
        if item not in seen:
            selected.append(item)
            seen.add(item)
    return selected


def request_from_args(args, *, backend: str, run: RunCase) -> BenchmarkRequest:
    return BenchmarkRequest(
        backend=backend,
        state_dtype=run.state_dtype,
        chunk_size=args.chunk_size,
        cp_mode=args.cp_mode,
        cp_pattern=args.cp_pattern,
        validate_outputs=not args.skip_validate,
        print_diffs=args.print_diffs,
        l2_flush=not args.no_l2_flush,
        warmup=args.warmup,
        iters=args.iters,
    )


def parse_chunk_size(value: str) -> int | None:
    if value == 'auto':
        return None
    try:
        chunk_size = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('chunk size must be auto, 1, 16, 32, or 64') from exc
    if chunk_size not in (1, 16, 32, 64):
        raise argparse.ArgumentTypeError('chunk size must be auto, 1, 16, 32, or 64')
    return chunk_size


def _automatic_chunk_size(device: str) -> int:
    major, minor = torch.cuda.get_device_capability(torch.device(device))
    arch = major * 100 + minor * 10
    if arch < 900:
        return 16
    if arch == 900:
        return 64
    if arch == 1200:
        return 32
    raise ValueError(f'automatic GDR chunk selection is not supported for arch={arch}')


def _is_recurrent_input(case: InputCase) -> bool:
    return isinstance(case.layout, Fixed) and case.layout.seq_len == 1 and case.input_dtype == torch.bfloat16


def resolve_benchmark_chunk_size(case: InputCase, requested: int | None, device: str) -> int:
    if requested is not None:
        return requested
    if _is_recurrent_input(case):
        return 1
    return _automatic_chunk_size(device)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='GDR benchmark runner.')
    parser.add_argument('--suite', choices=VALID_SUITES, default='smoke')
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--state-dtype', choices=('f32', 'bf16', 'all'), default='f32')
    parser.add_argument('--include-h0', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--hq', type=int, default=None)
    parser.add_argument('--hv', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--backend', default='reference')
    parser.add_argument('--chunk-size', type=parse_chunk_size, default='auto')
    parser.add_argument('--cp-mode', choices=VALID_CP_MODES, default='auto')
    parser.add_argument('--cp-pattern', choices=VALID_CP_PATTERNS, default='auto')
    parser.add_argument('--print-diffs', action='store_true')
    parser.add_argument('--no-l2-flush', action='store_true')
    parser.add_argument('--skip-validate', action='store_true')
    return parser


def _parse_csv_or_error(parser: argparse.ArgumentParser, value: str, valid: Iterable[str], name: str) -> list[str]:
    try:
        return parse_csv(value, valid, name)
    except ValueError as exc:
        parser.error(str(exc))


def _select_runs_or_error(parser: argparse.ArgumentParser, args) -> tuple[RunCase, ...]:
    try:
        if args.suite == 'custom':
            return custom_runs(args)
        families = make_families()
        automatic_chunk_size = _automatic_chunk_size(args.device) if args.chunk_size is None else args.chunk_size
        suite_request = SuiteRequest(
            suite=args.suite,
            state_dtypes=parse_state_dtypes(args.state_dtype),
            include_h0=args.include_h0,
            chunk_sizes=(automatic_chunk_size,),
        )
        runs = select_runs(suite_request, families)
        if args.chunk_size is None:
            runs = tuple(
                replace(run, chunk_size=resolve_benchmark_chunk_size(run.input, None, args.device))
                for run in runs
            )
        return runs
    except ValueError as exc:
        parser.error(str(exc))


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.iters < 0 or args.warmup < 0 or args.batch_size < 1:
        parser.error('--iters and --warmup must be >= 0; --batch-size must be >= 1')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise SystemExit('CUDA is required for the GDR benchmark')

    device = torch.device(args.device)
    backends = _parse_csv_or_error(parser, args.backend, VALID_BACKENDS, '--backend')
    runs = _select_runs_or_error(parser, args)

    print(f'mode={args.suite}')
    for run in runs:
        for backend in backends:
            request = request_from_args(args, backend=backend, run=run)
            try:
                row = run_case(run, request, device)
            except ValueError as exc:
                reason = str(exc)
                if reason not in CP_UNSUPPORTED_REASONS:
                    raise
                row = unsupported_row(run, request, reason)
            print(format_row(row))
