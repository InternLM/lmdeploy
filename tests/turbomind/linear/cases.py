from __future__ import annotations

from dataclasses import dataclass, replace
from typing import cast

from tests.turbomind.linear.model_specs import (
    SUPPORTED_TP_SIZES,
    ModelSpec,
    MoeSpec,
    TpAxis,
    WeightSpec,
    load_model_specs,
)

DTYPE_ALIASES = {
    'bf16': 'bf16',
    'fp16': 'fp16',
    'fp8_e4m3': 'fp8_e4m3',
    'uint4': 'uint4',
    'fp4_e2m1': 'fp4_e2m1',
}

VALID_SUITES = ('smoke', 'quick', 'full', 'custom')

# Powers-of-two cover aligned CTA tiles; odd / tile-residue Ms hit epilogue tails.
SMOKE_BATCHES = (1, 3, 16)
QUICK_BATCHES = (1, 3, 16, 17, 64, 65, 256, 1024, 4096)
FULL_BATCHES = (
    1,
    2,
    3,
    4,
    5,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8233,
    16384,
    16385,
    32768,
)


@dataclass(frozen=True, kw_only=True)
class TypeSpec:
    """Dtype / quant axis (mirrors testbed_v3 TestParameter ctor fields)."""

    name: str
    data_type: str
    weight_type: str
    input_type: str
    group_size: int

    def __post_init__(self) -> None:
        for field, value in (
            ('data_type', self.data_type),
            ('weight_type', self.weight_type),
            ('input_type', self.input_type),
        ):
            if value not in DTYPE_ALIASES:
                raise ValueError(f'unsupported_{field}_{value}')
        if self.group_size < 0:
            raise ValueError('group_size_must_be_non_negative')


@dataclass(frozen=True, kw_only=True)
class ShapeSpec:
    """Problem geometry axis (dims + MoE layout)."""

    name: str
    input_dim: int
    output_dim: int
    expert_num: int
    experts_per_token: int
    combine_experts: bool
    tp_axis: TpAxis
    max_tp: int
    max_ep: int
    # True: token-major x + f2n indices (w1/gate). False: expert-packed x, offsets only (w2/down).
    moe_indexed: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.output_dim <= 0:
            raise ValueError('dims_must_be_positive')
        if self.expert_num < 0:
            raise ValueError('expert_num_must_be_non_negative')
        if self.expert_num == 0 and self.experts_per_token != 0:
            raise ValueError('dense_shape_requires_experts_per_token_0')
        if self.expert_num > 0 and self.experts_per_token <= 0:
            raise ValueError('moe_shape_requires_positive_experts_per_token')
        if self.expert_num == 0 and self.moe_indexed:
            raise ValueError('dense_shape_cannot_be_moe_indexed')
        if self.tp_axis not in ('input', 'output'):
            raise ValueError(f'unsupported_tp_axis_{self.tp_axis}')
        if self.max_tp <= 0 or self.max_ep <= 0:
            raise ValueError('parallel_caps_must_be_positive')
        tp_dim = self.input_dim if self.tp_axis == 'input' else self.output_dim
        if tp_dim % self.max_tp:
            raise ValueError('tp_dimension_not_divisible_by_max_tp')
        if self.expert_num == 0:
            if self.max_ep != 1:
                raise ValueError('dense_shape_requires_max_ep_1')
        else:
            if self.expert_num % self.max_ep:
                raise ValueError('expert_num_not_divisible_by_max_ep')
            local_experts = self.expert_num // self.max_ep
            if local_experts <= 1:
                raise ValueError('expert_parallel_requires_multiple_local_experts')
            if local_experts < self.experts_per_token:
                raise ValueError('local_experts_less_than_experts_per_token')
            inter_size = self.output_dim // 2 if self.tp_axis == 'output' else self.input_dim
            if inter_size % self.max_tp or inter_size // self.max_tp < 256:
                raise ValueError('expert_inter_size_per_tp_less_than_256')


@dataclass(frozen=True, kw_only=True)
class LinearCase:
    """One cell of the type × shape product."""

    name: str
    input_dim: int
    output_dim: int
    data_type: str
    weight_type: str
    input_type: str
    group_size: int
    expert_num: int
    experts_per_token: int
    combine_experts: bool
    moe_indexed: bool
    type_name: str
    shape_name: str
    tp_axis: TpAxis
    max_tp: int
    max_ep: int
    tp: int = 1
    ep: int = 1
    # Optional SM90 gate/up block-pack + kGatedSilu epilogue.
    fuse_silu: bool = False

    def __post_init__(self) -> None:
        for field, value in (
            ('data_type', self.data_type),
            ('weight_type', self.weight_type),
            ('input_type', self.input_type),
        ):
            if value not in DTYPE_ALIASES:
                raise ValueError(f'unsupported_{field}_{value}')
        if self.expert_num < 0:
            raise ValueError('expert_num_must_be_non_negative')
        if self.expert_num == 0 and self.experts_per_token != 0:
            raise ValueError('dense_case_requires_experts_per_token_0')
        if self.expert_num > 0 and self.experts_per_token <= 0:
            raise ValueError('moe_case_requires_positive_experts_per_token')
        if self.expert_num == 0 and self.moe_indexed:
            raise ValueError('dense_case_cannot_be_moe_indexed')
        if self.tp_axis not in ('input', 'output'):
            raise ValueError(f'unsupported_tp_axis_{self.tp_axis}')
        if self.max_tp <= 0 or self.max_ep <= 0 or self.tp <= 0 or self.ep <= 0:
            raise ValueError('parallel_sizes_must_be_positive')
        if self.tp > self.max_tp or self.ep > self.max_ep:
            raise ValueError('parallel_size_exceeds_case_cap')
        if self.tp > 1 and self.ep > 1:
            raise ValueError('combined_tp_ep_case_not_supported')
        if self.fuse_silu and not self.shape_name.endswith('_gate_up'):
            raise ValueError('fuse_silu_requires_gate_up_shape')
        if self.fuse_silu and self.expert_num > 0 and not self.moe_indexed:
            raise ValueError('moe_fuse_silu_requires_indexed_gate_up')
        if self.fuse_silu and self.weight_type not in ('fp8_e4m3', 'bf16'):
            raise ValueError('fuse_silu_requires_fp8_or_bf16_weights')


@dataclass(frozen=True, kw_only=True)
class RunCase:
    case: LinearCase
    batch_size: int

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError('batch_size_must_be_positive')


def _compose(type_spec: TypeSpec, shape: ShapeSpec, *, fuse_silu: bool = False) -> LinearCase:
    name = f'{shape.name}__{type_spec.name}'
    if fuse_silu:
        name = f'{name}__fuse_silu'
    return LinearCase(
        name=name,
        input_dim=shape.input_dim,
        output_dim=shape.output_dim,
        data_type=type_spec.data_type,
        weight_type=type_spec.weight_type,
        input_type=type_spec.input_type,
        group_size=type_spec.group_size,
        expert_num=shape.expert_num,
        experts_per_token=shape.experts_per_token,
        combine_experts=shape.combine_experts,
        moe_indexed=shape.moe_indexed,
        type_name=type_spec.name,
        shape_name=shape.name,
        tp_axis=shape.tp_axis,
        max_tp=shape.max_tp,
        max_ep=shape.max_ep,
        fuse_silu=fuse_silu,
    )


# --- Types: commented combos from test_gemm_v2 / testbed_v3 ---
# QuantizeGroupwise supports: fp16×uint4, fp16×fp4, bf16×fp4 (not bf16×uint4).
#
# TypeSpec.name matches GemmDesc type substring (desc.h to_string):
#   {Ta}[{Qa}]_{Tb}[{Qb}]_{Tc}  with dtype tokens f16/bf16/e4m3/e2m1/u4
#   and glued QuantDesc (k128/b128/k32). Order is act → weight → output.

TYPE_BF16 = TypeSpec(name='bf16_bf16_bf16', data_type='bf16', weight_type='bf16', input_type='bf16', group_size=0)
TYPE_BF16_E4M3B128_BF16 = TypeSpec(
    name='bf16_e4m3b128_bf16',
    data_type='bf16',
    weight_type='fp8_e4m3',
    input_type='bf16',
    group_size=128,
)
TYPE_E4M3K128_E4M3B128_BF16 = TypeSpec(
    name='e4m3k128_e4m3b128_bf16',
    data_type='bf16',
    weight_type='fp8_e4m3',
    input_type='fp8_e4m3',
    group_size=128,
)
TYPE_BF16_E2M1K32_BF16 = TypeSpec(
    name='bf16_e2m1k32_bf16',
    data_type='bf16',
    weight_type='fp4_e2m1',
    input_type='bf16',
    group_size=32,
)
TYPE_F16 = TypeSpec(name='f16_f16_f16', data_type='fp16', weight_type='fp16', input_type='fp16', group_size=0)
TYPE_F16_U4K128_F16 = TypeSpec(
    name='f16_u4k128_f16',
    data_type='fp16',
    weight_type='uint4',
    input_type='fp16',
    group_size=128,
)
TYPE_F16_E2M1K32_F16 = TypeSpec(
    name='f16_e2m1k32_f16',
    data_type='fp16',
    weight_type='fp4_e2m1',
    input_type='fp16',
    group_size=32,
)
TYPE_F16_E4M3B128_F16 = TypeSpec(
    name='f16_e4m3b128_f16',
    data_type='fp16',
    weight_type='fp8_e4m3',
    input_type='fp16',
    group_size=128,
)

ALL_TYPES: tuple[TypeSpec, ...] = (
    TYPE_BF16,
    TYPE_BF16_E4M3B128_BF16,
    TYPE_E4M3K128_E4M3B128_BF16,
    TYPE_BF16_E2M1K32_BF16,
    TYPE_F16,
    TYPE_F16_U4K128_F16,
    TYPE_F16_E2M1K32_F16,
    TYPE_F16_E4M3B128_F16,
)

SMOKE_TYPES: tuple[TypeSpec, ...] = (TYPE_BF16_E4M3B128_BF16,)


# --- Shapes: curated model weight configurations loaded from YAML ---


def _moe_max_tp(model: ModelSpec, weight: WeightSpec) -> int:
    if weight.kind == 'moe_gate_up':
        inter_size = weight.output_dim // 2
    else:
        inter_size = weight.input_dim

    for tp in SUPPORTED_TP_SIZES:
        if tp <= model.max_tp and inter_size % tp == 0 and inter_size // tp >= 256:
            return tp

    raise ValueError(
        f'no_expert_safe_tp_{model.name}_{weight.name}'
        f'_inter_size_{inter_size}_max_tp_{model.max_tp}'
    )


def _make_shape(model: ModelSpec, weight: WeightSpec) -> ShapeSpec:
    is_moe = weight.kind in ('moe_gate_up', 'moe_down')
    if weight.kind == 'dense':
        tp_axis = cast(TpAxis, weight.tp_axis)
        max_tp = model.max_tp
    elif weight.kind == 'replicated':
        tp_axis = 'output'
        max_tp = 1
    else:
        tp_axis = 'output' if weight.kind == 'moe_gate_up' else 'input'
        max_tp = _moe_max_tp(model, weight)

    if is_moe:
        moe = cast(MoeSpec, model.moe)
        expert_num = moe.expert_num
        experts_per_token = moe.experts_per_token
    else:
        expert_num = 0
        experts_per_token = 0

    return ShapeSpec(
        name=f'{model.name}_{weight.name}',
        input_dim=weight.input_dim,
        output_dim=weight.output_dim,
        expert_num=expert_num,
        experts_per_token=experts_per_token,
        combine_experts=is_moe,
        tp_axis=tp_axis,
        max_tp=max_tp,
        max_ep=model.max_ep if is_moe else 1,
        moe_indexed=weight.kind == 'moe_gate_up',
    )


MODEL_SPECS: tuple[ModelSpec, ...] = load_model_specs()
ALL_SHAPES: tuple[ShapeSpec, ...] = tuple(
    _make_shape(model, weight)
    for model in MODEL_SPECS
    for weight in model.weights
)
_SHAPE_BY_NAME = {shape.name: shape for shape in ALL_SHAPES}
SMOKE_SHAPES: tuple[ShapeSpec, ...] = (_SHAPE_BY_NAME['llama2_7b_o'],)


def is_supported(type_spec: TypeSpec, shape: ShapeSpec) -> bool:
    """Drop type×shape cells with no TurboMind kernel / quant path.

    - uint4 MoE: gemm dispatch has no SM90 f16×u4 grouped kernel today
      (``No feasible kernel ... sm90_f16_u4k128_..._ibb_...``).
    - fp16×fp4: SM90 MXF4 configs are registered for bfloat16 only
      (``sm90_16816_4.cu``); half×e2m1 dense hits ``..._f16_e2m1k32_f16_...`` miss.
    - fp16×fp8: SM90 E4M3 configs use ``bfloat16_t`` as Tc only
      (``sm90_16816_8.cu``); half Tc hits ``..._e4m3..._f16_tnt_...`` miss.
    - fp16 MoE: SM90 gemm dispatch has no fp16 indexed/blocked grouped kernel
      (``No feasible kernel ... sm90_f16_f16_f16_tnt_ibb_...``); the GMMA
      grouped kernels are bf16-only.
    """
    if type_spec.weight_type == 'uint4' and shape.expert_num > 0:
        return False
    if type_spec.data_type == 'fp16' and type_spec.weight_type == 'fp4_e2m1':
        return False
    if type_spec.data_type == 'fp16' and type_spec.weight_type == 'fp8_e4m3':
        return False
    if type_spec.data_type == 'fp16' and type_spec.weight_type == 'fp16' and shape.expert_num > 0:
        return False
    return True


def supports_fuse_silu(type_spec: TypeSpec, shape: ShapeSpec) -> bool:
    """FP8/BF16 SM90 gate_up can optionally use block-pack + kGatedSilu.

    BF16 pairs 64-wide gate/up blocks; FP8 pairs 128-wide blocks.
    """
    if not shape.name.endswith('_gate_up'):
        return False
    if shape.expert_num > 0 and not shape.moe_indexed:
        return False
    if type_spec.weight_type not in ('fp8_e4m3', 'bf16'):
        return False
    block = 128 if type_spec.weight_type == 'fp8_e4m3' else 64
    return shape.output_dim % (2 * block) == 0


def expand_cases(
    types: tuple[TypeSpec, ...] | None = None,
    shapes: tuple[ShapeSpec, ...] | None = None,
) -> tuple[LinearCase, ...]:
    ts = types if types is not None else ALL_TYPES
    ss = shapes if shapes is not None else ALL_SHAPES
    cases: list[LinearCase] = []
    for t in ts:
        for s in ss:
            if not is_supported(t, s):
                continue
            cases.append(_compose(t, s, fuse_silu=False))
            if supports_fuse_silu(t, s):
                cases.append(_compose(t, s, fuse_silu=True))
    return tuple(cases)


ALL_CASES: tuple[LinearCase, ...] = expand_cases()


def case_by_name() -> dict[str, LinearCase]:
    return {c.name: c for c in ALL_CASES}


def select_types(suite: str, type_names: tuple[str, ...] | None) -> tuple[TypeSpec, ...]:
    by_name = {t.name: t for t in ALL_TYPES}
    if type_names:
        missing = [n for n in type_names if n not in by_name]
        if missing:
            raise ValueError(f'unknown_types_{missing}')
        return tuple(by_name[n] for n in type_names)
    if suite == 'smoke':
        return SMOKE_TYPES
    if suite in ('quick', 'full', 'custom'):
        return ALL_TYPES
    raise ValueError(f'unknown_suite_{suite}')


def select_shapes(suite: str, shape_names: tuple[str, ...] | None) -> tuple[ShapeSpec, ...]:
    by_name = {s.name: s for s in ALL_SHAPES}
    if shape_names:
        missing = [n for n in shape_names if n not in by_name]
        if missing:
            raise ValueError(f'unknown_shapes_{missing}')
        return tuple(by_name[n] for n in shape_names)
    if suite == 'smoke':
        return SMOKE_SHAPES
    if suite in ('quick', 'full', 'custom'):
        return ALL_SHAPES
    raise ValueError(f'unknown_suite_{suite}')


def select_cases(
    suite: str,
    case_names: tuple[str, ...] | None = None,
    type_names: tuple[str, ...] | None = None,
    shape_names: tuple[str, ...] | None = None,
) -> tuple[LinearCase, ...]:
    if case_names:
        by_name = case_by_name()
        missing = [n for n in case_names if n not in by_name]
        if missing:
            raise ValueError(f'unknown_cases_{missing}')
        return tuple(by_name[n] for n in case_names)
    return expand_cases(select_types(suite, type_names), select_shapes(suite, shape_names))


def select_batches(suite: str, batches: tuple[int, ...] | None) -> tuple[int, ...]:
    if batches is not None:
        if not batches:
            raise ValueError('batches_must_be_non_empty')
        return batches
    if suite == 'smoke':
        return SMOKE_BATCHES
    if suite == 'quick':
        return QUICK_BATCHES
    if suite == 'full':
        return FULL_BATCHES
    raise ValueError('custom_suite_requires_explicit_batches')


def _select_parallel_sizes(name: str, sizes: tuple[int, ...] | None) -> tuple[int, ...]:
    values = sizes if sizes is not None else (1,)
    if not values:
        raise ValueError(f'{name}_sizes_must_be_non_empty')
    if any(value <= 0 for value in values):
        raise ValueError(f'{name}_sizes_must_be_positive')
    return tuple(dict.fromkeys(values))


def _parallel_variants(
    case: LinearCase,
    tps: tuple[int, ...],
    eps: tuple[int, ...],
) -> tuple[tuple[int, int], ...]:
    variants = [(tp, 1) for tp in tps]
    if case.expert_num > 0:
        variants.extend((1, ep) for ep in eps)
    return tuple(dict.fromkeys(variants))


def _slice_case(case: LinearCase, tp: int, ep: int) -> LinearCase | None:
    if tp > case.max_tp or ep > case.max_ep:
        return None

    input_dim = case.input_dim
    output_dim = case.output_dim
    tp_dim = input_dim if case.tp_axis == 'input' else output_dim
    if tp_dim % tp:
        return None
    if case.tp_axis == 'input':
        input_dim //= tp
    else:
        output_dim //= tp

    expert_num = case.expert_num
    if expert_num:
        if expert_num % ep:
            return None
        expert_num //= ep

    if case.group_size and input_dim % case.group_size:
        return None
    if case.fuse_silu:
        block = 128 if case.weight_type == 'fp8_e4m3' else 64
        if output_dim % (2 * block):
            return None

    return replace(
        case,
        input_dim=input_dim,
        output_dim=output_dim,
        expert_num=expert_num,
        tp=tp,
        ep=ep,
    )


def expand_suite(
    suite: str,
    case_names: tuple[str, ...] | None = None,
    batches: tuple[int, ...] | None = None,
    type_names: tuple[str, ...] | None = None,
    shape_names: tuple[str, ...] | None = None,
    tps: tuple[int, ...] | None = None,
    eps: tuple[int, ...] | None = None,
) -> tuple[RunCase, ...]:
    if suite not in VALID_SUITES:
        raise ValueError(f'unknown_suite_{suite}')
    selected_tps = _select_parallel_sizes('tp', tps)
    selected_eps = _select_parallel_sizes('ep', eps)
    selected_batches = select_batches(suite, batches)
    runs: list[RunCase] = []
    for case in select_cases(suite, case_names, type_names, shape_names):
        for tp, ep in _parallel_variants(case, selected_tps, selected_eps):
            sliced = _slice_case(case, tp, ep)
            if sliced is None:
                continue
            runs.extend(RunCase(case=sliced, batch_size=m) for m in selected_batches)
    if not runs:
        raise ValueError('no_supported_runs')
    return tuple(runs)
