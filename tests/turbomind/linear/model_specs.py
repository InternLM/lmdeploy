from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

TpAxis = Literal['input', 'output']
WeightKind = Literal['dense', 'replicated', 'moe_gate_up', 'moe_down']

SUPPORTED_TP_SIZES = (8, 4, 2, 1)
WEIGHT_KINDS = ('dense', 'replicated', 'moe_gate_up', 'moe_down')
MODEL_SPECS_PATH = Path(__file__).with_name('models.yaml')


@dataclass(frozen=True, kw_only=True)
class MoeSpec:
    expert_num: int
    experts_per_token: int

    def __post_init__(self) -> None:
        if self.expert_num <= 0:
            raise ValueError('expert_num_must_be_positive')
        if self.experts_per_token <= 0:
            raise ValueError('experts_per_token_must_be_positive')
        if self.experts_per_token > self.expert_num:
            raise ValueError('experts_per_token_exceeds_expert_num')


@dataclass(frozen=True, kw_only=True)
class WeightSpec:
    name: str
    kind: WeightKind
    output_dim: int
    input_dim: int
    tp_axis: TpAxis | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('weight_name_must_be_non_empty')
        if self.kind not in WEIGHT_KINDS:
            raise ValueError(f'unsupported_weight_kind_{self.kind}')
        if self.output_dim <= 0 or self.input_dim <= 0:
            raise ValueError('weight_dims_must_be_positive')
        if self.kind == 'dense':
            if self.tp_axis not in ('input', 'output'):
                raise ValueError('dense_weight_requires_tp_axis')
        elif self.tp_axis is not None:
            raise ValueError(f'{self.kind}_weight_forbids_tp_axis')
        if self.kind == 'moe_gate_up' and self.output_dim % 2:
            raise ValueError('moe_gate_up_output_dim_must_be_even')


@dataclass(frozen=True, kw_only=True)
class ModelSpec:
    name: str
    max_tp: int
    max_ep: int
    weights: tuple[WeightSpec, ...]
    moe: MoeSpec | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError('model_name_must_be_non_empty')
        if self.max_tp not in SUPPORTED_TP_SIZES:
            raise ValueError(f'unsupported_model_max_tp_{self.max_tp}')
        if self.max_ep <= 0:
            raise ValueError('model_max_ep_must_be_positive')
        if not self.weights:
            raise ValueError('model_weights_must_be_non_empty')

        names = tuple(weight.name for weight in self.weights)
        if len(names) != len(set(names)):
            raise ValueError(f'duplicate_weight_names_{self.name}')

        has_moe_weights = any(weight.kind in ('moe_gate_up', 'moe_down') for weight in self.weights)
        if has_moe_weights != (self.moe is not None):
            raise ValueError(f'moe_metadata_and_weights_disagree_{self.name}')
        if self.moe is None:
            if self.max_ep != 1:
                raise ValueError(f'dense_model_requires_max_ep_1_{self.name}')
            return

        if self.moe.expert_num % self.max_ep:
            raise ValueError(f'expert_num_not_divisible_by_max_ep_{self.name}')
        local_experts = self.moe.expert_num // self.max_ep
        if local_experts <= 1:
            raise ValueError(f'expert_parallel_requires_multiple_local_experts_{self.name}')
        if local_experts < self.moe.experts_per_token:
            raise ValueError(f'local_experts_less_than_experts_per_token_{self.name}')


def _make_weight(config: dict[str, Any]) -> WeightSpec:
    return WeightSpec(**config)


def _make_model(config: dict[str, Any]) -> ModelSpec:
    values = dict(config)
    moe_config = values.get('moe')
    values['moe'] = MoeSpec(**moe_config) if moe_config is not None else None
    values['weights'] = tuple(_make_weight(weight) for weight in values['weights'])
    return ModelSpec(**values)


def load_model_specs(path: Path = MODEL_SPECS_PATH) -> tuple[ModelSpec, ...]:
    config = yaml.safe_load(path.read_text())
    models = tuple(_make_model(model) for model in config['models'])
    names = tuple(model.name for model in models)
    if len(names) != len(set(names)):
        raise ValueError('duplicate_model_names')
    return models
