# Copyright (c) OpenMMLab. All rights reserved.
"""Speculative method registry for the TurboMind backend."""
from __future__ import annotations

import os.path as osp
from dataclasses import dataclass

from lmdeploy.archs import get_model_arch
from lmdeploy.utils import get_model

from .models.base import INPUT_MODELS
from .models.qwen3_5_mtp import Qwen3_5MtpTextModel
from .models.utils import source_model_config
from .weight_format import TrivialFormat, WeightFormatResolver


@dataclass(frozen=True)
class DraftWeightSpec:
    """How one speculative method's draft weights are found and mapped."""

    input_model: str
    weight_source: str
    prefix: str = ''
    quantized: bool = False


DRAFT_WEIGHT_SPECS: dict[str, DraftWeightSpec] = {
    'eagle3': DraftWeightSpec(input_model='qwen3-eagle3', weight_source='sidecar'),
}


# TurboMind registers MTP heads under the C++ name "mtp" (see
# TM_REGISTER_SPECULATIVE_MODEL("mtp", ...)); the CLI exposes model-specific
# aliases. Normalize so --speculative-algorithm qwen3_5_mtp (and the other MTP
# aliases) reaches the registered MTP model. Must stay in sync with the
# registered speculative-model names in src/turbomind/models/speculative/.
_MTP_ALIASES = {'qwen3_5_mtp': 'mtp', 'hy3_mtp': 'mtp', 'deepseek_mtp': 'mtp'}


def normalize_spec_method(method: str) -> str:
    """Normalize a CLI speculative name to the TurboMind name."""
    return _MTP_ALIASES.get(method, method)


def build_draft_model(speculative_config,
                      target_model,
                      target_model_path,
                      engine_data_type,
                      download_dir=None):
    """Build the draft weight mapper and resolve the checkpoint it reads."""
    method = normalize_spec_method(speculative_config.method)

    if method == 'mtp':
        target_text = target_model.text_model
        draft = Qwen3_5MtpTextModel(
            target_text.cfg, resolver=target_text._resolver)
        return draft, target_model_path

    spec = DRAFT_WEIGHT_SPECS.get(method)
    if spec is None:
        raise ValueError(f'TurboMind does not support speculative method {method!r}; '
                         f'supported: {sorted(DRAFT_WEIGHT_SPECS)}')

    if spec.weight_source == 'sidecar':
        if not speculative_config.model:
            raise ValueError(f'speculative method {method!r} requires a draft model path')
        path = speculative_config.model
        if not osp.exists(path):
            path = get_model(path, download_dir)
    else:
        path = target_model_path

    _, hf_config = get_model_arch(path)
    draft_config = source_model_config(hf_config)

    if spec.quantized:
        raise NotImplementedError(f'quantized draft weights are not supported yet ({method!r})')
    resolver = WeightFormatResolver(formats=[TrivialFormat(weight_dtype=engine_data_type)])

    model = INPUT_MODELS.get(spec.input_model)(draft_config, resolver=resolver, prefix=spec.prefix)
    return model, path
