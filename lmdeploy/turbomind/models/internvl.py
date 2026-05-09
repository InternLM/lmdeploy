# Copyright (c) OpenMMLab. All rights reserved.
"""InternVL aggregate source model for TurboMind (legacy InternVLChatModel and
HF-style InternVL/InternS1)."""
from __future__ import annotations

from transformers import PretrainedConfig

from ..supported_models import SUPPORTED_ARCHS
from .base import INPUT_MODELS


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@INPUT_MODELS.register_module(name='internvl')
class InternVLModel:
    """Aggregate source model for InternVL checkpoints with any registered text
    model."""

    def __init__(self, cfg: PretrainedConfig, *, resolver):
        llm_cfg = _cfg_get(cfg, 'llm_config')
        if llm_cfg is None:
            llm_cfg = _cfg_get(cfg, 'text_config')
        if llm_cfg is None:
            raise ValueError(
                'InternVL TurboMind requires llm_config or text_config.')

        archs = _cfg_get(llm_cfg, 'architectures')
        if not archs:
            raise ValueError(
                'InternVL TurboMind requires architectures on llm_config or text_config.')

        text_model_arch = archs[0]
        text_model_registered_name = SUPPORTED_ARCHS.get(text_model_arch)
        if text_model_registered_name is None:
            raise ValueError(
                f'InternVL text model architecture {text_model_arch!r} '
                'is not supported by TurboMind.')

        text_model_cls = INPUT_MODELS.get(text_model_registered_name)
        self.text_model = text_model_cls(llm_cfg, resolver=resolver)
        self.vision_model = None

    def bind_runtime(self, *, ctx, root_handles,
                     attn_tp, mlp_tp, model_tp):
        self.text_model.bind_runtime(
            ctx=ctx,
            root_handles=root_handles,
            attn_tp=attn_tp,
            mlp_tp=mlp_tp,
            model_tp=model_tp,
        )

    @property
    def _vocab_size(self):
        return self.text_model.cfg.vocab_size

    @property
    def _loader_mappings(self):
        return list(getattr(type(self.text_model), '_loader_mappings', []))

    def model(self, pfx):
        self.text_model.model(pfx + 'language_model')
