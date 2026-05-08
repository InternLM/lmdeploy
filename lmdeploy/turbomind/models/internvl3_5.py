# Copyright (c) OpenMMLab. All rights reserved.
"""InternVL3.5 aggregate source model for TurboMind."""
from __future__ import annotations

from transformers import PretrainedConfig

from .base import INPUT_MODELS
from .qwen3 import Qwen3TextModel


def _cfg_get(cfg, name: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@INPUT_MODELS.register_module(name='internvl3_5')
class InternVL3_5Model:
    """Aggregate source model for Qwen3-backed InternVL3.5 checkpoints."""

    _text_pfx = 'language_model'
    _supported_inner_arch = 'Qwen3ForCausalLM'

    _uses_prefix = True

    def __init__(self, cfg: PretrainedConfig, *, resolver):
        llm_cfg = _cfg_get(cfg, 'llm_config')
        if llm_cfg is None:
            raise ValueError('InternVL3.5 TurboMind requires llm_config.')

        archs = _cfg_get(llm_cfg, 'architectures')
        if not archs:
            raise ValueError(
                'InternVL3.5 TurboMind requires llm_config.architectures.')

        inner_arch = archs[0]
        if inner_arch != self._supported_inner_arch:
            raise ValueError(
                'InternVL3.5 TurboMind currently supports only '
                f'{self._supported_inner_arch}, but got {inner_arch}.')

        self.text_model = Qwen3TextModel(llm_cfg, resolver=resolver)
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

    def model(self, pfx):
        self.text_model.model(pfx + self._text_pfx)
