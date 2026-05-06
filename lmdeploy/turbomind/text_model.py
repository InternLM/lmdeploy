# Copyright (c) OpenMMLab. All rights reserved.
"""TextModel — per-architecture model owning HF parsing and C++ configs."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

from .builders import NormBuilder, make_norm_config

if TYPE_CHECKING:
    from transformers import PretrainedConfig



class TextModel(ABC):
    """Text model: HF config -> C++ configs + weight commits.

    Subclass contract:
      - __init__ takes (cfg, *, resolver), calls super().__init__, then
        builds per-module C++ config templates as self._attn_cfg /
        self._ffn_cfg / self._moe_cfg / self._dn_cfg.
      - Factory method NAMES (attn/ffn/moe/linear_attn/mla/norm/...)
        are a convention for readability, NOT a protocol. Signatures
        may differ across subclasses. The base class provides no
        factory stubs; every subclass implements its own model()
        that calls root.add_token_embeds / root.add_lm_head on a
        TextModelBuilder for the root-level commits.
    """

    _loader_mappings: list = []


    # ------------------------------------------------------------------
    # Construction / parsing
    # ------------------------------------------------------------------

    def __init__(self, cfg: PretrainedConfig, *, resolver):
        """Store local config and shared runtime helpers.

        Source-model subclasses own architecture-specific field reads. Shared
        utilities in ``models.utils`` build common C++ module configs.
        """
        self.cfg: PretrainedConfig = cfg
        self._resolver = resolver

    @property
    def _vocab_size(self) -> int:
        return self.cfg.vocab_size


    # ------------------------------------------------------------------
    # Runtime binding (called by ModelLoader after model_comm exists)
    # ------------------------------------------------------------------

    def bind_runtime(self, *, ctx, root_handles,
                     attn_tp, mlp_tp, model_tp):
        self._ctx = ctx
        self._root_handles = root_handles
        self._attn_tp = attn_tp
        self._mlp_tp = mlp_tp
        self._model_tp = model_tp

    def set_params(self, params: dict):
        self.params = params

    # ------------------------------------------------------------------
    # Checkpoint access helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    def _linear(self, pfx: str, *, optional: bool = False):
        return self._resolver.resolve(self.params, pfx, optional=optional)



    # ------------------------------------------------------------------
    # Norm factories (shared across all models)
    # ------------------------------------------------------------------

    def norm(self, weight, *, dim=None):
        """Build a NormBuilder for *weight* under this model's contexts.

        ``dim`` defaults to ``weight.shape[-1]``.
        """
        cfg = make_norm_config(
            dim=dim if dim is not None else weight.shape[-1],
            norm_eps=self.cfg.rms_norm_eps,
        )
        m = NormBuilder(cfg, self._ctx)
        m.set_weight(weight)
        return m.build()
