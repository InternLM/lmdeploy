# Copyright (c) OpenMMLab. All rights reserved.
"""TextModel — per-architecture model owning HF parsing and C++ configs."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .builders import NormBuilder, make_norm_config

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from .checkpoint import Prefix
    from .linear import Linear


class TextModel(ABC):
    """Text model: HF config -> C++ configs + weight commits.

    Subclass contract:
      - __init__ takes (cfg, *, resolver), calls super().__init__, then
        builds per-module C++ config templates as self._attn_cfg /
        self._ffn_cfg / self._moe_cfg / self._dn_cfg.
      - Topology methods (model/attn/ffn/moe/linear_attn/mla/...) take a
        Prefix argument and use Prefix arithmetic for tensor reads.
      - The base class provides no factory stubs; every subclass
        implements its own model(pfx) that calls
        builder.add_token_embeds / builder.add_lm_head on a
        TextModelBuilder for the root-level commits.
    """

    _loader_mappings: list = []

    def __init__(self, cfg: PretrainedConfig, *, resolver):
        self.cfg: PretrainedConfig = cfg
        self._resolver = resolver

    @property
    def _vocab_size(self) -> int:
        return self.cfg.vocab_size

    def bind_runtime(self, *, ctx, root_handles,
                     attn_tp, mlp_tp, model_tp):
        self._ctx = ctx
        self._root_handles = root_handles
        self._attn_tp = attn_tp
        self._mlp_tp = mlp_tp
        self._model_tp = model_tp

    def _linear(self, pfx: Prefix, *,
                optional: bool = False) -> Linear | None:
        return self._resolver.resolve(pfx, optional=optional)

    def norm(self, pfx, transform=None):
        weight = pfx.pop('weight')
        if transform is not None:
            weight = transform(weight)
        cfg = make_norm_config(
            dim=weight.shape[-1],
            norm_eps=self.cfg.rms_norm_eps,
        )
        m = NormBuilder(cfg, self._ctx)
        m.set_weight(weight)
        return m.build()

    def model(self, pfx: Prefix) -> None:
        raise NotImplementedError(
            f'{type(self).__name__}.model(pfx) must be overridden')
