# Copyright (c) OpenMMLab. All rights reserved.
"""VisionModel — base class for native TurboMind vision sub-models."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from .checkpoint import Prefix
    from .linear import Linear


class VisionModel(ABC):
    """Vision model: HF config -> C++ configs, weights and runtime inputs.

    Subclasses own a vision sub-tree rooted at ``ModelRoot.vision_model``.
    They build that tree from checkpoint weights in :meth:`model` and convert
    frontend multimodal data to the corresponding TurboMind runtime input in
    :meth:`to_turbomind_multimodal`.
    """

    def __init__(self, cfg: PretrainedConfig, *, resolver):
        self.cfg: PretrainedConfig = cfg
        self._resolver = resolver

    def bind_runtime(self, *, ctx, root_handles, model_tp):
        """Bind runtime state shared by native vision model builders."""
        self._ctx = ctx
        self._root_handles = root_handles
        self._model_tp = model_tp

    def _linear(self, pfx: Prefix, *,
                optional: bool = False) -> Linear | None:
        return self._resolver.resolve(pfx, optional=optional)

    def model(self, pfx: Prefix) -> None:
        raise NotImplementedError(
            f'{type(self).__name__}.model(pfx) must be overridden')

    def to_turbomind_multimodal(self,
                                multimodal: list[dict[str, Any]]):
        raise NotImplementedError(
            f'{type(self).__name__}.to_turbomind_multimodal(multimodal) '
            'must be overridden')
