# Copyright (c) OpenMMLab. All rights reserved.
"""TextModel — per-architecture model owning HF parsing and C++ configs."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from .builders import NormBuilder, ParallelGroup, make_norm_config

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
                     attn_tp, mlp_tp, ep_size, model_tp):
        self._ctx = ctx
        self._root_handles = root_handles
        self._attn_tp = attn_tp
        self._mlp_tp = mlp_tp
        # EP is a relabel of MLP-TP (TurboMind asserts mlp_tp_size == ep and
        # mlp_tp_rank == ep_rank), so the per-GPU EP ranks live in
        # ``self._mlp_tp``; only the EP size is tracked separately.
        self._ep_size_val = max(1, ep_size or 1)
        self._model_tp = model_tp

    def _ep_size(self) -> int:
        return self._ep_size_val

    def _ep_group(self) -> ParallelGroup | None:
        """EP group for MoeBuilder, or None when EP is disabled."""
        return self._mlp_tp if self._ep_size() > 1 else None

    def _ffn_tp_group(self) -> ParallelGroup:
        if self._ep_size() > 1:
            return ParallelGroup(1, None)
        return self._mlp_tp

    def _expert_active_mask(self, expert_num: int, expert_idx: int):
        ep_size = self._ep_size()
        if ep_size <= 1:
            return None
        ranks = self._mlp_tp.ranks
        assert ranks is not None
        local = expert_num // ep_size
        return [rank * local <= expert_idx < (rank + 1) * local
                for rank in ranks]

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
