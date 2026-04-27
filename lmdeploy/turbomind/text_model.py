# Copyright (c) OpenMMLab. All rights reserved.
"""TextModel — per-architecture model owning HF parsing and C++ configs."""
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

from lmdeploy.utils import get_logger

from .builders import NormBuilder, make_norm_config
from .models.utils import parse_rope_param, rope_type_to_int

if TYPE_CHECKING:
    from lmdeploy.messages import TurbomindEngineConfig

logger = get_logger('lmdeploy')


class TextModel(ABC):
    """Text model: HF config -> C++ configs + weight commits.

    Subclass contract:
      - __init__ takes (hf_cfg, engine_cfg), calls super().__init__, then
        builds per-module C++ config templates as self._attn_cfg /
        self._ffn_cfg / self._moe_cfg / self._dn_cfg.
      - Factory method NAMES (attn/ffn/moe/linear_attn/mla/norm/...)
        are a convention for readability, NOT a protocol. Signatures
        may differ across subclasses. The base class provides no
        factory stubs; every subclass implements its own model()
        that calls root.add_token_embeds / root.add_lm_head on a
        TextModelBuilder for the root-level commits.
    """

    # Class-level: checkpoint loader hints (HF key renaming + layer regex).
    _layer_pattern: str = ''
    _loader_mappings: list = []


    # ------------------------------------------------------------------
    # Construction / parsing
    # ------------------------------------------------------------------

    def __init__(self, hf_cfg: dict, engine_cfg: TurbomindEngineConfig,
                 *, resolver):
        """Parse HF config into orchestration scalars.

        ``resolver`` is a ``WeightFormatResolver`` built by the converter.
        It carries the model compute dtype and the ordered list of
        candidate weight formats; ``self._linear()`` delegates to
        ``resolver.resolve()`` at weight-loading time.

        Subclasses override `_parse_base` (or extend in their own __init__)
        then construct C++ config templates and per-layer lists.
        """
        self.hf_cfg = hf_cfg
        self.engine_cfg = engine_cfg
        self._resolver = resolver
        self._dtype = self._cpp_dtype()
        self._parse_base(hf_cfg)

    def _parse_base(self, cfg: dict):
        """Fill canonical orchestration scalars from standard HF keys.

        Populated:
          _num_layer, _vocab_size, _norm_eps, _head_num, _kv_head_num,
          _head_dim, _hidden_units, _rope,
          _max_position_embeddings, _tie_embeddings,
          _model_name, _tune_layer_num, _embedding_size.

        Subclass responsibilities (not set here):
          _softmax_scale (subclass default 0, MLA+YaRN overrides)
        """
        self._num_layer = cfg['num_hidden_layers']
        self._vocab_size = cfg['vocab_size']
        self._norm_eps = cfg['rms_norm_eps']
        self._tie_embeddings = cfg.get('tie_word_embeddings', False)
        self._model_name = cfg.get('model_type', '')
        self._tune_layer_num = 1
        self._embedding_size = self._vocab_size

        attn_head_num = cfg['num_attention_heads']
        hidden = cfg['hidden_size']
        head_dim = cfg.get('head_dim') or (hidden // attn_head_num)
        kv_head_num = cfg.get('num_key_value_heads', attn_head_num)
        self._hidden_units = hidden
        self._head_dim = head_dim
        self._head_num = attn_head_num
        self._kv_head_num = kv_head_num

        self._rope, self._max_position_embeddings = parse_rope_param(
            cfg, head_dim)

        # Apply the deprecated --rope-scaling-factor override here so every
        # downstream consumer (subclass __init__, _apply_rope) sees the
        # patched rope before building C++ templates.
        if self.engine_cfg.rope_scaling_factor:
            self._rope.type = 'dynamic'
            self._rope.factor = self.engine_cfg.rope_scaling_factor
            self._rope.max_position_embeddings = self._max_position_embeddings
            logger.warning(
                '`--rope-scaling-factor` will be removed in a future release. '
                'Please instead use `--hf-overrides`.')

        # Default subclass can override (e.g. MLA+YaRN)
        self._softmax_scale = 0.0

    # ------------------------------------------------------------------
    # Runtime binding (called by ModelLoader after model_comm exists)
    # ------------------------------------------------------------------

    def bind_runtime(self, *, contexts, root_handles,
                     attn_ranks, mlp_ranks, model_tp_ranks):
        self._contexts = contexts
        self._root_handles = root_handles
        self._attn_ranks = attn_ranks
        self._mlp_ranks = mlp_ranks
        self._model_tp_ranks = model_tp_ranks

    def set_params(self, params: dict):
        self.params = params

    # ------------------------------------------------------------------
    # Checkpoint access helpers
    # ------------------------------------------------------------------

    def _get(self, key: str) -> torch.Tensor | None:
        return self.params.get(key)

    def _linear(self, pfx: str, *, optional: bool = False):
        return self._resolver.resolve(self.params, pfx, optional=optional)

    def _cpp_dtype(self):
        return self._resolver.data_type

    def _apply_rope(self, rope_cfg):
        """Copy self._rope fields into a C++ rope config object."""
        rope_cfg.type = rope_type_to_int(self._rope.type)
        rope_cfg.base = self._rope.base
        rope_cfg.dim  = self._rope.dim
        rope_cfg.factor = self._rope.factor
        rope_cfg.max_position_embeddings = self._max_position_embeddings
        if self._rope.type == 'yarn':
            rope_cfg.yarn_attention_factor = self._rope.attention_factor
            rope_cfg.yarn_beta_fast = self._rope.beta_fast
            rope_cfg.yarn_beta_slow = self._rope.beta_slow
        elif self._rope.type == 'llama3':
            rope_cfg.llama3_low_freq_factor = self._rope.low_freq_factor
            rope_cfg.llama3_high_freq_factor = self._rope.high_freq_factor
            rope_cfg.llama3_original_max_position_embeddings = self._rope.original_max_position_embeddings
        elif self._rope.type == 'mrope':
            rope_cfg.mrope_section = self._rope.mrope_section

    # ------------------------------------------------------------------
    # Norm factories (shared across all models)
    # ------------------------------------------------------------------

    def norm(self, weight, *, dim=None, data_type=None):
        """Build a NormBuilder for *weight* under this model's contexts.

        ``dim`` defaults to ``weight.shape[-1]``. ``data_type`` defaults to
        the model's compute dtype.
        """
        cfg = make_norm_config(
            dim=dim if dim is not None else weight.shape[-1],
            data_type=data_type if data_type is not None else self._dtype,
            norm_eps=self._norm_eps,
        )
        m = NormBuilder(cfg, self._contexts)
        m.set_weight(weight)
        return m.build()

