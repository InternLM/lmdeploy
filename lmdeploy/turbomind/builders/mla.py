# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import torch

from ..linear import Linear
from ._base import Builder, SplitSide

# ---------------------------------------------------------------------------
# MLA fold+pad pipeline (standalone functions)
# ---------------------------------------------------------------------------


def fold_kv_b(q_b: Linear, kv_b: Linear, wo: Linear, *,
              cfg) -> tuple[Linear, Linear]:
    """Fold kv_b into q_b and wo. Returns (q_b_folded, wo_folded).

    Splits kv_b into key-compressed (kc) and value-compressed (vc) parts. Folds kc into q_b via matmul (q_nope @ kc^T
    per head). Folds vc into wo via matmul (vc @ wo per head). All arithmetic in TM layout [in, out].
    """
    H = cfg.head_num
    P = cfg.qk_nope_dim
    S = cfg.qk_rope_dim
    R_q = cfg.q_lora_rank   # q_b input dim
    R = cfg.kv_lora_rank    # kv_b input dim, also fold expansion target
    V = wo.tensors['weight'].shape[0] // H  # v_head_dim (cfg value overridden)

    q_b_h = q_b.tensors['weight'].reshape(R_q, H, P + S)
    kc, vc = kv_b.tensors['weight'].reshape(R, H, P + V).split([P, V], dim=-1)
    q_nope, q_rope = q_b_h.split([P, S], dim=-1)

    # q_nope @ kc^T per head: [R_q, H, P] × [R, H, P] → [R_q, H, R]
    q_folded = torch.cat([
        torch.einsum('ihp,jhp->ihj', q_nope, kc),  # [R_q, H, R]
        q_rope,                                      # [R_q, H, S]
    ], dim=-1).reshape(R_q, H * (R + S))

    # vc @ wo per head
    o_folded = torch.einsum('rhv,hvn->hrn', vc,
                            wo.tensors['weight'].reshape(H, V, -1)
                            ).reshape(H * R, -1)

    return (Linear(tensors={'weight': q_folded.contiguous()},
                   weight_format=q_b.weight_format),
            Linear(tensors={'weight': o_folded.contiguous()},
                   weight_format=wo.weight_format))


def pad_wo_input(wo: Linear, *, cfg) -> Linear:
    """Pad wo input dim from head_num * cur_dim to head_num * size_per_head."""
    head_num = cfg.head_num
    size_per_head = cfg.head_dim
    w = wo.tensors['weight']
    cur_dim = w.shape[0] // head_num
    w = w.reshape(head_num, cur_dim, -1)
    w = torch.nn.functional.pad(w, (0, 0, size_per_head - cur_dim, 0))
    w = w.reshape(head_num * size_per_head, -1)
    return Linear(tensors={'weight': w.contiguous()},
                  weight_format=wo.weight_format)


# ---------------------------------------------------------------------------
# MLABuilder -- MLA projections, fold+pad, norms
# ---------------------------------------------------------------------------


class MLABuilder(Builder):
    """MLA (Multi-head Latent Attention) weight loading builder."""

    def add_projections(self, *, q_a_proj, q_b_proj, kv_a_proj, kv_b_proj,
                        wo):
        """Apply MLA fold+pad, then commit each projection."""
        q_b_proj, wo = fold_kv_b(q_b_proj, kv_b_proj, wo, cfg=self.config)
        wo = pad_wo_input(wo, cfg=self.config)

        for name, lin, side in [
            ('q_a_proj', q_a_proj, None),
            ('q_b_proj', q_b_proj, SplitSide.OUTPUT),
            ('kv_a_proj', kv_a_proj, None),
            ('wo', wo, SplitSide.INPUT),
        ]:
            self._add_linear(name, lin, split_side=side)
