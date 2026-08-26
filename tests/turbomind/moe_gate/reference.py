from __future__ import annotations

import torch


@torch.no_grad()
def moe_gate_v2_reference(
    logits: torch.Tensor,
    experts_per_token: int,
    *,
    softmax: bool = True,
    norm_topk: bool = False,
    routed_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for invokeMoeGate_V2 mode matrix.

    Returns f2n, f2E, en2f, offsets, scales.
    Layouts:
      f2n/f2E: [top_k * tokens]
      en2f/scales: [top_k, tokens]
      offsets: [experts + 1]
    """
    if logits.ndim != 2:
        raise ValueError(f'logits must be [tokens, experts], got {tuple(logits.shape)}')
    if not softmax and norm_topk:
        raise ValueError('unsupported: softmax=False with norm_topk=True')

    tokens, experts = logits.shape
    device = logits.device
    logits_f = logits.float()

    # Per-token: stable top-k by logit, then natural expert-id order.
    # torch.topk is not stable; implement via argsort.
    order = torch.argsort(logits_f, dim=-1, descending=True, stable=True)
    top_idx = order[:, :experts_per_token].sort(dim=-1).values  # [tokens, top_k], ascending eid

    if softmax:
        if norm_topk:
            top_logits = logits_f.gather(1, top_idx)
            selected = torch.softmax(top_logits, dim=-1) * routed_scale
        else:
            probs = torch.softmax(logits_f, dim=-1)
            selected = probs.gather(1, top_idx) * routed_scale
    else:
        selected = logits_f.gather(1, top_idx) * routed_scale

    scales = selected.transpose(0, 1).contiguous()  # [top_k, tokens]

    # eids in [top_k, tokens] matching scale slot order (natural eid order).
    eids = top_idx.transpose(0, 1).contiguous().to(torch.int32)

    # Flattened slots: flat = slot * tokens + token, slot in [0, top_k).
    flat_eids = eids.reshape(-1)
    flat_tokens = torch.arange(tokens, device=device, dtype=torch.int32).repeat(experts_per_token)

    # Sort by (expert, token): expert ascending, then token ascending.
    sort_key = flat_eids.to(torch.int64) * tokens + flat_tokens.to(torch.int64)
    perm = torch.argsort(sort_key, stable=True)

    f2n = flat_tokens[perm].contiguous()
    f2E = flat_eids[perm].contiguous()

    en2f = torch.empty(experts_per_token * tokens, device=device, dtype=torch.int32)
    en2f[perm] = torch.arange(experts_per_token * tokens, device=device, dtype=torch.int32)
    en2f = en2f.view(experts_per_token, tokens).contiguous()

    counts = torch.bincount(flat_eids.to(torch.int64), minlength=experts)
    offsets = torch.zeros(experts + 1, device=device, dtype=torch.int32)
    offsets[1:] = counts.to(torch.int32).cumsum(0)

    return f2n, f2E, en2f, offsets, scales
