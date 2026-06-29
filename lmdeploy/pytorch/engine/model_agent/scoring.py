# Copyright (c) OpenMMLab. All rights reserved.
import torch


@torch.inference_mode()
def compute_input_ce_loss(logits: torch.Tensor,
                          input_ids: torch.Tensor,
                          seq_length: torch.Tensor,
                          prev_last_logit: torch.Tensor | None = None):
    """Compute the per-sequence cross-entropy sum of the input prompt.

    For one sequence ``[t0, t1, ..., t_{n-1}]``, the logits at position ``i``
    predict ``t_{i+1}``, so the scored targets are ``t1..t_{n-1}`` (``n - 1`` of
    them); the final position predicts the first *generated* token (no
    ground-truth) and is excluded. The returned ``ce_loss`` is the summed,
    unnormalized negative log-likelihood; the caller divides by ``n - 1`` to
    get the mean cross-entropy loss, matching ``Pipeline.get_ppl`` semantics.

    Only this cheap scalar reduction is kept; the full ``[num_tokens, vocab]``
    logits tensor is never transferred off device.

    Chunked prefill (a single long prompt split across several forward steps to
    bound peak memory) needs ``prev_last_logit`` to stay exact: the per-chunk
    loop below scores positions ``0..L-2`` of each chunk and leaves the last
    position unscored, because its target is the *next* chunk's first token,
    which is not in this chunk. The next chunk recovers it via
    ``prev_last_logit``. Without it a prompt split into ``k`` chunks would
    under-count ``k - 1`` boundary tokens.

    Args:
        logits: ``[num_tokens, vocab]`` logits of this forward step -- all
            sequences of the batch concatenated along dim 0.
        input_ids: ``[num_tokens]`` (or ``[1, num_tokens]``) token ids of this
            step, in the same order as ``logits``.
        seq_length: per-sequence token counts; sums to ``num_tokens``. Splits
            ``logits``/``input_ids`` back into individual sequences.
        prev_last_logit: the previous chunk's last logit row ``[1, vocab]``,
            supplied only for non-first chunks of a chunked prefill. When given,
            this chunk's first token is scored against it (the cross-chunk
            boundary). Chunked prefill always runs **one** sequence at a time
            (``inputs_maker`` only chunks when ``len(running) == 1``), so this
            boundary belongs to the sole sequence (index 0).

    Returns:
        torch.Tensor: ``ce_loss`` of shape ``[batch_size]`` (summed NLL per
            sequence).
    """
    input_ids = input_ids.flatten()
    sections = seq_length.tolist() if isinstance(seq_length, torch.Tensor) else list(seq_length)
    batch_size = len(sections)
    ce_loss = logits.new_zeros(batch_size, dtype=torch.float32)
    logits_per_seq = logits.split(sections)
    ids_per_seq = input_ids.split(sections)
    for i, (_logits, _ids) in enumerate(zip(logits_per_seq, ids_per_seq)):
        if _logits.size(0) < 2:
            # decode step or single token: no in-prompt target to score
            continue
        target = _ids[1:].long()
        # fused cross-entropy (sum of NLL); avoids materializing log_softmax,
        # matching Pipeline._get_ppl numerically
        ce_loss[i] = torch.nn.functional.cross_entropy(_logits[:-1].float(), target, reduction='sum')
    if prev_last_logit is not None:
        # Only chunked prefill passes prev_last_logit, and the engine chunks a
        # single sequence at a time -- hence exactly one sequence here, index 0.
        # Score its first token against the previous chunk's last logit (the
        # cross-chunk boundary the per-chunk loop above leaves out).
        assert batch_size == 1, 'prev_last_logit is only valid for single-sequence chunked prefill'
        first_tok = input_ids[:1].long()
        ce_loss[0] += torch.nn.functional.cross_entropy(prev_last_logit.float(), first_tok, reduction='sum')
    return ce_loss
