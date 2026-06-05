# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import xgrammar as xgr

    from ..engine.guided_process import GuidedDecodingManager


class GuidedSpecHelper:
    """Guided-decoding support for speculative decoding.

    Wraps a :class:`GuidedDecodingManager` and provides spec-decoding-specific
    operations that cannot be handled by :class:`FusedLogitsProcessor` because
    speculative decoding needs:

    * Position-serial bitmasking across N+1 positions (not 1).
    * Forked matchers to preserve originals for target-side verification.
    * Rejection-sampling-driven token acceptance (not direct argmax).
    * Draft-vocab bitmask translation (Eagle3).

    Instead of passing ``guided_decoding_manager`` into ``FusedLogitsProcessor``,
    the spec-decoding path constructs a ``GuidedSpecHelper`` and calls its
    methods at the appropriate points.

    All public methods are no-ops when constructed with ``guided_manager=None``
    or when no guided processors are active, so callers never need to guard
    with ``if guided_helper:`` or ``if processors:``.
    """

    def __init__(self, guided_manager: GuidedDecodingManager | None = None):
        self._mgr = guided_manager

    @property
    def manager(self) -> GuidedDecodingManager | None:
        """Access the underlying :class:`GuidedDecodingManager`."""
        return self._mgr

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def cleanup_sessions(self, session_ids: list[int] | None):
        """Remove grammar processors for ended sessions."""
        if self._mgr is None or not session_ids:
            return
        for session_id in session_ids:
            self._mgr.remove_processor(session_id)

    def get_processors(self, session_ctx, response_formats) -> dict[int, xgr.GrammarMatcher]:
        """Get grammar processors for active guided sessions.

        Returns an empty dict when no manager is set or no sessions are
        guided, so callers can use ``if processors:`` uniformly.
        """
        if self._mgr is None or session_ctx is None:
            return {}
        return self._mgr.get_processors(session_ctx, response_formats)

    # ------------------------------------------------------------------
    # Draft side (called from proposer.get_outputs)
    # ------------------------------------------------------------------

    async def prepare_bitmask(self, logits: torch.Tensor,
                              processors: dict[int, xgr.GrammarMatcher] | None) -> torch.Tensor | None:
        """Allocate and fill a guided-decoding bitmask for draft logits.

        Returns the filled bitmask tensor (or ``None`` if no guided processors
        are active).  The caller is responsible for applying the bitmask —
        some proposers (e.g. Eagle3) may need to translate the bitmask to
        their draft vocabulary first.
        """
        if not processors or self._mgr is None:
            return None
        bitmask = self._mgr.allocate_batched_bitmap(logits.size(0))

        def _fill():
            for idx, proc in processors.items():
                self._mgr.fill_bitmap(proc, bitmask, idx)

        await asyncio.to_thread(_fill)
        return bitmask

    def apply_bitmask(self, logits: torch.Tensor, bitmask: torch.Tensor | None):
        """Apply a guided bitmask to logits.

        No-op when *bitmask* is ``None``.
        """
        if bitmask is None or self._mgr is None:
            return
        self._mgr.apply_batched_bitmap(logits, bitmask)

    async def accept_draft_tokens(self, draft_token_ids: torch.Tensor,
                                  processors: dict[int, xgr.GrammarMatcher] | None):
        """Accept draft tokens on the provided (forked) grammar matchers.

        In speculative decoding the matchers are typically forked from the
        originals (created in :meth:`SpecModelAgent._async_model_forward`),
        so this method accepts on whichever matchers are passed in.
        """
        if not processors or self._mgr is None:
            return
        cpu_ids = draft_token_ids[:, 0].cpu()

        def _accept():
            for idx, proc in processors.items():
                self._mgr.accept_token(proc, cpu_ids[idx].item())

        await asyncio.to_thread(_accept)

    # ------------------------------------------------------------------
    # Target side: position-serial bitmask with forked matchers
    # ------------------------------------------------------------------

    async def apply_serial_bitmask(
        self,
        scores_3d: torch.Tensor,
        processors: dict[int, xgr.GrammarMatcher],
        draft_token_ids: torch.LongTensor,
        num_spec_tokens: int,
    ):
        """Apply position-serial grammar mask to target logits.

        Forks the provided processors, applies bitmask at each speculative
        position, and advances the forks using the draft tokens.  The original
        processors are **not** modified.

        No-op when *processors* is empty.

        Args:
            scores_3d: ``[batch_size, num_expand, vocab_size]`` logits tensor
                (modified in-place).
            processors: Original grammar matchers indexed by batch position.
            draft_token_ids: ``[batch_size, num_spec_tokens]`` draft tokens
                from the proposer.  Forks are advanced using these (not
                argmax) because target logits are conditioned on the draft
                tokens.
            num_spec_tokens: Number of speculative tokens per step.
        """
        if not processors or self._mgr is None:
            return
        forked = {idx: proc.fork() for idx, proc in processors.items()}
        cpu_draft = draft_token_ids.cpu()
        batch_size = scores_3d.size(0)
        num_expand = scores_3d.size(1)
        bitmask = self._mgr.allocate_batched_bitmap(batch_size)

        for pos in range(num_expand):
            await asyncio.to_thread(self._fill_bitmask, forked, bitmask)
            pos_logits = scores_3d[:, pos, :]
            self._mgr.apply_batched_bitmap(pos_logits, bitmask)
            scores_3d[:, pos, :] = pos_logits

            # Advance fork using draft tokens for draft positions.
            if pos < num_spec_tokens:
                await asyncio.to_thread(self._accept_forked_at_pos, forked, cpu_draft, pos)

    # ------------------------------------------------------------------
    # Token acceptance (rejection-sampling-aware)
    # ------------------------------------------------------------------

    async def accept_rejection_sampled_tokens(
        self,
        processors: dict[int, xgr.GrammarMatcher],
        num_rejected: torch.Tensor,
        output_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        num_spec_tokens: int,
    ):
        """Accept rejection-sampled tokens on original grammar matchers.

        After rejection sampling, the original matchers must be advanced to
        reflect the accepted tokens.  For each sequence, ``num_spec_tokens -
        num_rejected`` draft tokens are accepted, followed by the bonus token.

        No-op when *processors* is empty.

        Args:
            processors: Original (non-forked) grammar matchers.
            num_rejected: Per-sequence rejection counts (GPU or CPU tensor).
            output_token_ids: Accepted output tokens ``[batch, num_spec]``
                (GPU or CPU tensor).
            next_token_ids: Bonus tokens ``[batch]`` (GPU or CPU tensor).
            num_spec_tokens: Number of speculative tokens per step.
        """
        if not processors or self._mgr is None:
            return
        cpu_num_rejected = num_rejected.cpu() if num_rejected.is_cuda else num_rejected
        cpu_output_token_ids = output_token_ids.cpu() if output_token_ids.is_cuda else output_token_ids
        cpu_next_token_ids = next_token_ids.cpu() if next_token_ids.is_cuda else next_token_ids

        def _accept():
            for idx, processor in processors.items():
                n_rejected = cpu_num_rejected[idx].item()
                n_valid_draft = num_spec_tokens - n_rejected
                for pos in range(n_valid_draft):
                    tid = cpu_output_token_ids[idx, pos].item()
                    if tid >= 0:
                        self._mgr.accept_token(processor, tid)
                self._mgr.accept_token(processor, cpu_next_token_ids[idx].item())

        await asyncio.to_thread(_accept)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fill_bitmask(self, processors: dict, bitmask: torch.Tensor):
        for idx, proc in processors.items():
            self._mgr.fill_bitmap(proc, bitmask, idx)

    def _accept_forked_at_pos(self, forked: dict, cpu_draft: torch.Tensor, pos: int):
        for idx, fork_proc in forked.items():
            self._mgr.accept_token(fork_proc, cpu_draft[idx, pos].item())
