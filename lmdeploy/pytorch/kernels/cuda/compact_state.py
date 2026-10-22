# Copyright (c) OpenMMLab. All rights reserved.
"""Device-only compact bank planning and consumed-prefix selection.

These operations move IDs, never state payloads. The host owner validates and pins every private bank snapshot and any
shared input before submission.
"""
import torch
import triton
import triton.language as tl


# Live batch changes on admission/drain; it only masks rows, not the layout.
@triton.jit(do_not_specialize=['B'])
def _select_ids(bank, inputs, old_outputs, consumed, next_inputs, next_outputs,
                runtime_ids, request_ids, histories, sources, checkpoint_ids, checkpoint_offsets,
                cu_seqlens, decode_ids,
                DECODE_CHECKPOINT: tl.constexpr, FINAL_CHECKPOINT: tl.constexpr, A: tl.constexpr,
                BIND: tl.constexpr, RECORD: tl.constexpr, OVERRIDE: tl.constexpr,
                B, V: tl.constexpr, OLD_W: tl.constexpr, W: tl.constexpr,
                ADVANCE: tl.constexpr, PARTIAL_BANK: tl.constexpr, ALLOW_INPUT_ALIAS: tl.constexpr,
                ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    current = tl.load(inputs + rows, rows < B, other=-1)
    live_request = rows < B
    if BIND or RECORD:
        request = tl.load(request_ids + rows, rows < B, other=-1)
        live_request = (rows < B) & (request > 0) & (request < A)
    if BIND:
        history = tl.load(histories + rows, rows < B, other=-1)
        retained = tl.load(runtime_ids + request, live_request, other=-1)
        current = tl.where(history == 0, 0, retained)
        if OVERRIDE:
            source = tl.load(sources + rows, rows < B, other=-1)
            current = tl.where(source == -1, current, source)
        current = tl.where(live_request & (history >= 0), current, -1)
    valid = (current >= 0) & live_request
    if ADVANCE:
        count = tl.load(consumed + rows, rows < B, other=-1)
        accepted = tl.load(old_outputs + rows * OLD_W + count - 1,
                           (rows < B) & (count > 0) & (count <= OLD_W), other=-1)
        if ALLOW_INPUT_ALIAS and not PARTIAL_BANK:
            previous_last = tl.load(old_outputs + rows * OLD_W + OLD_W - 1, rows < B, other=-1)
            # Once a private input was overwritten, zero advancement cannot
            # retain it. Fail closed instead of reporting the new value as old.
            valid = valid & ((count != 0) | (previous_last != current))
        current = tl.where(count == 0, current, accepted)
        valid = valid & (count >= 0) & (count <= OLD_W) & ((count == 0) | (current > 0))
    ids = tl.load(bank + rows[:, None] * V + cols[None, :],
                  (rows[:, None] < B) & (cols[None, :] < V), other=-1)
    if PARTIAL_BANK:
        size = tl.sum(((cols[None, :] < V) & (ids > 0)).to(tl.int32), 1)
        # Positive IDs form a prefix; padding represents no physical units.
        canonical = tl.where(cols[None, :] < size[:, None], ids > 0, ids == -1)
        valid = valid & (size > 0) & (tl.min((cols[None, :] >= V) | canonical, 1) != 0)
    else:
        size = tl.full((ROWS,), V, tl.int32)
        valid = valid & (tl.min((cols[None, :] >= V) | (ids > 0), 1) != 0)
    # A shared input may be outside the bank; start at slot0 in that case.
    position = tl.max(tl.where((cols[None, :] < V) & (ids == current[:, None]), cols[None, :], -1), 1)
    slots = (position[:, None] + 1 + cols[None, :]) % tl.maximum(size[:, None], 1)
    outputs = tl.gather(ids, slots, 1)
    output_valid = valid
    if PARTIAL_BANK:
        # Worker advancement can precede CPU decode admission. Keep the final
        # prefill input but mask unrunnable outputs until a complete rebind.
        required_input = (position >= 0).to(tl.int32) if not ALLOW_INPUT_ALIAS else 0
        output_valid = valid & (W <= size - required_input)
    if FINAL_CHECKPOINT or DECODE_CHECKPOINT:
        if FINAL_CHECKPOINT:
            checkpoint = tl.load(checkpoint_ids + rows, rows < B, other=-1)
            offset = tl.load(checkpoint_offsets + rows, rows < B, other=0)
            length = (tl.load(cu_seqlens + rows + 1, rows < B, other=0)
                      - tl.load(cu_seqlens + rows, rows < B, other=0))
            final_checkpoint = (checkpoint > 0) & (offset == length) & (length > 0)
        else:
            checkpoint = tl.load(decode_ids + rows, rows < B, other=-1)
            final_checkpoint = checkpoint > 0
        # The owner reserved this one-shot producer outside every writable
        # bank. Fail closed on accidental source/bank alias, not a silent copy.
        private_alias = tl.sum(((cols[None, :] < V) & (ids == checkpoint[:, None])).to(tl.int32), 1) > 0
        valid = valid & (~final_checkpoint | ((checkpoint != current) & ~private_alias))
        outputs = tl.where(final_checkpoint[:, None], checkpoint[:, None], outputs)
        output_valid = valid & (final_checkpoint | output_valid)
    if RECORD:
        tl.store(runtime_ids + request, tl.where(valid, current, -1), live_request)
    tl.store(next_inputs + rows, tl.where(valid, current, -1), rows < B)
    tl.store(next_outputs + rows[:, None] * W + cols[None, :],
             tl.where(output_valid[:, None], outputs, -1), (rows[:, None] < B) & (cols[None, :] < W))


def select_compact_state_ids(bank_ids: torch.Tensor, input_ids: torch.Tensor, width: int, *,
                             output_ids: torch.Tensor | None = None,
                             consumed: torch.Tensor | None = None,
                             runtime_ids: torch.Tensor | None = None,
                             request_ids: torch.Tensor | None = None,
                             history_lengths: torch.Tensor | None = None,
                             source_ids: torch.Tensor | None = None,
                             prefill=None, decode_ids: torch.Tensor | None = None,
                             allow_partial_bank: bool = False,
                             allow_input_alias: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Select a consumed output and plan distinct private next destinations.

    ``consumed`` counts states actually computed/retained by the previous
    forward, NOT accepted draft tokens alone.0 retains input;1 selects output0.
    Out-of-range counts/invalid source rows fail closed to -1 IDs. Bank IDs must
    be positive, row-private, unique and owner-held; this kernel is not an owner.
    Partial prefill banks may have a positive prefix and -1 padding. An
    insufficient bank masks future outputs without losing its selected input;
    decode must rebind a fully admitted bank before model execution.

    Explicit ``allow_input_alias`` permits V=W, including AR V=1. The owner
    must exclude shared inputs from its bank and guarantee positive retained
    advancement when a private input is reused. Zero advancement after such
    overwrite fails closed; genuine zero-advance producers need disjoint banks.
    For partial-prefill metadata only, count0 means an empty row which computed
    no output and did not overwrite its input; that input remains valid.
    """
    minimum_bank = 1 if allow_input_alias else 2
    if bank_ids.ndim != 2 or bank_ids.shape[1] < minimum_bank:
        raise ValueError(f'Expected compact bank table [B,V], V>={minimum_bank}.')
    batch, bank_width = bank_ids.shape
    max_width = bank_width if allow_input_alias else bank_width - 1
    if not isinstance(width, int) or not 1 <= width <= max_width:
        raise ValueError('Output width must satisfy 1<=W<V unless exclusive input reuse is authorized.')
    if input_ids.shape != (batch,):
        raise ValueError('Expected one compact input ID per bank row.')
    if (output_ids is None) != (consumed is None):
        raise ValueError('Previous outputs and consumed counts must be supplied together.')
    for ids in (bank_ids, input_ids, output_ids):
        if ids is not None and (ids.dtype != torch.int64 or ids.device != bank_ids.device or not ids.is_contiguous()):
            raise ValueError('Compact ID tensors must be contiguous int64 on the same device.')
    old_width = 0
    if output_ids is not None:
        if output_ids.ndim != 2 or output_ids.shape[0] != batch or not 1 <= output_ids.shape[1] <= max_width:
            raise ValueError('Previous output geometry must match the bank.')
        old_width = output_ids.shape[1]
        if (consumed.shape != (batch,) or consumed.device != bank_ids.device or not consumed.is_contiguous()
                or consumed.dtype not in (torch.int32, torch.int64)):
            raise ValueError('Consumed counts must be contiguous per-request device integers.')
    if (runtime_ids is None) != (request_ids is None):
        raise ValueError('Runtime IDs and auxiliary request IDs must be supplied together.')
    if runtime_ids is not None:
        if (runtime_ids.ndim != 1 or runtime_ids.dtype != torch.int64
                or runtime_ids.device != bank_ids.device or not runtime_ids.is_contiguous()):
            raise ValueError('Expected a contiguous int64 runtime ID record on the bank device.')
        for tensor in (request_ids, history_lengths, source_ids):
            if tensor is not None and (tensor.shape != (batch,) or tensor.dtype != torch.int64
                                       or tensor.device != bank_ids.device or not tensor.is_contiguous()):
                raise ValueError('Runtime binding requires contiguous per-request int64 metadata on the bank device.')
        if (consumed is None) == (history_lengths is None):
            raise ValueError('Runtime selection must either bind a history or record consumed outputs.')
    if history_lengths is not None and runtime_ids is None:
        raise ValueError('History binding requires a runtime ID record.')
    if source_ids is not None and history_lengths is None:
        raise ValueError('Explicit source IDs belong only to a one-forward history binding.')
    checkpoint_ids = None if prefill is None else prefill.checkpoint_ids
    checkpoint_offsets = None if prefill is None else prefill.checkpoint_offsets
    cu_seqlens = None if checkpoint_ids is None else prefill.cu_seqlens
    if checkpoint_ids is not None:
        if history_lengths is None or consumed is not None or width != 1:
            raise ValueError('Final checkpoint selection belongs only to a one-output prefill binding.')
        for tensor, shape in ((checkpoint_ids, (batch,)), (checkpoint_offsets, (batch,)),
                              (cu_seqlens, (batch + 1,))):
            if (tensor is None or tensor.shape != shape or tensor.dtype != torch.int64
                    or tensor.device != bank_ids.device or not tensor.is_contiguous()):
                raise ValueError('Final checkpoint selection requires matching canonical device metadata.')
    if decode_ids is not None:
        if width != 1 or consumed is not None or history_lengths is not None or prefill is not None:
            raise ValueError('Decode checkpoint selection requires a one-output AR bank rebind.')
        if (decode_ids.shape != (batch,) or decode_ids.dtype != torch.int64
                or decode_ids.device != bank_ids.device or not decode_ids.is_contiguous()):
            raise ValueError('Decode checkpoint IDs must match the bank rows and device.')
    if bank_ids.device.type != 'cuda':
        raise ValueError('Dynamic compact selection requires CUDA; host binding is a separate owner operation.')
    next_inputs = torch.empty_like(input_ids)
    next_outputs = bank_ids.new_empty((batch, width))
    if batch:
        _select_ids[(triton.cdiv(batch, 32),)](
            bank_ids, input_ids, output_ids, consumed, next_inputs, next_outputs,
            runtime_ids, request_ids, history_lengths, source_ids, checkpoint_ids, checkpoint_offsets,
            cu_seqlens, decode_ids,
            FINAL_CHECKPOINT=checkpoint_ids is not None, DECODE_CHECKPOINT=decode_ids is not None,
            A=0 if runtime_ids is None else runtime_ids.numel(), BIND=history_lengths is not None,
            RECORD=runtime_ids is not None and consumed is not None, OVERRIDE=source_ids is not None,
            B=batch, V=bank_width, OLD_W=old_width, W=width, ADVANCE=consumed is not None,
            PARTIAL_BANK=allow_partial_bank, ALLOW_INPUT_ALIAS=allow_input_alias,
            ROWS=32, COLS=triton.next_power_of_2(bank_width), num_warps=4)
    return next_inputs, next_outputs


@triton.jit(do_not_specialize=['B'])
def _stage_ids(bank, inputs, outputs, static_bank, static_inputs, static_outputs,
               B, PAD_B: tl.constexpr, V: tl.constexpr, W: tl.constexpr,
               ROWS: tl.constexpr, COLS: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, COLS)
    bank_ids = tl.load(bank + rows[:, None] * V + cols[None, :],
                       (rows[:, None] < B) & (cols[None, :] < V), other=-1)
    output_ids = tl.load(outputs + rows[:, None] * W + cols[None, :],
                         (rows[:, None] < B) & (cols[None, :] < W), other=-1)
    input_ids = tl.load(inputs + rows, rows < B, other=-1)
    tl.store(static_bank + rows[:, None] * V + cols[None, :], bank_ids,
             (rows[:, None] < PAD_B) & (cols[None, :] < V))
    tl.store(static_outputs + rows[:, None] * W + cols[None, :], output_ids,
             (rows[:, None] < PAD_B) & (cols[None, :] < W))
    tl.store(static_inputs + rows, input_ids, rows < PAD_B)


def stage_compact_state_ids(source, destination):
    """Copy a submitted ID snapshot into runner-owned buffers; pad with -1.

    No input mutation, allocation, payload movement or accepted-ID selection. The runner must order this fill after the
    prior replay on its forward stream.
    """
    batch, bank_width = source.bank_ids.shape
    padded_batch = destination.bank_ids.shape[0]
    width = source.output_ids.shape[1]
    if source.allow_input_alias != destination.allow_input_alias:
        raise ValueError('Compact graph staging cannot change input reuse authorization.')
    if (batch > padded_batch or destination.bank_ids.shape[1] != bank_width
            or destination.output_ids.shape[1] != width):
        raise ValueError('Compact graph staging requires matching bank/update widths and sufficient batch capacity.')
    if source.input_ids.device != destination.input_ids.device or source.input_ids.device.type != 'cuda':
        raise ValueError('Compact graph staging requires CUDA tables on the same device.')
    if padded_batch:
        _stage_ids[(triton.cdiv(padded_batch, 32),)](
            source.bank_ids, source.input_ids, source.output_ids,
            destination.bank_ids, destination.input_ids, destination.output_ids,
            B=batch, PAD_B=padded_batch, V=bank_width, W=width,
            ROWS=32, COLS=triton.next_power_of_2(bank_width), num_warps=4)
