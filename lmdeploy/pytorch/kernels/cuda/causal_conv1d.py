# Copyright (c) OpenMMLab. All rights reserved.
import tilelang
import tilelang.language as T
import torch

# The kernels below is modified from: https://github.com/Dao-AILab/causal-conv1d


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}, )
def causal_conv1d_fwd(hidden_size, width, has_bias, activation, dtype, stride_x, has_init_states, num_warps,
                      ChunkSizeL=64, indexed_state=False, state_stride=None, index_dtype=torch.int32,
                      weight_dtype=None, bias_dtype=None):
    """TileLang kernel for causal convolution forward pass.

    Each thread processes one output position for all channels sequentially. When has_init_states=True, the kernel reads
    per-sequence initial states from Init_states[seq_id, channel, :] for positions before a sequence's start boundary,
    instead of contributing zero.
    """
    num_threads = num_warps * 32
    # GLM keeps convolution parameters in FP32 while activations/history are
    # BF16. Load the original parameters into the existing FP32 accumulators;
    # narrowing them to the activation dtype changes the model's computation.
    weight_dtype = dtype if weight_dtype is None else weight_dtype
    bias_dtype = dtype if bias_dtype is None else bias_dtype
    num_bits = T.DataType(dtype).bits
    num_bytes = num_bits // 8
    # elems_per_row <= num_threads
    elems_per_row = 128 // num_bytes
    ChunkSizeC = elems_per_row
    silu_activation = activation in ['silu', 'swish']

    l_per_thread = min(ChunkSizeC * ChunkSizeL // num_threads, ChunkSizeL)
    assert num_threads * l_per_thread == ChunkSizeC * ChunkSizeL
    thrs_per_row = ChunkSizeL // l_per_thread
    assert thrs_per_row * l_per_thread == ChunkSizeL
    sum_seqlen = T.dynamic('sum_seqlen')
    n_seqs = T.dynamic('n_seqs')
    n_units = T.dynamic('n_units')
    if state_stride is None:
        state_stride = (hidden_size * (width - 1), width - 1, 1)

    @T.prim_func
    def causal_conv1d_fwd_main(
        X: T.StridedTensor([hidden_size, sum_seqlen], dtype=dtype, strides=(1, stride_x)),
        W: T.Tensor([hidden_size, width], dtype=weight_dtype),
        seq_idx: T.Tensor([sum_seqlen], dtype=T.int32),
        Bias: T.Tensor([hidden_size], dtype=bias_dtype) = None,
        Init_states: T.StridedTensor([n_units, hidden_size, width - 1], dtype=dtype, strides=state_stride) = None,
        Out: T.StridedTensor([hidden_size, sum_seqlen], dtype=dtype, strides=(1, hidden_size)) = None,
        Final_States: T.Tensor([hidden_size, width - 1], dtype=dtype) = None,
        Input_ids: T.Tensor([n_seqs], dtype=index_dtype) = None,
    ):
        # Process sum_seqlen output positions across all threads and blocks
        # every cta process (ChunkSizeC, ChunkSizeL) output tile
        with T.Kernel(T.ceildiv(hidden_size, ChunkSizeC), T.ceildiv(sum_seqlen, ChunkSizeL),
                      threads=num_threads) as (bc, bl):

            x_smem = T.alloc_shared((ChunkSizeL + width - 1, ChunkSizeC), dtype)

            # load x(copy can not be used on strided tensor)
            for lidx, cidx in T.Parallel(ChunkSizeL, ChunkSizeC):
                glidx = bl * ChunkSizeL + lidx
                gcidx = bc * ChunkSizeC + cidx
                x_smem[lidx + width - 1, cidx] = T.if_then_else(glidx >= 0 and glidx < sum_seqlen, X[gcidx, glidx],
                                                                T.cast(0.0, dtype))
            for lidx, cidx in T.Parallel(width, ChunkSizeC):
                glidx = bl * ChunkSizeL + lidx - width + 1
                gcidx = bc * ChunkSizeC + cidx
                x_smem[lidx, cidx] = T.if_then_else(glidx >= 0 and glidx < sum_seqlen, X[gcidx, glidx],
                                                    T.cast(0.0, dtype))

            x_local = T.alloc_local((width - 1 + l_per_thread, ), T.float32)
            seq_idx_local = T.alloc_local((width - 1 + l_per_thread, ), seq_idx.dtype)
            w_local = T.alloc_local((width, ), T.float32)
            if has_bias:
                bias_var = T.alloc_var(T.float32)
            else:
                bias_var = 0.0
            T.clear(w_local)

            tid = T.get_thread_binding(0)
            row_idx = tid // thrs_per_row
            col_idx = tid % thrs_per_row
            c_idx = bc * ChunkSizeC + row_idx

            # load w/b
            if c_idx < hidden_size:
                for widx in T.unroll(width):
                    w_local[widx] = W[c_idx, widx]
                if has_bias:
                    bias_var = Bias[c_idx]

            # load x
            for i in T.unroll(l_per_thread + width - 1):
                x_local[i] = x_smem[col_idx * l_per_thread + i, row_idx]

            # load seq_idx
            for i in T.unroll(l_per_thread + width - 1):
                gi = bl * ChunkSizeL + col_idx * l_per_thread + i - (width - 1)
                seq_idx_local[i] = T.if_then_else(gi >= 0 and gi < sum_seqlen, seq_idx[gi], -1)

            out_vals = T.alloc_local((l_per_thread, ), T.float32)
            for i in T.unroll(l_per_thread):
                out_vals[i] = bias_var
                seq_idx_cur = seq_idx_local[i + width - 1]
                if seq_idx_cur < 0:
                    out_vals[i] = 0.0
                    continue

                if has_init_states:
                    # Count how many consecutive positions before the output
                    # belong to the same sequence (k_val). Positions outside
                    # that range need init state instead of x data.
                    k_val = T.alloc_var(T.int32)
                    k_val = 0
                    for j in T.unroll(width - 1):
                        k_val = T.if_then_else(
                            (seq_idx_local[i + width - 2 - j] == seq_idx_cur) and (k_val == j), j + 1, k_val)

                    for w in T.unroll(width):
                        if seq_idx_local[i + w] == seq_idx_cur:
                            out_vals[i] += w_local[w] * x_local[i + w]
                        else:
                            # w goes from 0..width-1, output is at i+width-1.
                            # The init state column: (width-1) - (width-1 - w) - 1 + k_val = w - 1 + k_val
                            # But more directly: the distance from seq start
                            # for this position is k_val + w (counting from
                            # the leftmost halo). init_col maps to the state.
                            init_col = k_val + w
                            if init_col < width - 1:
                                input_id = T.alloc_var(index_dtype)
                                input_id = seq_idx_cur
                                if indexed_state:
                                    input_id = Input_ids[seq_idx_cur]
                                if not indexed_state or (input_id >= 0 and input_id < n_units):
                                    out_vals[i] += w_local[w] * T.cast(
                                        Init_states[input_id, c_idx, init_col], T.float32)
                else:
                    for w in T.unroll(width):
                        out_vals[i] += T.if_then_else(seq_idx_local[i + w] == seq_idx_cur,
                                                      w_local[w] * x_local[i + w], 0.0)

                if silu_activation:
                    out_vals[i] = T.sigmoid(out_vals[i]) * out_vals[i]

            for i in T.unroll(l_per_thread):
                x_smem[col_idx * l_per_thread + i, row_idx] = out_vals[i]

            for lidx, cidx in T.Parallel(ChunkSizeL, ChunkSizeC):
                glidx = bl * ChunkSizeL + lidx
                gcidx = bc * ChunkSizeC + cidx
                Out[gcidx, glidx] = T.if_then_else(glidx >= 0 and glidx < sum_seqlen, x_smem[lidx, cidx],
                                                   T.cast(0.0, dtype))

    return causal_conv1d_fwd_main


def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """Causal 1D convolution function using TileLang kernel.

    Args:
        x: Input tensor of shape [batch_size, hidden_size, sequence_length]
           Note: batch_size must be 1
        weight: Convolution weights of shape [hidden_size, kernel_size]
        bias: Optional bias of shape [hidden_size]
        seq_idx: Sequence indices of shape [sequence_length] to handle multiple sequences
        initial_states: Per-sequence initial states [n_seqs, hidden_size, kernel_size-1]
        return_final_states: Whether to return final states
        final_states_out: Output tensor for final states
        activation: Activation function name ('silu', 'gelu', 'relu', or None)

    Returns:
        output: Convolution result of shape [batch_size, hidden_size, sequence_length]
        (and final_states if return_final_states=True)
    """
    assert x.dim() == 3, 'x should be in shape of [batch_size, hidden_size, sum_seqlen]'
    assert x.size(0) == 1, 'batch_size should be 1 for continuous batching'
    assert x.stride(1) == 1, 'x should be in channel last format'
    assert weight.dim() == 2, 'weight should be in shape of [hidden_size, kernel_size]'
    assert seq_idx is not None, 'seq_idx is required for causal_conv1d_fn'
    assert activation in ['silu', 'swish', None]
    assert not return_final_states, 'return_final_states=True is not supported in this version'

    _, hidden_size, _ = x.shape
    kernel_size = weight.shape[1]
    dtype = x.dtype
    has_init_states = initial_states is not None

    # Reshape to 2D format for kernel: [hidden_size, sum_seqlen]
    x_2d = x.squeeze(0)  # [hidden_size, sum_seqlen]
    seq_idx_1d = seq_idx.squeeze(0) if seq_idx.dim() > 1 else seq_idx  # [sum_seqlen]

    # Initialize output tensor, hidden_size first for better memory access pattern
    out = x_2d.new_empty(x_2d.size(1), hidden_size)
    out = out.T

    # Create and call the TileLang kernel
    num_warps = 4  # Tunable parameter
    kernel = causal_conv1d_fwd(hidden_size, kernel_size, bias is not None, activation, dtype, x.stride(2),
                               has_init_states, num_warps)

    kernel(
        x_2d,
        weight,
        seq_idx_1d,
        bias,
        initial_states,
        out,
        None,
        None,
    )

    # Reshape back to original format: [1, hidden_size, sum_seqlen]
    out = out.unsqueeze(0)

    return out


@tilelang.jit
def _export_conv_history(channels, width, dtype, stride_x, state_stride, index_dtype, has_checkpoint):
    """Export both boundaries after all convolution readers have finished."""
    length, count, units = T.dynamic('length'), T.dynamic('count'), T.dynamic('units')

    @T.prim_func
    def kernel(
        X: T.StridedTensor((channels, length), dtype=dtype, strides=(1, stride_x)),
        State: T.StridedTensor((units, channels, width - 1), dtype=dtype, strides=state_stride),
        Inputs: T.Tensor((count,), index_dtype),
        Finals: T.Tensor((count,), index_dtype),
        Lengths: T.Tensor((count + 1,), index_dtype),
        Checkpoints: T.Tensor((count,), index_dtype) = None,
        Offsets: T.Tensor((count,), index_dtype) = None,
    ):
        with T.Kernel(count, T.ceildiv(channels, 128), threads=128) as (row, block):
            channel = block * 128 + T.get_thread_binding(0)
            source, target = Inputs[row], Finals[row]
            start, end = Lengths[row], Lengths[row + 1]
            if channel < channels and source >= 0 and source < units and end > start:
                final_history = T.alloc_local((width - 1,), dtype)
                checkpoint_history = T.alloc_local((width - 1,), dtype)
                checkpoint = T.alloc_var(index_dtype)
                boundary = T.alloc_var(index_dtype)
                checkpoint = -1
                boundary = start
                if has_checkpoint:
                    checkpoint = Checkpoints[row]
                    boundary = start + Offsets[row]
                valid_final = target > 0 and target < units
                valid_checkpoint = (checkpoint > 0 and checkpoint < units and checkpoint != source
                                    and checkpoint != target and boundary > start and boundary <= end)
                # Read both histories before an aliased final overwrites the source.
                if valid_final:
                    for j in T.unroll(width - 1):
                        position = end - (width - 1) + j
                        final_history[j] = T.if_then_else(position >= start, X[channel, position],
                            State[source, channel, width - 1 + position - start])
                if valid_checkpoint:
                    for j in T.unroll(width - 1):
                        position = boundary - (width - 1) + j
                        checkpoint_history[j] = T.if_then_else(position >= start, X[channel, position],
                            State[source, channel, width - 1 + position - start])
                if valid_final:
                    for j in T.unroll(width - 1):
                        State[target, channel, j] = final_history[j]
                if valid_checkpoint:
                    for j in T.unroll(width - 1):
                        State[checkpoint, channel, j] = checkpoint_history[j]

    return kernel


def indexed_causal_conv1d_fwd(x, weight, pool, input_ids, final_ids, *, seq_idx, cu_seqlens, bias=None,
                             activation=None, checkpoint_ids=None, checkpoint_offsets=None):
    """Packed prefill with unit inputs and direct ordered K-1 outputs.

    The owner must validate private destinations against every input/holder and supply consistent token-to-row and
    cumulative-length metadata. A checkpoint offset is relative to this forward, not the request's absolute token count.
    All metadata is prebuilt; this entrypoint never reads device IDs on the CPU. Final/checkpoint histories are exported
    after all convolution readers finish, including owner-authorized input reuse. No persistent-state staging is used.
    """
    if x.ndim != 3 or x.shape[0] != 1 or x.stride(1) != 1:
        raise ValueError('Expected channel-last packed input [1,C,T].')
    _, channels, length = x.shape
    width = weight.shape[1]
    if (weight.shape != (channels, width) or not weight.is_contiguous()
            or weight.dtype not in (x.dtype, torch.float32)):
        raise ValueError('Expected contiguous convolution weight [C,K] in activation dtype or FP32.')
    if pool.ndim != 3 or pool.shape[1:] != (channels, width - 1) or pool.dtype != x.dtype:
        raise ValueError('Expected ordered K-1 state pool [units,C,K-1] in compute dtype.')
    if activation not in ('silu', 'swish', None):
        raise ValueError('Unsupported activation.')
    if (checkpoint_ids is None) != (checkpoint_offsets is None):
        raise ValueError('Checkpoint IDs and offsets must be supplied together.')
    count = input_ids.numel()
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected int32 or int64 metadata.')
    for tensor in (input_ids, final_ids, checkpoint_ids, checkpoint_offsets):
        if tensor is not None and (tensor.shape != (count,) or tensor.dtype != input_ids.dtype
                                   or not tensor.is_contiguous()):
            raise ValueError('Expected contiguous per-request IDs/offsets of the same dtype.')
    if cu_seqlens.shape != (count + 1,) or cu_seqlens.dtype != input_ids.dtype or not cu_seqlens.is_contiguous():
        raise ValueError('Expected cumulative lengths in ID dtype.')
    if seq_idx.shape != (length,) or seq_idx.dtype != torch.int32 or not seq_idx.is_contiguous():
        raise ValueError('Expected contiguous int32 token-to-row indices.')
    if bias is not None and (bias.shape != (channels,) or bias.dtype not in (x.dtype, torch.float32)
                             or not bias.is_contiguous()):
        raise ValueError('Expected contiguous bias in activation dtype or FP32.')
    for tensor in (weight, pool, input_ids, final_ids, seq_idx, cu_seqlens, bias, checkpoint_ids, checkpoint_offsets):
        if tensor is not None and tensor.device != x.device:
            raise ValueError('All tensors must be on the compute device.')
    out = x.new_empty(length, channels).T
    if length:
        # Keep indexed metadata/register pressure bounded; larger tiles regress
        # short and ragged inputs even with history export outside the math loop.
        # A short packed input needs only one eight-token tile. This is a host
        # shape bound (also valid for ragged rows), not a device length readback.
        kernel = causal_conv1d_fwd(channels, width, bias is not None, activation, x.dtype, x.stride(2), True, 4,
                                  ChunkSizeL=8 if length <= 8 else 16, indexed_state=True, state_stride=pool.stride(),
                                  index_dtype=input_ids.dtype,
                                  weight_dtype=weight.dtype, bias_dtype=bias.dtype if bias is not None else x.dtype)
        kernel(x.squeeze(0), weight, seq_idx, bias, pool, out, None, input_ids)
        export = _export_conv_history(channels, width, x.dtype, x.stride(2), pool.stride(),
                                      input_ids.dtype, checkpoint_ids is not None)
        export(x.squeeze(0), pool, input_ids, final_ids, cu_seqlens, checkpoint_ids, checkpoint_offsets)
    return out.unsqueeze(0)


@tilelang.jit(pass_configs={
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}, )
def causal_conv1d_update_fwd(hidden_size: int, seqlen: int, state_len: int, width: int, has_bias: bool,
                             activation: str | None, dtype, conv_stride: tuple[int, int, int], is_circular_buffer: bool,
                             has_state_indices: bool, num_warps: int, weight_dtype=None, bias_dtype=None,
                             x_stride=None, has_output_state_indices: bool = False, index_dtype=torch.int32):
    """TileLang kernel for causal convolution forward pass.

    Each thread processes one output position for all channels sequentially.
    """
    num_threads = num_warps * 32
    silu_activation = activation in ['silu', 'swish']
    weight_dtype = dtype if weight_dtype is None else weight_dtype
    bias_dtype = dtype if bias_dtype is None else bias_dtype

    advance_len = seqlen
    batch = T.dynamic('batch')
    conv_batch = T.dynamic('conv_batch')
    conv_batch_stride = T.dynamic('conv_batch_stride')
    if x_stride is None:
        x_stride = (hidden_size * seqlen, seqlen, 1)
    update_idx_base = -(width - 1)

    @T.prim_func
    def causal_conv1d_update_main(
        X: T.StridedTensor((batch, hidden_size, seqlen), dtype=dtype, strides=x_stride),
        Conv_State: T.StridedTensor((conv_batch, hidden_size, state_len),
                                    dtype=dtype,
                                    strides=(conv_batch_stride, conv_stride[1], conv_stride[2])),
        W: T.Tensor((hidden_size, width), dtype=weight_dtype),
        Bias: T.Tensor((hidden_size, ), dtype=bias_dtype) = None,
        Out: T.Tensor((batch, hidden_size, seqlen), dtype=dtype) = None,
        Cache_seqlens: T.Tensor((batch, ), dtype=T.int32) = None,
        Conv_state_indices: T.Tensor((batch, ), dtype=index_dtype) = None,
        Output_state_indices: T.Tensor((batch, seqlen), dtype=T.int64) = None,
    ):
        with T.Kernel(batch, T.ceildiv(hidden_size, num_threads), threads=num_threads) as (bi, bc):
            tidx = T.get_thread_binding(0)
            batch_id = bi
            channel_id = bc * num_threads + tidx

            # load conv state index
            conv_state_batch_coord = T.if_then_else(has_state_indices, T.cast(Conv_state_indices[batch_id], T.int64),
                                                    T.cast(batch_id, T.int64))

            # get update_ids
            if is_circular_buffer:
                cache_seqlen = T.if_then_else(is_circular_buffer, Cache_seqlens[batch_id] % state_len, 0)
                # alloc_var outside branch would leads to error
                # when is_circular_buffer=False and has_state_indices=False
                # out_val would share same var of update_idx
                # seems like a bug of tilelang
                update_idx = T.alloc_var(T.int32)
                update_idx = cache_seqlen + update_idx_base
                update_idx = T.if_then_else(update_idx >= 0, update_idx, update_idx + state_len)
            else:
                update_idx = update_idx_base

            # skip padding tokens
            # tilelang does not support return in branch,
            # so I have to create this ugly branch to skip the computation for padding tokens
            if conv_state_batch_coord < 0 or (has_output_state_indices and conv_state_batch_coord >= conv_batch):
                for i in T.unroll(seqlen, unroll_factor=2):
                    Out[batch_id, channel_id, i] = 0.0
            else:
                # load bias and weight
                bias_val = T.if_then_else(has_bias, T.cast(Bias[channel_id], T.float32), T.cast(0.0, T.float32))
                weight_vals = T.alloc_local((width, ), T.float32)
                for i in T.unroll(width):
                    weight_vals[i] = W[channel_id, i]

                # fill conv states and read x_vals
                x_vals = T.alloc_local((width, ), T.float32)
                if has_output_state_indices:
                    for i in T.unroll(width - 1):
                        x_vals[i] = Conv_State[conv_state_batch_coord, channel_id, i]
                elif not is_circular_buffer:
                    for i in T.unroll(state_len - advance_len - (width - 1), unroll_factor=2):
                        Conv_State[conv_state_batch_coord, channel_id, i] = Conv_State[conv_state_batch_coord,
                                                                                       channel_id, i + advance_len]
                    for i in T.unroll(width - 1):
                        state_val = Conv_State[conv_state_batch_coord, channel_id, state_len - (width - 1) + i]
                        if i < advance_len + (width - 1) and state_len - advance_len - (width - 1) + i >= 0:
                            Conv_State[conv_state_batch_coord, channel_id,
                                       state_len - advance_len - (width - 1) + i] = state_val
                        x_vals[i] = state_val
                else:
                    for i in T.unroll(width - 1):
                        state_val = Conv_State[conv_state_batch_coord, channel_id, update_idx]
                        update_idx = (update_idx + 1) % state_len
                        x_vals[i] = state_val

                # compute output
                for i in T.unroll(seqlen, unroll_factor=2):
                    x_val = X[batch_id, channel_id, i]
                    if has_output_state_indices:
                        # A channel's input history is already thread-local.
                        # Only the last candidate may reuse an owner-authorized
                        # exclusive input; earlier candidates remain distinct.
                        dst = T.cast(Output_state_indices[batch_id, i], T.int64)
                        if (dst > 0 and dst < conv_batch and (dst != conv_state_batch_coord
                                or (i == seqlen - 1))):
                            for j in T.unroll(width - 2):
                                Conv_State[dst, channel_id, j] = x_vals[j + 1]
                            Conv_State[dst, channel_id, width - 2] = x_val
                    elif not is_circular_buffer:
                        if i < advance_len and state_len - advance_len + i >= 0:
                            Conv_State[conv_state_batch_coord, channel_id, state_len - advance_len + i] = x_val
                    else:
                        Conv_State[conv_state_batch_coord, channel_id, update_idx] = x_val
                        update_idx = (update_idx + 1) % state_len
                    x_vals[width - 1] = x_val
                    out_val = T.alloc_var(T.float32)
                    out_val = bias_val
                    for j in T.unroll(width):
                        out_val += weight_vals[j] * x_vals[j]
                    if silu_activation:
                        out_val = T.sigmoid(out_val) * out_val
                    Out[batch_id, channel_id, i] = out_val
                    # shift x_vals
                    for j in T.unroll(width - 1):
                        x_vals[j] = x_vals[j + 1]

    return causal_conv1d_update_main


# TODO: support complex layout
def causal_conv1d_update(x,
                         conv_state,
                         weight,
                         bias=None,
                         activation=None,
                         cache_seqlens=None,
                         conv_state_indices=None,
                         output_state_indices=None):
    """Tilelang implementation of causal_conv1d_update.

    Only the last indexed candidate may reuse a private input after reading its channel-local history. The owner must
    guarantee exclusivity and positive retained advancement; shared or zero-advance contracts require disjoint
    destinations. Legacy calls remain unchanged.

    Internal indexed mode uses ordered K-1 history per unit and int64 [B,T] output IDs. Sources are read-only,
    destinations are private; the host owner must validate uniqueness and disjointness from all live inputs before
    submission. Negative output IDs mask unused candidates; unit zero is never writable. This does not enable compact
    mode in the serving engine.
    """
    assert x.dim() in (2, 3)
    assert conv_state.dim() == 3
    assert weight.dim() == 2
    assert activation in ['silu', 'swish', None]

    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)

    has_bias = bias is not None
    width = weight.size(-1)
    _, hidden_size, seqlen = x.shape
    batch = x.size(0)
    state_len = conv_state.size(-1)

    if conv_state_indices is not None:
        assert conv_state_indices.dim() == 1 and conv_state_indices.is_contiguous()
        assert conv_state_indices.dtype in ((torch.int32, torch.int64) if output_state_indices is not None
                                             else (torch.int32,))
        assert conv_state_indices.device == x.device
        assert conv_state_indices.numel() == batch
    if cache_seqlens is not None:
        assert cache_seqlens.dim() == 1 and cache_seqlens.is_contiguous()
        assert cache_seqlens.dtype == torch.int32
        assert cache_seqlens.device == x.device
        assert cache_seqlens.numel() == batch

    if output_state_indices is not None:
        if conv_state_indices is None or cache_seqlens is not None or state_len != width - 1 or width < 2:
            raise ValueError('Indexed conv requires input IDs, ordered K-1 history (K>=2), and no ring offsets.')
        if (output_state_indices.shape != (batch, seqlen)
                or output_state_indices.dtype != torch.int64
                or output_state_indices.device != x.device
                or not output_state_indices.is_contiguous()):
            raise ValueError('Output state IDs must be contiguous int64 [B, T] on the input device.')

    out = x.new_empty(x.shape)

    num_warps = 2
    kernel = causal_conv1d_update_fwd(hidden_size=hidden_size,
                                      seqlen=seqlen,
                                      state_len=state_len,
                                      width=width,
                                      has_bias=has_bias,
                                      activation=activation,
                                      dtype=x.dtype,
                                      conv_stride=conv_state.stride(),
                                      is_circular_buffer=cache_seqlens is not None,
                                      has_state_indices=conv_state_indices is not None,
                                      num_warps=num_warps,
                                      weight_dtype=weight.dtype,
                                      bias_dtype=bias.dtype if bias is not None else x.dtype,
                                      x_stride=x.stride(),
                                      has_output_state_indices=output_state_indices is not None,

                                      index_dtype=conv_state_indices.dtype if conv_state_indices is not None
                                      else torch.int32)

    kernel(x, conv_state, weight, bias, out, cache_seqlens, conv_state_indices, output_state_indices)

    if unsqueeze:
        out = out.squeeze(-1)

    return out
