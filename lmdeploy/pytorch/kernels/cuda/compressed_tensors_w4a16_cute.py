# Copyright (c) OpenMMLab. All rights reserved.
"""Hopper W4A16: TMA double buffering, register dequantization, BF16 WGMMA RS.

Weights retain compressed-tensors' offset-binary INT4 / group-32 layout.
Routing is padded once to make activation tiles TMA addressable. Swapping
GEMM A/B puts dequantized weights in registers and BF16 activations in shared
memory. No FP8 rounding or expanded weight cache is used.
"""
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
import triton
import triton.language as tl
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack, make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from .moe.fused_moe import _get_sorted_idx_blocks, _renormalize, moe_reduce


@dsl_user_op
def _dequant_pair(word, shift, scale, *, loc=None, ip=None):
    """Two adjacent INT4 codes -> BF16x2, with exact -8 then BF16 scale.

    The 128+bias bit construction follows Marlin's BF16 conversion strategy;
    our canonical nibble layout additionally moves the high nibble to bit 16.
    Do not fuse bias subtraction with scaling: that changes rounding.
    """
    return Int32(
        llvm.inline_asm(T.i32(), [
            Int32(word).ir_value(loc=loc, ip=ip),
            Int32(shift).ir_value(loc=loc, ip=ip),
            Float32(scale).ir_value(loc=loc, ip=ip)
        ], '{ .reg .b32 v, hi, pair, ss, bias; .reg .b16 s; '
                        'shr.u32 v, $1, $2; shl.b32 hi, v, 12; '
                        'lop3.b32 pair, hi, 0x000f0000, 0x43004300, 0xea; '
                        'lop3.b32 pair, v, 0x0000000f, pair, 0xea; '
                        'mov.b32 bias, 0x43084308; sub.rn.bf16x2 pair, pair, bias; '
                        'cvt.rn.bf16.f32 s, $3; mov.b32 ss, {s, s}; '
                        'mul.rn.bf16x2 $0, pair, ss; }',
                        '=r,r,r,f',
                        has_side_effects=False,
                        is_align_stack=False,
                        loc=loc,
                        ip=ip))


@triton.jit
def _gather(X, Y, ROUTES, ENDS, BLOCK_END, EXPERTS, OFFSETS, K: tl.constexpr, STRIDE: tl.constexpr, TOPK: tl.constexpr,
            BM: tl.constexpr, NB: tl.constexpr, BK: tl.constexpr):
    block = tl.program_id(0)
    cols = tl.program_id(1) * BK + tl.arange(0, BK)
    rows = tl.arange(0, BM)
    active = block < tl.load(BLOCK_END + NB - 1)
    expert = tl.load(EXPERTS + block, active, 0)
    start = tl.load(OFFSETS + block, active, 0)
    end = tl.load(ENDS + expert, active, 0)
    valid = active & (start + rows < end)
    route = tl.load(ROUTES + start + rows, valid, 0)
    x = tl.load(X + (route // TOPK)[:, None] * STRIDE + cols[None, :], valid[:, None] & (cols[None, :] < K), 0)
    tl.store(Y + (block * BM + rows[:, None]) * K + cols[None, :], x, active & (cols[None, :] < K))


class CuteW4A16Gemm:
    """One warpgroup consumes a two-stage TMA pipeline (weights +
    activations)."""

    def __init__(self,
                 block_m=8,
                 block_k=128,
                 stages=2,
                 split_k=1,
                 fast_dequant=False,
                 single_route=False,
                 n_major=False):
        self.bm = block_m
        self.bk = block_k
        self.stages = stages
        self.split_k = split_k
        self.fast_dequant = fast_dequant
        self.single_route = single_route
        self.n_major = n_major

    @cute.kernel
    def kernel(self, q: cute.Tensor, w: cute.Tensor, s: cute.Tensor, out: cute.Tensor, routes: cute.Tensor,
               ends: cute.Tensor, block_end: cute.Tensor, experts: cute.Tensor, offsets: cute.Tensor,
               scatter: cutlass.Constexpr, mma: cute.TiledMma, atom_q: cute.CopyAtom, atom_w: cute.CopyAtom,
               b_layout: cute.ComposedLayout, w_layout: cute.Layout, ts: cute.Tensor, atom_s: cute.CopyAtom,
               s_layout: cute.Layout, stage_scales: cutlass.Constexpr):
        bid, tile_n, split = cute.arch.block_idx()
        if cutlass.const_expr(self.n_major):
            tile_n, bid, split = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        smem = utils.SmemAllocator()
        barriers = smem.allocate_array(cutlass.Int64, self.stages)
        b = smem.allocate_tensor(cutlass.BFloat16, b_layout.outer, byte_alignment=128, swizzle=b_layout.inner)
        wp = smem.allocate_tensor(Int32, w_layout, byte_alignment=128)
        if cutlass.const_expr(stage_scales):
            sp = smem.allocate_tensor(s.element_type, s_layout, byte_alignment=128)
        if bid < block_end[block_end.shape[0] - 1]:
            # The predicate is CTA-uniform. Inactive capacity tiles need
            # neither initialized barriers nor a CTA synchronization.
            if tid == 0:
                for stage in cutlass.range_constexpr(self.stages):
                    cute.arch.mbarrier_init(barriers + stage, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.sync_threads()
            expert = Int64(experts[bid])
            gq = cute.local_tile(q, (self.bm, self.bk), (bid, None))
            gw = cute.local_tile(w, (64, self.bk // 8, 1), (tile_n, None, expert))
            gw = gw[None, None, 0, None]
            tb, tq = cpasync.tma_partition(atom_q, 0, cute.make_layout(1), cute.group_modes(b, 0, 2),
                                           cute.group_modes(gq, 0, 2))
            tw, tg = cpasync.tma_partition(atom_w, 0, cute.make_layout(1), cute.group_modes(wp, 0, 2),
                                           cute.group_modes(gw, 0, 2))
            if cutlass.const_expr(stage_scales):
                gs = cute.local_tile(ts, (64, self.bk // 32, 1), (tile_n, None, expert))
                gs = gs[None, None, 0, None]
                tss, tsg = cpasync.tma_partition(atom_s, 0, cute.make_layout(1), cute.group_modes(sp, 0, 2),
                                                 cute.group_modes(gs, 0, 2))
            total_tiles = cute.ceil_div(s.shape[2] * 32, self.bk)
            per_split = cute.ceil_div(total_tiles, self.split_k)
            first_tile = split * per_split
            tiles = min(per_split, total_tiles - first_tile)
            tx_bytes = self.bm * self.bk * 2 + 64 * self.bk // 2
            if cutlass.const_expr(stage_scales):
                tx_bytes += 64 * (self.bk // 32) * (s.element_type.width // 8)
            # Prime the pipeline. Each transaction completes the stage's
            # mbarrier; CTA sync + WGMMA wait protect it against early reuse.
            if tid // 32 == 0:
                for stage in cutlass.range_constexpr(self.stages):
                    if stage < tiles:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, tx_bytes)
                        cute.copy(atom_q, tq[None, first_tile + stage], tb[None, stage], tma_bar_ptr=barriers + stage)
                        cute.copy(atom_w, tg[None, first_tile + stage], tw[None, stage], tma_bar_ptr=barriers + stage)
                        if cutlass.const_expr(stage_scales):
                            cute.copy(atom_s,
                                      tsg[None, first_tile + stage],
                                      tss[None, stage],
                                      tma_bar_ptr=barriers + stage)
            thr = mma.get_slice(tid)
            ca = thr.partition_A(cute.make_identity_tensor((64, self.bk)))
            cc = thr.partition_C(cute.make_identity_tensor((64, self.bm)))
            ra = mma.make_fragment_A(ca.shape)
            tmp = cute.make_rmem_tensor(ra.shape, Float32)
            rb = mma.make_fragment_B(thr.partition_B(b))
            acc = cute.make_rmem_tensor(mma.partition_shape_C((64, self.bm)), Float32)
            acc.fill(0)
            atom = cute.make_mma_atom(mma.op)
            atom.set(warpgroup.Field.ACCUMULATE, True)
            for tile in range(tiles):
                stage = tile % self.stages
                phase = (tile // self.stages) % 2
                cute.arch.mbarrier_wait(barriers + stage, phase)
                if cutlass.const_expr(self.fast_dequant and s.element_type == cutlass.BFloat16):
                    ra32 = cute.recast_tensor(ra, Int32)
                    for i in cutlass.range_constexpr(cute.size(ra32)):
                        n, k = ca[2 * i]
                        value = Int32(0)
                        valid = True
                        if cutlass.const_expr(s.shape[1] % 64 != 0):
                            valid = tile_n * 64 + n < s.shape[1]
                        if cutlass.const_expr((s.shape[2] * 32) % self.bk != 0):
                            valid = valid and (first_tile + tile) * self.bk + k < s.shape[2] * 32
                        if valid:
                            if cutlass.const_expr(stage_scales):
                                scale = sp[n, k // 32, stage]
                            else:
                                scale = s[expert, tile_n * 64 + n, ((first_tile + tile) * self.bk + k) // 32]
                            value = _dequant_pair(wp[n, k // 8, stage], (k % 8) * 4, scale)
                        ra32[i] = value
                else:
                    for i in cutlass.range_constexpr(cute.size(ra)):
                        n, k = ca[i]
                        value = Float32(0)
                        valid = True
                        if cutlass.const_expr(s.shape[1] % 64 != 0):
                            valid = tile_n * 64 + n < s.shape[1]
                        if cutlass.const_expr((s.shape[2] * 32) % self.bk != 0):
                            valid = valid and (first_tile + tile) * self.bk + k < s.shape[2] * 32
                        if valid:
                            word = wp[n, k // 8, stage]
                            code = ((word >> ((k % 8) * 4)) & 15) - 8
                            if cutlass.const_expr(stage_scales):
                                scale = sp[n, k // 32, stage]
                            else:
                                scale = s[expert, tile_n * 64 + n, ((first_tile + tile) * self.bk + k) // 32]
                            value = Float32(code) * Float32(scale)
                        tmp[i] = value
                    ra.store(tmp.load().to(cutlass.BFloat16))
                warpgroup.fence()
                for sub in cutlass.range_constexpr(self.bk // 16):
                    cute.mma_atom_call(atom, acc[None, 0, 0], ra[None, 0, sub], rb[None, 0, sub, stage], acc[None, 0,
                                                                                                             0])
                warpgroup.commit_group()
                warpgroup.wait_group(0)
                cute.arch.sync_threads()
                # TMA for the recycled slot overlaps the next tile's unpack,
                # scale multiplication and asynchronous BF16 tensor-core MMA.
                if tid // 32 == 0 and tile + self.stages < tiles:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(barriers + stage, tx_bytes)
                    cute.copy(atom_q,
                              tq[None, first_tile + tile + self.stages],
                              tb[None, stage],
                              tma_bar_ptr=barriers + stage)
                    cute.copy(atom_w,
                              tg[None, first_tile + tile + self.stages],
                              tw[None, stage],
                              tma_bar_ptr=barriers + stage)
                    if cutlass.const_expr(stage_scales):
                        cute.copy(atom_s,
                                  tsg[None, first_tile + tile + self.stages],
                                  tss[None, stage],
                                  tma_bar_ptr=barriers + stage)
            for i in cutlass.range_constexpr(cute.size(acc)):
                n = tile_n * 64 + cc[i][0]
                row = cc[i][1]
                if n < s.shape[1]:
                    if cutlass.const_expr(scatter):
                        if cutlass.const_expr(self.single_route):
                            if row == 0:
                                out[bid, n] = out.element_type(acc[i])
                        else:
                            sorted_row = offsets[bid] + row
                            if sorted_row < ends[expert]:
                                route = Int32(routes[sorted_row])
                                out[route, n] = out.element_type(acc[i])
                    else:
                        out[split * (out.shape[0] // self.split_k) + bid * self.bm + row, n] = out.element_type(acc[i])

    @cute.jit
    def __call__(self, q: cute.Tensor, packed: cute.Tensor, s: cute.Tensor, out: cute.Tensor, routes: cute.Tensor,
                 ends: cute.Tensor, block_end: cute.Tensor, experts: cute.Tensor, offsets: cute.Tensor,
                 scatter: cutlass.Constexpr, stream: cuda.CUstream):
        mma = cute.make_tiled_mma(
            warpgroup.MmaF16BF16Op(cutlass.BFloat16, Float32, (64, self.bm, 16), warpgroup.OperandSource.RMEM,
                                   warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K))
        b_layout = sm90_utils.make_smem_layout_b(utils.LayoutEnum.ROW_MAJOR, (64, self.bm, self.bk), cutlass.BFloat16,
                                                 self.stages)
        w_layout = cute.make_layout((64, self.bk // 8, self.stages), stride=(self.bk // 8, 1, 64 * self.bk // 8))
        w = cute.make_tensor(
            packed.iterator,
            cute.make_layout((packed.shape[1], packed.shape[2], packed.shape[0]),
                             stride=(packed.stride[1], packed.stride[2], packed.stride[0])))
        atom_q, tq = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), q,
                                                 cute.slice_(b_layout, (None, None, 0)), (self.bm, self.bk))
        atom_w, tw = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), w,
                                                 cute.slice_(w_layout, (None, None, 0)), (64, self.bk // 8))
        # TMA requires at least 16 contiguous bytes. Smaller K tiles retain
        # direct scale loads, including unaligned/tail test shapes.
        stage_scales = (self.bk // 32 * (s.element_type.width // 8) >= 16
                        and s.stride[1] * (s.element_type.width // 8) % 16 == 0)
        s_layout = cute.make_layout((64, self.bk // 32, self.stages), stride=(self.bk // 32, 1, 64 * self.bk // 32))
        ts, atom_s = s, atom_w
        if cutlass.const_expr(stage_scales):
            sv = cute.make_tensor(
                s.iterator,
                cute.make_layout((s.shape[1], s.shape[2], s.shape[0]), stride=(s.stride[1], s.stride[2], s.stride[0])))
            atom_s, ts = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), sv,
                                                     cute.slice_(s_layout, (None, None, 0)), (64, self.bk // 32))
        grid = (experts.shape[0], cute.ceil_div(s.shape[1], 64), self.split_k)
        if cutlass.const_expr(self.n_major):
            grid = (grid[1], grid[0], grid[2])
        self.kernel(tq, tw, s, out, routes, ends, block_end, experts, offsets, scatter, mma, atom_q, atom_w, b_layout,
                    w_layout, ts, atom_s, s_layout, stage_scales).launch(grid=grid, block=(128, 1, 1), stream=stream)


@lru_cache(maxsize=128)
def _compile_gemm(layouts, device_index, scatter, block_m, block_k, stages, split_k, fast_dequant, single_route,
                  n_major):
    dtypes = {torch.int32: Int32, torch.int64: Int64, torch.bfloat16: cutlass.BFloat16, torch.float32: Float32}
    tensors = [
        make_fake_tensor(dtypes[dtype], shape, stride=stride, assumed_align=16) for shape, stride, dtype in layouts
    ]
    with torch.cuda.device(device_index):
        return cute.compile(CuteW4A16Gemm(block_m, block_k, stages, split_k, fast_dequant, single_route, n_major),
                            *tensors, scatter, cuda.CUstream(0))


def _launch_gemm(x,
                 packed,
                 scales,
                 out,
                 metadata,
                 scatter=True,
                 block_m=8,
                 block_k=128,
                 stages=2,
                 split_k=1,
                 fast_dequant=False,
                 reduce_split=True,
                 single_route=False,
                 n_major=False):
    routes, _, ends, block_end, experts, offsets = metadata
    if scatter and split_k != 1:
        raise ValueError('split-K is supported only for the padded gate/up output')
    target = out if split_k == 1 else torch.empty(
        (out.shape[0] * split_k, out.shape[1]), device=out.device, dtype=torch.float32)
    tensors = (x, packed, scales, target, routes, ends, block_end, experts, offsets)
    layouts = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors)
    fn = _compile_gemm(layouts, x.device.index, scatter, block_m, block_k, stages, split_k, fast_dequant, single_route,
                       n_major)
    fn(*(from_dlpack(t, assumed_align=16) for t in tensors),
       cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    if split_k != 1 and reduce_split:
        _sum_split[(triton.cdiv(out.numel(), 512), )](target, out, block_end, out.shape[0], out.shape[1],
                                                      block_end.numel(), block_m, split_k, 512)
    return out if reduce_split else target


@triton.jit
def _sum_split(X, Y, BLOCK_END, M: tl.constexpr, N: tl.constexpr, E: tl.constexpr, BM: tl.constexpr,
               SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active_rows = tl.load(BLOCK_END + E - 1) * BM
    valid = (idx < M * N) & (idx // N < active_rows)
    value = tl.full((BLOCK, ), 0, tl.float32)
    for part in range(SPLIT):
        value += tl.load(X + part * M * N + idx, valid, 0)
    tl.store(Y + idx, value, idx < M * N)


def gather_routed(x, metadata, topk, block_m=8):
    routes, _, ends, block_end, experts, offsets = metadata
    padded = x.new_empty((experts.numel() * block_m, x.shape[1]))
    _gather[(experts.numel(), triton.cdiv(x.shape[1], 128))](x, padded, routes, ends, block_end, experts,
                                                             offsets, x.shape[1], x.stride(0), topk, block_m,
                                                             block_end.numel(), 128)
    return padded


@triton.jit
def _gather_single(X, Y, BLOCK_END, K: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr, BLOCK: tl.constexpr):
    """One token: each route owns a block; sorting/counting is unnecessary."""
    block = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + idx % K, (idx < BM * K) & (idx < K), 0)
    tl.store(Y + block * BM * K + idx, value, idx < BM * K)
    if block == 0 and tl.program_id(1) == 0:
        tl.store(BLOCK_END, TOPK)


@triton.jit
def _activate_padded(X,
                     Y,
                     BLOCK_END,
                     M: tl.constexpr,
                     F: tl.constexpr,
                     E: tl.constexpr,
                     BM: tl.constexpr,
                     BLOCK: tl.constexpr,
                     SPLIT: tl.constexpr = 1):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = idx // F, idx % F
    valid = (row < M) & (row < tl.load(BLOCK_END + E - 1) * BM)
    gate = tl.load(X + row * 2 * F + col, valid, 0).to(tl.float32)
    up = tl.load(X + row * 2 * F + F + col, valid, 0).to(tl.float32)
    for part in range(1, SPLIT):
        gate += tl.load(X + part * M * 2 * F + row * 2 * F + col, valid, 0)
        up += tl.load(X + part * M * 2 * F + row * 2 * F + F + col, valid, 0)
    # Preserve the original GEMM -> BF16 -> SiLU rounding boundary.
    gate = gate.to(Y.dtype.element_ty).to(tl.float32)
    up = up.to(Y.dtype.element_ty).to(tl.float32)
    value = gate / (1 + tl.exp(-gate)) * up
    tl.store(Y + idx, value, valid)


def _validate_inputs(hidden_states, gate_up_packed, gate_up_scale, down_packed, down_scale):
    x = hidden_states
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16:
        raise ValueError('cute requires 2D CUDA BF16 activations')
    if x.stride(1) != 1:
        raise ValueError('cute requires contiguous activation channels')
    if gate_up_packed.ndim != 3 or down_packed.ndim != 3:
        raise ValueError('cute requires 3D packed expert weights')
    if torch.cuda.get_device_capability(x.device)[0] != 9:
        raise ValueError('cute requires Hopper SM90')
    e, n, _ = gate_up_packed.shape
    m, k = x.shape
    if e < 1 or k < 32 or k % 32 or n < 64 or n % 64:
        raise ValueError('cute requires positive experts, group-size 32 and aligned gate/up channels')
    for packed, scale, out_dim, in_dim in ((gate_up_packed, gate_up_scale, n, k), (down_packed, down_scale, k, n // 2)):
        if packed.shape != (e, out_dim, in_dim // 8) or scale.shape != (e, out_dim, in_dim // 32):
            raise ValueError('cute requires packed [E,N,K/8] and scales [E,N,K/32]')
        if packed.dtype != torch.int32 or scale.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError('cute requires INT32 packed weights and BF16/FP32 scales')
        if packed.device != x.device or scale.device != x.device:
            raise ValueError('cute weights and activations must be on the same CUDA device')
        if not packed.is_contiguous() or not scale.is_contiguous():
            raise ValueError('cute requires contiguous weights and scales')
    return e, n, m, k


def fused_moe_w4a16_cute(hidden_states: torch.Tensor,
                         gate_up_packed: torch.Tensor,
                         gate_up_scale: torch.Tensor,
                         down_packed: torch.Tensor,
                         down_scale: torch.Tensor,
                         topk_weights: torch.Tensor,
                         topk_ids: torch.Tensor,
                         topk: int,
                         renormalize: bool = False,
                         num_bits: int = 4,
                         group_size: int = 32,
                         allow_invalid_routes: bool = False) -> torch.Tensor:
    """Hopper TP / DeepEP normal provider; routing stays on GPU during
    graphs."""
    if num_bits != 4 or group_size != 32:
        raise ValueError('cute supports only INT4 group-size 32')
    if allow_invalid_routes and renormalize:
        raise ValueError('Cannot renormalize an EP-local route subset')
    x = hidden_states
    e, n, m, k = _validate_inputs(x, gate_up_packed, gate_up_scale, down_packed, down_scale)
    if topk < 1 or (topk > e and not allow_invalid_routes):
        raise ValueError('cute requires 1 <= top_k <= E for global routes')
    if topk_ids.shape != (m, topk) or topk_weights.shape != (m, topk):
        raise ValueError('Routing dimensions must match token count and top_k')
    if topk_ids.dtype not in (torch.int32,
                              torch.int64) or topk_ids.device != x.device or topk_weights.device != x.device:
        raise ValueError('Routing requires CUDA int32/int64 ids on the activation device')
    if m == 0:
        return torch.empty_like(x)
    routing_ids, routing_e = topk_ids, e
    if allow_invalid_routes:
        valid = (topk_ids >= 0) & (topk_ids < e)
        topk_weights = torch.where(valid, topk_weights, 0)
        # The shared sorter requires nonnegative IDs and topk <= E. A private
        # sentinel preserves route positions but is excluded from local GEMMs.
        routing_ids = torch.where(valid, topk_ids, e).reshape(-1, 1)
        routing_e += 1
    single_route = m == 1 and not allow_invalid_routes
    # Static shape selection is safe for CUDA graphs: routing remains on GPU.
    bm = 8 if m * topk <= e * 8 else 16 if m * topk <= e * 16 else 32 if m * topk <= e * 32 else 64
    split = 8 if m <= 1 and k >= 1024 else 4 if m <= 8 and k >= 1024 else 1
    gate_bk = 256 if m > 1 and k % 256 == 0 else 128
    if single_route:
        ids = topk_ids.flatten().contiguous()
        block_end = torch.empty(1, device=x.device, dtype=torch.int32)
        # Unused general-route fields alias ids; the single-route epilogue
        # indexes output by block, also correctly handling duplicate ids.
        metadata = (ids, ids, ids, block_end, ids, ids)
        padded = x.new_empty((topk * bm, k))
        _gather_single[(topk, triton.cdiv(bm * k, 1024))](x, padded, block_end, k, topk, bm, 1024)
    else:
        metadata = _get_sorted_idx_blocks(routing_ids.contiguous(), routing_e, e, 0, bm)
        padded = gather_routed(x, metadata, topk, bm)
    # Invalid routes must be zero even if their supplied weight is NaN/nonzero.
    out = x.new_zeros((m * topk, k)) if allow_invalid_routes else x.new_empty((m * topk, k))
    _run_experts(padded, gate_up_packed, gate_up_scale, down_packed, down_scale, metadata, out, bm, gate_bk, split,
                 single_route, m > 8)
    return moe_reduce(out.view(m, topk, k), _renormalize(topk_weights, renormalize), fp32_acc=True)


def _run_experts(padded,
                 gate_up_packed,
                 gate_up_scale,
                 down_packed,
                 down_scale,
                 metadata,
                 out,
                 bm,
                 gate_bk=256,
                 split=1,
                 single_route=False,
                 n_major=True):
    """Common padded gate/activation/down pipeline for TP and both EP modes."""
    x, n = padded, gate_up_packed.shape[1]
    gate_up = x.new_empty((padded.shape[0], n))
    gate_up = _launch_gemm(padded,
                           gate_up_packed,
                           gate_up_scale,
                           gate_up,
                           metadata,
                           False,
                           bm,
                           block_k=gate_bk,
                           split_k=split,
                           fast_dequant=True,
                           reduce_split=False,
                           single_route=single_route,
                           n_major=n_major)
    activated = x.new_empty((padded.shape[0], n // 2))
    _activate_padded[(triton.cdiv(activated.numel(), 512), )](gate_up, activated, metadata[3], activated.shape[0],
                                                              n // 2, metadata[3].numel(), bm, 512, split)
    _launch_gemm(activated,
                 down_packed,
                 down_scale,
                 out,
                 metadata,
                 True,
                 bm,
                 fast_dequant=True,
                 single_route=single_route)


@triton.jit
def _masked_block_metadata(COUNTS, ENDS, BLOCK_END, EXPERTS, OFFSETS, E: tl.constexpr, CAP: tl.constexpr,
                           BM: tl.constexpr, BE: tl.constexpr, BC: tl.constexpr):
    expert = tl.program_id(0)
    ids = tl.arange(0, BE)
    counts = tl.minimum(tl.maximum(tl.load(COUNTS + ids, ids < E, 0), 0), CAP)
    blocks = tl.cdiv(counts, BM)
    start = tl.sum(tl.where(ids < expert, blocks, 0))
    count = tl.sum(tl.where(ids == expert, counts, 0))
    end = start + tl.cdiv(count, BM)
    tl.store(ENDS + expert, expert * CAP + count)
    tl.store(BLOCK_END + expert, end)
    b = tl.arange(0, BC)
    tl.store(EXPERTS + start + b, expert, start + b < end)
    tl.store(OFFSETS + start + b, expert * CAP + b * BM, start + b < end)


def fused_moe_w4a16_cute_masked(hidden_states: torch.Tensor,
                                gate_up_packed: torch.Tensor,
                                gate_up_scale: torch.Tensor,
                                down_packed: torch.Tensor,
                                down_scale: torch.Tensor,
                                masked_m: torch.Tensor,
                                num_bits: int = 4,
                                group_size: int = 32) -> torch.Tensor:
    """Unweighted [E_local, capacity, H] experts for DeepEP low-latency
    combine.

    Compact only valid tiles on device; never read uninitialized receive tails. DeepEP, not this kernel, applies router
    weights at combine time.
    """
    if num_bits != 4 or group_size != 32:
        raise ValueError('cute supports only INT4 group-size 32')
    if hidden_states.ndim != 3 or not hidden_states.is_contiguous():
        raise ValueError('masked cute requires contiguous [experts, capacity, hidden]')
    e, capacity, k = hidden_states.shape
    if capacity < 1 or masked_m.shape != (e, ) or masked_m.dtype not in (torch.int32, torch.int64):
        raise ValueError('masked_m must contain one integer count per expert and capacity must be positive')
    if masked_m.device != hidden_states.device or not masked_m.is_contiguous():
        raise ValueError('masked_m must be contiguous on the activation device')
    x = hidden_states.view(-1, k)
    weight_e, _, _, _ = _validate_inputs(x, gate_up_packed, gate_up_scale, down_packed, down_scale)
    if weight_e != e:
        raise ValueError('masked weights must match the local expert count')
    bm = 8
    routes = torch.arange(e * capacity, device=x.device, dtype=torch.int32)
    ends = torch.empty(e, device=x.device, dtype=torch.int32)
    block_end = torch.empty_like(ends)
    experts = torch.empty(e * triton.cdiv(capacity, bm), device=x.device, dtype=torch.int32)
    offsets = torch.empty_like(experts)
    _masked_block_metadata[(e, )](masked_m, ends, block_end, experts, offsets, e, capacity, bm,
                                  triton.next_power_of_2(e), triton.next_power_of_2(triton.cdiv(capacity, bm)))
    metadata = (routes, ends, ends, block_end, experts, offsets)
    padded = gather_routed(x, metadata, 1, bm)
    out = torch.zeros_like(x)
    _run_experts(padded,
                 gate_up_packed,
                 gate_up_scale,
                 down_packed,
                 down_scale,
                 metadata,
                 out,
                 bm,
                 gate_bk=256 if k % 256 == 0 else 128)
    return out.view_as(hidden_states)
