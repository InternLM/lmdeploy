# Copyright (c) OpenMMLab. All rights reserved.
"""SM90 block-FP8 MegaMoE for the GLM-5.3 expert layout."""

from __future__ import annotations

import json
from functools import partial
from importlib.util import find_spec
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from lmdeploy.pytorch import envs
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul
from lmdeploy.pytorch.kernels.cuda.moe.ep_fp8 import project_gate_up
from lmdeploy.pytorch.model_inputs import get_step_ctx_manager
from lmdeploy.utils import get_logger

from .ep_utils import mask_invalid_tokens

if TYPE_CHECKING:
    from lmdeploy.pytorch.model_inputs import BuildModelContext
    from lmdeploy.pytorch.nn.moe.blocked_fp8 import FusedMoEBlockedF8

logger = get_logger('lmdeploy')
ExpertWeights = tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _deep_gemm_provider():
    if find_spec('deep_gemm_mega') is not None:
        import deep_gemm_mega as provider
    else:
        import deep_gemm as provider
    return provider


@triton.jit
def _pack_gate_up(src, scale, dst, H: tl.constexpr, I: tl.constexpr, MERGE_SCALES: tl.constexpr,
                  BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    row, col = offset // H, offset % H
    # Eight adjacent rows of gate followed by eight of up, for the same GLU channel.
    gate_row = row // 16 * 8 + row % 8
    src_row = gate_row + (row // 8 % 2) * I
    value = tl.load(src + expert * (2 * I * H) + src_row * H + col)
    if MERGE_SCALES:
        scale_cols: tl.constexpr = H // 128
        base = scale + expert * (2 * I // 128 * scale_cols) + col // 128
        gate = tl.load(base + gate_row // 128 * scale_cols)
        up = tl.load(base + (gate_row + I) // 128 * scale_cols)
        old_scale = tl.where(row // 8 % 2 == 0, gate, up)
        new_scale = tl.maximum(gate, up)
        ratio = tl.where(new_scale > 0, old_scale / new_scale, 1.0)
        value = (value.to(tl.float32) * ratio).to(src.dtype.element_ty)
    tl.store(dst + expert * (2 * I * H) + offset, value)


@triton.jit(do_not_specialize=['active_mask'])
def _unpack_gate_up(src, dst, active_mask, H: tl.constexpr, I: tl.constexpr, BLOCK: tl.constexpr):
    expert = tl.program_id(1)
    if (active_mask.to(tl.uint64) >> expert) & 1:
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        row, col = offset // H, offset % H
        channel = row % I
        src_row = channel // 8 * 16 + (row // I) * 8 + channel % 8
        expert_offset = expert * (2 * I * H)
        value = tl.load(src + expert_offset + src_row * H + col)
        tl.store(dst + expert_offset + offset, value)


@triton.jit(do_not_specialize=['rows'])
def _stage_mega_moe(hidden, ids, weights, x, x_sf, route_ids, route_weights, valid_tokens, shared_x, shared_sf,
                    rows, H: tl.constexpr, K: tl.constexpr, HIDDEN_ROW: tl.constexpr,
                    SF_ROW: tl.constexpr, SF_COL: tl.constexpr, IDS_ROW: tl.constexpr, WEIGHTS_ROW: tl.constexpr,
                    HAS_VALID_COUNT: tl.constexpr, DUMMY: tl.constexpr, ROUTED_SCALE: tl.constexpr,
                    MIN_SCALE: tl.constexpr, EXACT_DIV: tl.constexpr, SHARED_BLOCK_M: tl.constexpr,
                    SHARED_SF_COL: tl.constexpr, TAIL_ROWS: tl.constexpr, CAPACITY: tl.constexpr):
    row = tl.program_id(0)
    has_row = row < rows
    if TAIL_ROWS > 0:
        if not has_row:
            # The pull backend scans every routing slot, even on idle ranks.
            # Clear a contiguous tile of invalid routes without staging activations.
            start = rows + (row - rows) * TAIL_ROWS
            slots = start * K + tl.arange(0, TAIL_ROWS * K)
            tl.store(route_ids + slots, -1, mask=slots < CAPACITY * K)
            tl.store(route_weights + slots, 0, mask=slots < CAPACITY * K)
            return
    valid = has_row & (not DUMMY)
    if HAS_VALID_COUNT:
        valid = has_row & (row < tl.load(valid_tokens))
    columns = tl.arange(0, H)
    values = tl.load(hidden + row * HIDDEN_ROW + columns, mask=valid, other=0).to(tl.float32)
    grouped = tl.reshape(values, (H // 128, 128))
    scales = tl.maximum(tl.max(tl.abs(grouped), 1) / 448.0, MIN_SCALE)
    if EXACT_DIV:
        scaled = tl.div_rn(grouped, scales[:, None])
    else:
        scaled = grouped / scales[:, None]
    quantized = tl.reshape(tl.clamp(scaled, -448.0, 448.0), (H,))
    tl.store(x + row * H + columns, quantized, mask=has_row)
    tl.store(x_sf + row * SF_ROW + tl.arange(0, H // 128) * SF_COL, scales, mask=has_row)
    if SHARED_BLOCK_M > 0:
        # DeepGEMM lays each shared-expert block of token scales out in a
        # 128-row TMA tile, with the M lanes permuted within the tile.
        m = row % SHARED_BLOCK_M
        sf_row = row // SHARED_BLOCK_M * 128 + (m % 32) * 4 + m // 32
        tl.store(shared_x + row * H + columns, quantized, mask=has_row | (rows == 0))
        tl.store(shared_sf + sf_row + tl.arange(0, H // 128) * SHARED_SF_COL, scales, mask=has_row | (rows == 0))

    slots = tl.arange(0, K)
    expert_ids = tl.load(ids + row * IDS_ROW + slots, mask=valid, other=-1)
    topk_weights = tl.load(weights + row * WEIGHTS_ROW + slots, mask=expert_ids >= 0, other=0)
    tl.store(route_ids + row * K + slots, expert_ids)
    tl.store(route_weights + row * K + slots, topk_weights * ROUTED_SCALE)


class Sm90MegaMoE:
    """Model-owned runner. All MoE layers enqueue on the same compute stream."""

    merge_gate_up_scales = False
    fuses_shared_expert = False

    def __init__(self, capacity, group, config, *, weight_layout='dual'):
        self.capacity = capacity
        self.group = group
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.output_scale = config.routed_scaling_factor
        self.swiglu_limit = config.swiglu_limit
        self.prefill_capacity = 0
        self.prefill_runners: tuple[Sm90MegaMoE, ...] = ()
        if weight_layout not in ('dual', 'compact', 'interleaved', 'interleaved_fused'):
            raise ValueError(f'Unsupported MegaMoE FC1 weight layout: {weight_layout!r}.')
        if weight_layout != 'dual' and self.merge_gate_up_scales:
            raise ValueError('A single FC1 weight layout requires separate gate/up block scales.')
        self.weight_layout = weight_layout
        self.compact_weights = weight_layout != 'dual'
        self.prefill_gate_up_buffer = None
        if weight_layout.startswith('interleaved'):
            provider = _deep_gemm_provider()
            if not hasattr(provider, 'sm90_fp8_grouped_gate_up'):
                raise ImportError('Interleaved prefill requires the SM90 grouped gate/up extension for DeepGEMM.')
            self._gemm_gate_up = provider.sm90_fp8_grouped_gate_up
            self._interleaved_act = partial(silu_and_mul, swiglu_limit=self.swiglu_limit, precise_mul=True,
                                            gate_up_interleave=8, quantize_fp8=weight_layout == 'interleaved_fused')

    def configure_experts(self, experts: FusedMoEBlockedF8):
        """Configure canonical weight loading before the checkpoint is read."""
        if self.compact_weights:
            experts.gate_up.gate_up_interleave = 8
            if self.weight_layout.startswith('interleaved'):
                experts.impl.prefill_gate_up_projection = self._prefill_interleaved
            else:
                experts.impl.prefill_gate_up_projection = self._prefill_unpack
                if self.prefill_gate_up_buffer is None:
                    self.prefill_gate_up_buffer = torch.empty_like(experts.gate_up.weight)

    def unpack_gate_up(self, packed: torch.Tensor, received: list[int]) -> torch.Tensor:
        """Restore only experts addressed by this rank's DeepEP prefill."""
        active = sum(1 << expert for expert, tokens in enumerate(received) if tokens > 0)
        if active:
            block = 16384
            _unpack_gate_up[(packed.shape[1] * packed.shape[2] // block, packed.shape[0])](
                packed.view(torch.uint8), self.prefill_gate_up_buffer.view(torch.uint8), active,
                self.hidden_size, self.intermediate_size, block, num_warps=8)
        return self.prefill_gate_up_buffer

    def _prefill_unpack(self, input_fp8, weight_fp8, m_indices, received, act_func):
        packed, scale = weight_fp8
        return project_gate_up(input_fp8, (self.unpack_gate_up(packed, received), scale), m_indices, act_func)

    def _prefill_interleaved(self, input_fp8, weight_fp8, m_indices, received, act_func):
        return project_gate_up(input_fp8, weight_fp8, m_indices, self._interleaved_act, gemm=self._gemm_gate_up)

    def prepare_weights(self, experts: FusedMoEBlockedF8, previous: ExpertWeights | None) -> ExpertWeights:
        from lmdeploy.pytorch.nn.moe.blocked_fp8 import FusedMoEBlockedF8

        if not isinstance(experts, FusedMoEBlockedF8) or experts.gate_up.weight.dtype != torch.float8_e4m3fn:
            raise ValueError('SM90 MegaMoE requires E4M3 block-FP8 expert weights.')
        rank = self.group.rank()
        local = self.num_experts // self.group.size()
        if experts.expert_list != list(range(rank * local, (rank + 1) * local)):
            raise ValueError('SM90 MegaMoE requires consecutive experts on each EP rank.')

        gate_up = experts.gate_up.weight
        gate_up_sf = experts.gate_up.weight_scale_inv
        down = experts.down.weight
        down_sf = experts.down.weight_scale_inv
        if not gate_up.is_contiguous() or not gate_up_sf.is_contiguous():
            raise ValueError('SM90 MegaMoE packing requires contiguous checkpoint FC1 weights and scales.')
        if self.compact_weights:
            packed = gate_up
        else:
            packed = torch.empty_like(gate_up) if previous is None else previous[0][0]
            block = 2048
            _pack_gate_up[(local, gate_up.shape[1] * gate_up.shape[2] // block)](
                gate_up, gate_up_sf, packed, self.hidden_size, self.intermediate_size, self.merge_gate_up_scales, block)
        packed_sf, down_sf = self.prepare_scales(gate_up, gate_up_sf, down, down_sf)
        if previous is not None:
            previous[0][1].copy_(packed_sf)
            if previous[1][1].data_ptr() != down_sf.data_ptr():
                previous[1][1].copy_(down_sf)
            if previous[1][0].data_ptr() != down.data_ptr():
                previous[1][0].copy_(down)
            return (packed, previous[0][1]), previous[1]
        return (packed, packed_sf), (down, down_sf)

    def __call__(self, hidden, topk_weights, topk_ids, weights: ExpertWeights, **kwargs):
        if hidden.shape[0] > self.capacity:
            raise ValueError(f'MegaMoE got {hidden.shape[0]} rows with capacity {self.capacity}.')
        return self.forward(hidden, topk_weights, topk_ids, weights, **kwargs)

    def for_step(self, context):
        """Every EP rank selects the same runner, regardless of its local phase."""
        if context.global_is_decoding():
            return self
        if self.prefill_capacity and context.dp_meta is not None:
            counts = context.dp_meta.dp_token_counts
            if counts:
                max_rows = max(counts)
                if max_rows > self.prefill_capacity:
                    return None
                if max_rows <= self.capacity:
                    return self
                return next((runner for runner in self.prefill_runners if max_rows <= runner.capacity), None)
        return None

    def _stage_inputs(self, hidden, topk_weights, topk_ids, *, clear_tail, min_scale, exact_div, shared_block_m=0):
        manager = get_step_ctx_manager()
        ctx = manager.current_context() if manager is not None else None
        ws = self.workspace
        rows = hidden.shape[0]
        tail_rows = 64 if clear_tail else 0
        launch_rows = rows + triton.cdiv(ws.x.shape[0] - rows, tail_rows) if clear_tail else max(1, rows)
        valid_count = None if ctx is None else ctx.moe_valid_token_count
        is_dummy = False if ctx is None else ctx.is_dummy
        shared_x = ws.shared_l1_acts if shared_block_m else None
        shared_sf = ws.shared_l1_acts_sf if shared_block_m else None
        shared_sf_col = shared_sf.stride(1) if shared_sf is not None else 0
        _stage_mega_moe[(launch_rows,)](
            hidden, topk_ids, topk_weights, ws.x.view(torch.float8_e4m3fn), ws.x_sf, ws.topk_idx, ws.topk_weights,
            valid_count, shared_x, shared_sf, rows, self.hidden_size, self.top_k, hidden.stride(0), *ws.x_sf.stride(),
            topk_ids.stride(0), topk_weights.stride(0), valid_count is not None, is_dummy, self.output_scale,
            min_scale, exact_div, shared_block_m, shared_sf_col, tail_rows, ws.x.shape[0], num_warps=8)


class DeepGemmMegaMoE(Sm90MegaMoE):

    def __init__(self, capacity, group, config, *, fuse_shared=False, weight_layout='dual',
                 adaptive_block_m=False):
        super().__init__(capacity, group, config, weight_layout=weight_layout)
        provider = _deep_gemm_provider()
        if not hasattr(provider, 'transform_fp8_block_sf_for_sm90_mega_moe'):
            raise ImportError('DeepGEMM MegaMoE requires the SM90 FP8 block-scale API from DeepGEMM PR #57.')
        self._launch = provider.fp8_fp4_mega_moe
        self._transform_sf = provider.transform_fp8_block_sf_for_sm90_mega_moe
        self._transform_weights = provider.transform_weights_for_mega_moe
        self._get_block_m = provider.get_block_m_for_mega_moe
        self.fuses_shared_expert = fuse_shared
        self.adaptive_block_m = adaptive_block_m
        self._shared_block_ms: dict[int, int] = {}
        self.workspace = provider.get_symm_buffer_for_mega_moe(
            group, self.num_experts, capacity, self.top_k, self.hidden_size, self.intermediate_size,
            num_shared_experts=int(fuse_shared), mma_type='fp8xfp8')
        self.shared_block_m = self._shared_block_m(capacity)

    def _shared_block_m(self, token_hint):
        if not self.fuses_shared_expert:
            return 0
        block_m = self._shared_block_ms.get(token_hint)
        if block_m is None:
            block_m = self._get_block_m(self.group.size(), self.num_experts, self.workspace.num_max_tokens_per_rank,
                                        token_hint, self.top_k, 'fp8xfp8')
            self._shared_block_ms[token_hint] = block_m
        return block_m

    def _token_hint(self):
        if self.adaptive_block_m:
            ctx = get_step_ctx_manager().current_context()
            if ctx.dp_meta is not None and ctx.dp_meta.dp_token_counts:
                counts = ctx.dp_meta.dp_token_counts
                return triton.cdiv(sum(counts), len(counts))
        return self.capacity

    def prepare_scales(self, gate_up, gate_up_sf, down, down_sf):
        return self._transform_sf(gate_up, gate_up_sf, is_l1=True), self._transform_sf(down, down_sf, is_l1=False)

    def stage_inputs(self, hidden, topk_weights, topk_ids, shared_block_m=None):
        self._stage_inputs(hidden, topk_weights, topk_ids, clear_tail=False, min_scale=1.e-4 / 448., exact_div=False,
                           shared_block_m=self.shared_block_m if shared_block_m is None else shared_block_m)

    def prepare_shared_weights(self, shared_expert, previous=None):
        gate_up, down = shared_expert.gate_up_proj, shared_expert.down_proj
        packed = self._transform_weights((gate_up.weight, gate_up.weight_scale_inv),
                                         (down.weight, down.weight_scale_inv), arch_major=9)
        if previous is None:
            return packed
        for new_pair, old_pair in zip(packed, previous):
            for new, old in zip(new_pair, old_pair):
                if old.data_ptr() != new.data_ptr():
                    old.copy_(new)
        return previous

    def forward(self, hidden, topk_weights, topk_ids, weights, shared_weights=None):
        if self.fuses_shared_expert and shared_weights is None:
            raise ValueError('DeepGEMM shared-expert fusion requires packed shared weights.')
        rows = hidden.shape[0]
        result = hidden.new_empty((max(1, rows), self.hidden_size))
        token_hint = self._token_hint()
        shared_block_m = self._shared_block_m(token_hint) if self.adaptive_block_m else self.shared_block_m
        self.stage_inputs(hidden, topk_weights, topk_ids, shared_block_m)
        # Choose the same BLOCK_M on every rank, including when a rank has no live rows.
        shared_l1, shared_l2 = shared_weights if shared_weights is not None else (None, None)
        self._launch(result, *weights, self.workspace, activation_clamp=self.swiglu_limit,
                     shared_l1_weights=shared_l1, shared_l2_weights=shared_l2, num_tokens_for_block_m=token_hint)
        return result if rows else result[:0]


class FlashInferMegaMoE(Sm90MegaMoE):
    fused_staging = True

    def __init__(self, capacity, group, config, *, dedup_dispatch=False, exact_scales=False, weight_layout='dual',
                 knobs=None):
        # The stock provider has one scale per 128 physical rows, which
        # contains 64 gate and 64 up rows. The extension accepts both scales.
        self.merge_gate_up_scales = not exact_scales
        self.knobs = knobs
        super().__init__(capacity, group, config, weight_layout=weight_layout)
        from flashinfer.moe_ep import BootstrapConfig, FleetParams, MoEEpTensors, Sm90PullFp8MegaMoeConfig
        from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
        from flashinfer.moe_ep.core.runtime import bootstrap_moe_ep_runtime

        self._tensors = MoEEpTensors
        bootstrap = BootstrapConfig(world_size=group.size(), rank=group.rank(), process_group=group,
                                    device=torch.cuda.current_device())
        fleet = FleetParams(num_experts=self.num_experts, max_tokens_per_rank=capacity,
                            token_hidden_size=self.hidden_size)
        extra = {}
        if exact_scales:
            if not hasattr(Sm90PullFp8MegaMoeConfig, 'fc1_split_weight_scales'):
                raise ImportError('Exact FlashInfer FC1 scales require the SM90 split-scale extension.')
            extra['fc1_split_weight_scales'] = True
        kernel_config = Sm90PullFp8MegaMoeConfig(intermediate_size=self.intermediate_size, top_k=self.top_k,
                                               fp8_scale_mode='blockwise', gate_up_clamp=self.swiglu_limit,
                                               dedup_dispatch=dedup_dispatch, knobs=knobs, **extra)
        self.kernel = create_mega_kernel(kernel_config)
        self.kernel.bind_ep_bootstrap(bootstrap)
        self.kernel.validate_init(bootstrap, fleet)
        self.runtime = bootstrap_moe_ep_runtime(bootstrap, self.kernel.runtime_requirements(bootstrap))
        self.workspace = self.kernel.prepare_workspace(bootstrap, fleet)

    def prepare_scales(self, gate_up, gate_up_sf, down, down_sf):
        gate, up = gate_up_sf.chunk(2, dim=1)
        if self.merge_gate_up_scales:
            return torch.maximum(gate, up).repeat_interleave(2, dim=1).contiguous(), down_sf
        return torch.stack((gate, up), dim=2).flatten(1, 2), down_sf

    def stage_inputs(self, hidden, topk_weights, topk_ids):
        if self.fused_staging:
            # FlashInfer scans the routing plane up to capacity. Clear tail
            # routes in the same launch that quantizes the live activation rows.
            self._stage_inputs(hidden, topk_weights, topk_ids, clear_tail=True, min_scale=1.e-30, exact_div=True)
            return
        ids = topk_ids.to(dtype=torch.int32, copy=True)
        mask_invalid_tokens(ids)
        inputs = self._tensors(hidden_states=hidden, topk_ids=ids, topk_weights=topk_weights * self.output_scale)
        self.kernel.stage_inputs(inputs, self.workspace, quantize_input=True)

    def forward(self, hidden, topk_weights, topk_ids, weights):
        rows = hidden.shape[0]
        # Even an empty source rank owns experts required by its peers.
        result = hidden.new_empty((max(1, rows), self.hidden_size))
        self.stage_inputs(hidden, topk_weights, topk_ids)
        (w1, s1), (w2, s2) = weights
        transformed = ((w1.transpose(1, 2), s1, None, None), (w2.transpose(1, 2), s2, None, None))
        output = self.kernel.compute(self.workspace, transformed, output=result)
        return output if rows else output[:0]


def build_sm90_mega_moe(config, dtype: torch.dtype, device: torch.device, build_ctx: BuildModelContext) -> Sm90MegaMoE:
    dist_ctx = get_dist_manager().current_context()
    dist_config = dist_ctx.dist_config
    if (dist_config.dp, dist_config.ep, dist_config.tp) != (8, 8, 1):
        raise ValueError('SM90 MegaMoE requires DP8 + EP8 + TP1.')
    if dtype != torch.bfloat16 or device is None or torch.device(device).type != 'cuda' or (
            torch.cuda.get_device_capability(device) != (9, 0)):
        raise ValueError('SM90 MegaMoE requires BF16 activations on a CUDA SM90 GPU.')
    quant = build_ctx.quant_config
    if quant.quant_method != 'fp8' or tuple(quant.weight_block_size or ()) != (128, 128) or quant.scale_fmt is not None:
        raise ValueError('SM90 MegaMoE requires 128x128 block-FP8 weights with FP32 scales.')
    expert_shape = (config.hidden_size, config.moe_intermediate_size,
                    config.n_routed_experts, config.num_experts_per_tok)
    if expert_shape != (4096, 2048, 288, 8):
        raise ValueError(f'Unsupported GLM-5.3-Flash expert shape for SM90 MegaMoE: {expert_shape}.')
    capacity = build_ctx.deep_ep_max_tokens_per_rank
    if build_ctx.moe_backend == 'deepgemm_mega_moe':
        build_runner = partial(DeepGemmMegaMoE, group=dist_ctx.ep_gpu_group, config=config,
                               fuse_shared=envs.glm53_deepgemm_fuse_shared,
                               weight_layout=envs.glm53_deepgemm_weight_layout)
    else:
        knobs = json.loads(envs.glm53_flashinfer_knobs) if envs.glm53_flashinfer_knobs else None
        if knobs is not None and not isinstance(knobs, dict):
            raise ValueError('LMDEPLOY_GLM53_FLASHINFER_KNOBS must be a JSON object.')
        build_runner = partial(FlashInferMegaMoE, group=dist_ctx.ep_gpu_group, config=config, knobs=knobs,
                               exact_scales=envs.glm53_flashinfer_exact_scales,
                               weight_layout=envs.glm53_flashinfer_weight_layout)
    runner = build_runner(capacity)
    prefill_capacity = envs.glm53_mega_prefill_capacity
    if prefill_capacity > 0:
        prefill_options = {'adaptive_block_m': True} if isinstance(runner, DeepGemmMegaMoE) else {}
        # Pull kernels scan a fixed workspace and tune their geometry for it.
        # Keep smaller plans for cache-hit turns and mixed steps with a short prefix.
        buckets = (64, 256, 1024) if isinstance(runner, FlashInferMegaMoE) else ()
        capacities = tuple(c for c in buckets if capacity < c < prefill_capacity)
        if prefill_capacity > capacity:
            capacities += (prefill_capacity,)
        runner.prefill_capacity = prefill_capacity
        runner.prefill_runners = tuple(build_runner(c, **prefill_options) for c in capacities)
        logger.info('GLM-5.3 MegaMoE mixed and prefill limit=%d rows/rank, workspace sizes=%s.',
                    prefill_capacity, (capacity, *capacities))
    logger.info('GLM-5.3 MoE decode uses %s, capacity=%d rows/rank, shared expert fused=%s, FC1 layout=%s, '
                'separate gate/up scales=%s.', build_ctx.moe_backend, runner.capacity, runner.fuses_shared_expert,
                runner.weight_layout, not runner.merge_gate_up_scales)
    return runner
