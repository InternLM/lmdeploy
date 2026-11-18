# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""W4A16 grouped MoE implementation driven by the TurboMind Linear API.

Differences from ``TritonFusedMoEW4A16Impl``:

- Expert weights pack once into native handles (``LinkLinearExperts``), one
  handle per projection. Forward consumes only activations and routing.
- The slot layout follows the TurboMind blocked striding. Rows of the
  activated intermediate sit expert-major at ``offsets[e]``. The PyTorch
  sorter stays the routing source of truth, so both backends walk the same
  route table (see ``blocked_route``).
- SM86 has no gated-silu grouped family. Gate and up run as two indexed
  GEMMs and SiLU applies in python, matching the TurboMind C++ separate
  w1/w3 production path. Families that represent the fusion (SM90) pack one
  interleaved gate/up handle instead. The probe is a real ``fuse_weight``
  call, not a plan query, because some families expose a silu plan without
  a fusion block.

Communication stays on the python side (``moe_reduce``), identical to the
compressed-tensors route.
"""

import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.kernels.cuda.activation import silu_and_mul
from lmdeploy.pytorch.kernels.cuda.moe.fused_moe import _get_sorted_idx, moe_reduce
from lmdeploy.turbomind.linear import Weight, get_linear
from lmdeploy.turbomind.weight_format import AWQFormat
from lmdeploy.utils import get_logger

from ..tm_moe_modules import FusedMoETmW4A16BuildSpec, FusedMoETmW4A16Impl

logger = get_logger('lmdeploy')


def blocked_route(topk_ids: torch.Tensor, num_experts: int):
    """Convert flat top-k routes into the TurboMind blocked slot layout.

    Returns ``(indices, offsets, sorted_routes, valid)``:

    - ``indices`` int32 ``[num_routes]``: source token row for every slot.
      Slots sit expert-major, so ``indices[offsets[e] + i]`` is the token of
      the i-th route of expert ``e``.
    - ``offsets`` int32 ``[num_experts + 1]``: expert prefix sums. The last
      entry starts the skip block.
    - ``sorted_routes`` long ``[num_routes]``: flat route position of every
      slot, the inverse permutation for the output scatter.
    - ``valid`` bool ``[m, topk]``: false for skip sentinels.

    The PyTorch sorter behind ``_get_sorted_idx`` is the same one the triton
    fused_moe anchor uses, so both backends consume one route table. Skip
    sentinels (``-1``) park in an extra block after the last real expert.
    The grouped GEMM walks ``num_experts`` groups only, so it never writes
    their output rows. The caller zeroes them before the reduce.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f'topk_ids must be a 2D tensor, but got dim={topk_ids.dim()}')
    if topk_ids.shape[1] > num_experts:
        raise ValueError(f'topk_ids.size(1) must be <= num_experts, but got top_k={topk_ids.shape[1]} '
                         f'and num_experts={num_experts}')
    valid = (topk_ids >= 0) & (topk_ids < num_experts)
    # Park invalid routes behind the real experts. The sorter sees one extra
    # expert; the grouped GEMM never launches that block.
    routing_ids = torch.where(valid, topk_ids, torch.full_like(topk_ids, num_experts))
    sorted_routes, exp_start, exp_end = _get_sorted_idx(routing_ids, num_experts + 1)
    offsets = torch.cat((exp_start[:num_experts], exp_end[num_experts - 1:num_experts])).to(torch.int32)
    top_k = topk_ids.shape[1]
    indices = sorted_routes.div(top_k, rounding_mode='floor').to(torch.int32)
    return indices, offsets, sorted_routes.long(), valid


class TurbomindMoeW4A16Impl(FusedMoETmW4A16Impl):
    """Adapt canonical AWQ expert tensors to the TurboMind grouped channel."""

    def __init__(
        self,
        top_k: int,
        num_experts: int,
        hidden_dim: int,
        ffn_dim: int,
        group_size: int,
        renormalize: bool = False,
    ):
        if top_k < 1 or num_experts < 1 or top_k > num_experts:
            raise ValueError(f'Expected 1 <= top_k <= num_experts, got {top_k} and {num_experts}')
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.group_size = group_size
        self.renormalize = renormalize
        self._linear = None
        self._plan: dict = {}
        self._w_gate: Weight | None = None
        self._w_up: Weight | None = None
        self._w_down: Weight | None = None
        self._w13: Weight | None = None
        # None means the fusion probe has not run yet. False is sticky: the
        # SM86 u4 families have no gated-silu block, and re-probing on every
        # hot update would waste a failed pack.
        self._fused_silu: bool | None = None

    def _weight_plan(self, linear, fusion_type: str | None, dtype: torch.dtype):
        """Cached registry plan.

        The query is deterministic per format.
        """
        key = fusion_type or 'plain'
        if key not in self._plan:
            self._plan[key] = linear.get_weight_plan(
                weight_format=AWQFormat(block_in=self.group_size),
                dtype=dtype,
                grouped=True,
                fusion_type=fusion_type,
            )
        return self._plan[key]

    @staticmethod
    def _expert_views(tensor: torch.Tensor) -> list[torch.Tensor]:
        """Split one stacked expert tensor into its per-expert views."""
        return [tensor[expert] for expert in range(tensor.shape[0])]

    def _release(self):
        weights = (self._w_gate, self._w_up, self._w_down, self._w13)
        self._w_gate = self._w_up = self._w_down = self._w13 = None
        if all(weight is None for weight in weights):
            self._linear = None
            return
        # Forward is asynchronous and may use any engine stream. Release the
        # prepared weights only after all device work completes.
        torch.cuda.synchronize()
        for weight in weights:
            if weight is not None:
                weight.close()
        self._linear = None

    def __del__(self):
        try:
            if self._w_gate is None and self._w_up is None and self._w_down is None and self._w13 is None:
                return
            self._release()
        except Exception:
            pass

    def update_weights(
        self,
        gate_up_qweight: torch.Tensor,
        gate_up_scales: torch.Tensor,
        gate_up_qzeros: torch.Tensor,
        down_qweight: torch.Tensor,
        down_scales: torch.Tensor,
        down_qzeros: torch.Tensor,
    ):
        """Pack the canonical AWQ expert tensors into the TurboMind channel."""
        self._release()

        num_words = self.ffn_dim // 8
        gate_qweight = gate_up_qweight[..., :num_words]
        up_qweight = gate_up_qweight[..., num_words:]
        gate_scales = gate_up_scales[..., :self.ffn_dim]
        up_scales = gate_up_scales[..., self.ffn_dim:]
        gate_qzeros = gate_up_qzeros[..., :num_words]
        up_qzeros = gate_up_qzeros[..., num_words:]

        linear = get_linear()
        dtype = gate_up_scales.dtype
        if self._fused_silu is None:
            # The probe is the pack itself: families without a fusion block
            # (SM86 u4) reject inside fuse_weight, and the fallback packing
            # below keeps the layer on the separate w1/w3 path.
            try:
                self._w13 = linear.fuse_weight(
                    (self._expert_views(gate_qweight), self._expert_views(up_qweight)),
                    plan=self._weight_plan(linear, 'silu', dtype),
                    scales=(self._expert_views(gate_scales), self._expert_views(up_scales)),
                    zeros=(self._expert_views(gate_qzeros), self._expert_views(up_qzeros)),
                )
                self._fused_silu = True
            except NotImplementedError:
                # Sticky: the sticky flag spares hot updates a repeated
                # failed pack on devices without a fused family.
                self._fused_silu = False

        if self._fused_silu:
            if self._w13 is None:
                self._w13 = linear.fuse_weight(
                    (self._expert_views(gate_qweight), self._expert_views(up_qweight)),
                    plan=self._weight_plan(linear, 'silu', dtype),
                    scales=(self._expert_views(gate_scales), self._expert_views(up_scales)),
                    zeros=(self._expert_views(gate_qzeros), self._expert_views(up_qzeros)),
                )
            self._w_gate = self._w_up = None
        else:
            plan = self._weight_plan(linear, None, dtype)
            self._w_gate = linear.prepare_weight(
                self._expert_views(gate_qweight),
                plan=plan,
                scales=self._expert_views(gate_scales),
                zeros=self._expert_views(gate_qzeros),
            )
            self._w_up = linear.prepare_weight(
                self._expert_views(up_qweight),
                plan=plan,
                scales=self._expert_views(up_scales),
                zeros=self._expert_views(up_qzeros),
            )
            self._w13 = None
        self._w_down = linear.prepare_weight(
            self._expert_views(down_qweight),
            plan=self._weight_plan(linear, None, dtype),
            scales=self._expert_views(down_scales),
            zeros=self._expert_views(down_qzeros),
        )

        self._linear = linear

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.LongTensor,
    ):
        """Run the routed experts on the blocked slot layout and reduce."""
        if self._w_down is None or (self._w13 is None and (self._w_gate is None or self._w_up is None)):
            raise RuntimeError('update_weights() must run before forward (TM packed expert weights missing)')
        linear = self._linear

        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if not topk_weights.is_contiguous():
            topk_weights = topk_weights.contiguous()

        in_shape = hidden_states.shape
        x = hidden_states.flatten(0, -2)
        if not x.is_contiguous():
            x = x.contiguous()
        num_tokens = x.shape[0]
        if num_tokens == 0:
            return x.new_zeros(in_shape[:-1] + (self.hidden_dim, ))

        indices, offsets, sorted_routes, valid = blocked_route(topk_ids, self.num_experts)
        slots = indices.shape[0]

        if self._fused_silu:
            exec13 = linear.get_exec_plan(x, self._w13, offsets=offsets, indices=indices)
            inter, _ = linear.forward_moe(x, self._w13, exec_plan=exec13, offsets=offsets, indices=indices)
            if inter.shape[-1] != self.ffn_dim:
                raise RuntimeError(
                    f'The fused SiLU family returned {inter.shape[-1]} columns, expected {self.ffn_dim}; '
                    'the family does not apply the epilogue activation it advertised.')
        else:
            # One exec plan serves gate and up: same problem geometry, and the
            # native per-problem cache keeps the second query cheap.
            exec13 = linear.get_exec_plan(x, self._w_gate, offsets=offsets, indices=indices)
            gate, _ = linear.forward_moe(x, self._w_gate, exec_plan=exec13, offsets=offsets, indices=indices)
            up, _ = linear.forward_moe(x, self._w_up, exec_plan=exec13, offsets=offsets, indices=indices)
            inter = silu_and_mul(torch.cat((gate, up), dim=-1))

        # The grouped GEMM consumes the slot-major activation with pointer
        # shifts only; rows of the output sit at offsets[e].
        exec2 = linear.get_exec_plan(inter, self._w_down, offsets=offsets)
        slot_out, _ = linear.forward_moe(inter, self._w_down, exec_plan=exec2, offsets=offsets)

        # Un-permute slot-major rows to flat route order. Skip sentinels land
        # in the block behind offsets[-1] that no kernel wrote, so zero them
        # before the weighted reduce.
        expert_out = torch.zeros((slots, self.hidden_dim), dtype=slot_out.dtype, device=slot_out.device)
        expert_out.index_copy_(0, sorted_routes, slot_out)
        expert_out = expert_out.view(num_tokens, self.top_k, self.hidden_dim)
        expert_out.masked_fill_(~valid.unsqueeze(-1), 0)
        return moe_reduce(expert_out, topk_weights, fp32_acc=True)


def turbomind_moe_support_reason(in_features: int, out_features: int, w_bit: int, group_size: int,
                                 dtype: torch.dtype | None = None) -> str | None:
    """Return why the TurboMind grouped MoE route cannot serve this expert
    shape, or None when it can.

    Mirrors ``tm_probe.turbomind_support_reason`` with ``grouped=True``. The
    kernel registry is the ground truth. This probe only translates its
    verdict into a readable reason.
    """
    if w_bit != 4:
        return f'w_bit must be 4, but got {w_bit}'
    if group_size <= 0:
        return f'group_size must be positive, but got {group_size}'
    if not torch.cuda.is_available():
        return 'CUDA is unavailable'

    from lmdeploy import turbomind
    if not turbomind.is_available():
        return 'TurboMind native modules are not built'

    from lmdeploy.turbomind.linear import get_linear

    probe_dtype = torch.float16 if dtype is None else dtype
    try:
        plan = get_linear().get_weight_plan(
            weight_format=AWQFormat(block_in=group_size),
            dtype=probe_dtype,
            grouped=True,
        )
    except NotImplementedError as e:
        logger.debug('TurboMind grouped probe rejected format '
                     f'awq group_size={group_size} dtype={probe_dtype}: {e}')
        return f'no grouped GEMM family accepts awq group_size={group_size} ' \
               f'dtype={probe_dtype}: {e}'

    (min_k, min_n), (align_k, align_n) = plan.shape_constraints
    for name, in_features_dim, out_features_dim in (('gate_up', in_features, out_features),
                                                    ('down', out_features, in_features)):
        if in_features_dim % align_k != 0:
            return f'{name}: K must be a multiple of {align_k}, but got {in_features_dim}'
        if out_features_dim % align_n != 0:
            return f'{name}: N must be a multiple of {align_n}, but got {out_features_dim}'
        if in_features_dim < min_k:
            return f'{name}: K must be at least {min_k}, but got {in_features_dim}'
        if out_features_dim < min_n:
            return f'{name}: N must be at least {min_n}, but got {out_features_dim}'
    return None


def tm_moe_route(group_size: int,
                 w_bit: int,
                 dtype: torch.dtype | None,
                 hidden_dim: int,
                 ffn_dim: int,
                 prefix: str = '') -> str | None:
    """Decide the grouped MoE route with the unified TM kernel semantics.

    ``LMDEPLOY_TM_KERNELS`` decides through the ``moe_grouped`` class;
    with the big env silent for that class, the legacy
    ``LMDEPLOY_W4A16_GEMM_BACKEND`` keeps its three states.
    ``auto`` probes and reports the fallback reason. ``on`` raises on an
    unsupported layer. ``off`` routes straight to the triton fused_moe
    path. Returns None when the TurboMind grouped route serves this
    layer, else the reason string.
    """
    state = _envs.tm_kernel_state('moe_grouped')
    label = prefix or 'w4a16_moe'
    shapes = f'{hidden_dim}x{ffn_dim}'
    if state == 'off':
        logger.info(f'W4A16 MoE GEMM routing: {label} [{shapes}] -> fused_moe triton (backend={state}).')
        return f'W4A16 GEMM backend={state}'
    reason = turbomind_moe_support_reason(hidden_dim, ffn_dim, w_bit, group_size, dtype)
    if reason is None:
        logger.info(f'W4A16 MoE GEMM routing: {label} [{shapes}] -> turbomind (backend={state}).')
        return None
    if state == 'on':
        from lmdeploy import turbomind

        cause = turbomind._import_error
        raise RuntimeError(
            f'TurboMind grouped MoE GEMM was requested but is unavailable or incompatible: {reason}.'
        ) from cause
    logger.info(f'W4A16 MoE GEMM routing: {label} [{shapes}] -> fused_moe triton (fallback: {reason}).')
    return reason


def _build_fused_moe_tm(spec: FusedMoETmW4A16BuildSpec) -> FusedMoETmW4A16Impl:
    """Build the grouped MoE implementation on the TurboMind channel.

    The module builder consults ``tm_moe_route`` before choosing this
    implementation. Reaching the builder with a declined route means an env
    flip between the two build steps. Fail loudly instead of silently
    switching the weight layout.
    """
    if _envs.tm_kernel_state('moe_grouped') == 'off':
        raise RuntimeError(
            'TurboMind grouped MoE spec built while the moe_grouped kernel state is off '
            '(LMDEPLOY_TM_KERNELS or LMDEPLOY_W4A16_GEMM_BACKEND); '
            'the route decision and the operator build disagree.')
    return TurbomindMoeW4A16Impl(
        top_k=spec.top_k,
        num_experts=spec.num_experts,
        hidden_dim=spec.hidden_dim,
        ffn_dim=spec.ffn_dim,
        group_size=spec.group_size,
        renormalize=spec.renormalize,
    )
