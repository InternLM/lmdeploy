# Copyright (c) Shanghai AI Laboratory. All rights reserved.
"""TurboMind grouped W4A16 routed experts over canonical AWQ parameters.

The module accepts the canonical AWQ layout that every supported checkpoint
source normalizes into at load time: AWQ native, plus compressed-tensors and
GPTQ through ``normalize_ct_weights``/``normalize_gptq_weights``. Weights
stay packed per projection and per expert. The TurboMind channel packs them
once in ``update_weights``, and forward consumes only activations plus
routing.

Scope: eager default TP (moe-TP slices every expert along the
expert width). EP and DeepEP dispatch stay on the compressed-tensors route.
"""

from itertools import product

import torch
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.tm_moe_modules import FusedMoETmW4A16BuildSpec, turbomind_moe_support_reason
from lmdeploy.pytorch.backends.cuda.tm_weight_release import TorchWeightSpill
from lmdeploy.pytorch.backends.tm_moe_modules import FusedMoETmW4A16Impl
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import get_dist_manager, get_tp_world_rank
from lmdeploy.pytorch.models.patch import get_build_model_context

from .base import FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce, update_dims

_PARTS = ('qweight', 'scales', 'qzeros')


class TmMoeWeights(nn.Module):
    """Rank-local canonical AWQ parameters for one fused MoE projection."""

    def __init__(self, num_experts: int, in_features: int, out_features: int, group_size: int, weight_type: str,
                 device: torch.device):
        super().__init__()
        if weight_type not in ('gate_up', 'down'):
            raise ValueError(f'Unknown weight type: {weight_type!r}')
        self.weight_type = weight_type
        self.group_size = group_size
        self.in_features = in_features
        self.out_features = out_features
        # gate_up stores the gate half first, the up half second; the split
        # marks the per-projection width inside the fused storage.
        self.split = out_features // 2 if weight_type == 'gate_up' else None
        num_words = out_features // 8
        num_groups = in_features // group_size

        shapes = {
            'qweight': (num_experts, in_features, num_words),
            'scales': (num_experts, num_groups, out_features),
            'qzeros': (num_experts, num_groups, num_words),
        }
        self._loaded_parts: set[tuple[int, str, str]] = set()
        for name, shape in shapes.items():
            dtype = torch.int32 if name != 'scales' else torch.float16
            parameter = nn.Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
            parameter.weight_loader = self.weight_loader
            self.register_parameter(name, parameter)

    def _expected_shards(self) -> tuple[str, ...]:
        if self.weight_type == 'gate_up':
            return ('gate', 'up')
        return ('down', )

    def _expected_keys(self) -> set:
        """The complete loaded-part ledger of a fully loaded module."""
        return set(product(range(self.qweight.shape[0]), self._expected_shards(), _PARTS))

    def _half(self, shard_id: str) -> tuple[int, int]:
        """Column range of one logical projection inside the fused gate_up
        storage."""
        if self.weight_type == 'gate_up':
            if shard_id == 'gate':
                return 0, self.split
            if shard_id == 'up':
                return self.split, 2 * self.split
            raise ValueError(f'gate_up weights reject shard {shard_id!r}')
        if shard_id == 'down':
            return 0, self.out_features
        raise ValueError(f'down weights reject shard {shard_id!r}')

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, expert_id: int = -1,
                      shard_id: str = ''):
        """Validate and copy one checkpoint tensor owned by this rank."""
        if expert_id < 0 or shard_id not in self._expected_shards():
            raise ValueError(f'AWQ routed-expert loading requires expert_id and a {self._expected_shards()} '
                             f'shard, got expert_id={expert_id} shard_id={shard_id!r}')
        if param is self.qweight:
            part = 'qweight'
        elif param is self.scales:
            part = 'scales'
        elif param is self.qzeros:
            part = 'qzeros'
        else:
            raise ValueError(f'Unknown AWQ routed-expert part for {param!r}')
        key = (expert_id, shard_id, part)
        if key in self._loaded_parts:
            if self._loaded_parts == self._expected_keys():
                # A complete module re-fed from the start is a hot update:
                # reset the ledger and accept the new stream.
                self._loaded_parts.clear()
            else:
                raise RuntimeError(f'Duplicate AWQ checkpoint tensor for expert={expert_id}, shard={shard_id}, '
                                   f'part={part}')

        world_size, rank = get_tp_world_rank('moe')
        # gate/up slice the expert width (last axis of the checkpoint
        # tensor); down slices the input width. The moe-TP grid keeps every
        # sliced side on the checkpoint group grid (guarded by the probe),
        # so equal chunks stay scale-coherent.
        chunk_axis = 0 if self.weight_type == 'down' else 1
        shard = loaded_weight.chunk(world_size, dim=chunk_axis)[rank]

        begin, end = self._half(shard_id)
        if part == 'scales':
            destination = param.data[expert_id, :, begin:end]
        elif part == 'qweight':
            destination = param.data[expert_id, :, begin // 8:end // 8]
        else:
            destination = param.data[expert_id, :, begin // 8:end // 8]
        if tuple(shard.shape) != tuple(destination.shape):
            raise RuntimeError(f'AWQ routed-expert shard shape mismatch for expert={expert_id}, '
                               f'shard={shard_id}, part={part}: source={tuple(shard.shape)}, '
                               f'destination={tuple(destination.shape)}; check the moe-TP slicing grid')
        destination.copy_(shard.to(destination.device))
        self._loaded_parts.add(key)

    def validate_complete(self):
        """Reject inference unless every local projection triplet was
        loaded."""
        expected = self._expected_keys()
        missing = sorted(expected - self._loaded_parts)
        unexpected = sorted(self._loaded_parts - expected)
        if missing or unexpected:
            missing_preview = ', '.join(map(str, missing[:8]))
            unexpected_preview = ', '.join(map(str, unexpected[:8]))
            raise RuntimeError(f'Incomplete AWQ {self.weight_type} weights: missing={len(missing)} '
                               f'[{missing_preview}], unexpected={len(unexpected)} [{unexpected_preview}]')


class FusedMoETmW4A16(FusedMoEBase):
    """TurboMind grouped W4A16 MoE over canonical AWQ expert weights."""

    # The routed reduce matches the compressed-tensors contract: FP32
    # products and one outer TP reduction.
    tp_reduce_dtype = torch.float32

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        renormalize: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        all_reduce: bool = True,
        num_bits: int = 4,
        group_size: int = 128,
        layer_idx: int = 0,
    ):
        device = device or torch.device('cpu')
        dtype = dtype or torch.float16
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f'TurboMind W4A16 requires a 16-bit floating dtype, got {dtype}')

        self.init_dist_args(all_reduce)
        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        if dist_config.ep > 1:
            raise RuntimeError('TurboMind W4A16 MoE does not wire EP dispatch in this batch; '
                               'expert-parallel models must stay on the compressed-tensors route.')
        if self.tp_mode != TPMode.DEFAULT:
            raise RuntimeError('TurboMind W4A16 MoE only supports eager default TP mode.')
        if dist_config.enable_eplb:
            raise RuntimeError('TurboMind W4A16 MoE does not support EPLB.')
        if dist_config.enable_microbatch:
            raise RuntimeError('TurboMind W4A16 MoE does not support microbatch execution.')

        super().__init__(tp=self.tp, tp_mode=self.tp_mode, do_renormalize=renormalize)

        hidden_dim, local_ffn_dim = update_dims(hidden_dim, ffn_dim)
        # The builder checked the global shapes; the moe-TP slice must sit on
        # the same registry alignment grid or the pack would miscompute.
        reason = turbomind_moe_support_reason(hidden_dim, local_ffn_dim, num_bits, group_size, dtype)
        if reason is not None:
            raise RuntimeError(f'TurboMind W4A16 MoE rejected the TP-sliced expert geometry: {reason}')

        self.impl: FusedMoETmW4A16Impl = get_backend().build_op(
            FusedMoETmW4A16BuildSpec(
                top_k=top_k,
                num_experts=num_experts,
                hidden_dim=hidden_dim,
                ffn_dim=local_ffn_dim,
                renormalize=renormalize,
                group_size=group_size,
                output_dtype=dtype,
                prefix=f'layer_{layer_idx}',
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

        self.gate_up = TmMoeWeights(num_experts, hidden_dim, 2 * local_ffn_dim, group_size, 'gate_up', device)
        self.down = TmMoeWeights(num_experts, local_ffn_dim, hidden_dim, group_size, 'down', device)
        self.hidden_dim = hidden_dim
        self.ffn_dim = local_ffn_dim
        self.num_experts = num_experts
        self.release_torch_weights = True
        self.dtype = dtype
        self.device = device
        self._spill: TorchWeightSpill | None = None

    def update_weights(self):
        """Pack the loaded AWQ experts into the TurboMind channel, then spill
        the torch weights."""
        if self.gate_up.qweight.numel() == 0:
            # The module was not part of this update bucket; its storages
            # are spilled and packing would read the emptied tensors.
            return
        self.gate_up.validate_complete()
        self.down.validate_complete()
        self.impl.update_weights(
            self.gate_up.qweight.data,
            self.gate_up.scales.data,
            self.gate_up.qzeros.data,
            self.down.qweight.data,
            self.down.scales.data,
            self.down.qzeros.data,
        )
        if not self.release_torch_weights:
            # Keep the torch copies resident (mirrors the dense route's MLA
            # exemption).
            return
        if self._spill is None:
            self._spill = TorchWeightSpill()
        # Bias is not supported on routed experts, so nothing stays on the
        # GPU for forward.
        self._spill.spill({
            'gate_up.qweight': self.gate_up.qweight,
            'gate_up.scales': self.gate_up.scales,
            'gate_up.qzeros': self.gate_up.qzeros,
            'down.qweight': self.down.qweight,
            'down.scales': self.down.scales,
            'down.qzeros': self.down.qzeros,
        })

    def dispatch(self, state: dict):
        """Gather eager default-TP inputs."""
        moe_type = state['moe_type']
        if moe_type != MoeType.Default:
            raise NotImplementedError(f'TurboMind W4A16 MoE does not support MoE mode: {moe_type}')
        hidden_states, topk_weights, topk_idx = moe_gather_inputs(
            state['hidden_states'],
            state['topk_weights'],
            state['topk_idx'],
            group=self.gather_group,
        )
        return {
            'hidden_states': hidden_states,
            'topk_idx': topk_idx,
            'topk_weights': topk_weights,
            'moe_type': moe_type,
        }

    def gemm(self, state: dict):
        """Run the TurboMind grouped implementation."""
        output = self.impl.forward(
            state['hidden_states'],
            state['topk_weights'],
            state['topk_idx'],
        )
        return {'hidden_states': output, 'moe_type': state['moe_type']}

    def combine(self, state: dict):
        """Apply the existing MoE TP reduction contract exactly once."""
        moe_type = state['moe_type']
        if moe_type != MoeType.Default:
            raise NotImplementedError(f'TurboMind W4A16 MoE does not support MoE mode: {moe_type}')
        if self.all_reduce:
            state['hidden_states'] = moe_reduce(
                state['hidden_states'],
                rank=self.tp_rank,
                tp_mode=self.tp_mode,
                group=self.tp_group,
            )
        return {'hidden_states': state['hidden_states'], 'moe_type': moe_type}

    def wait(self, state: dict):
        """Async execution is outside the synchronous eager path."""
        raise NotImplementedError('TurboMind W4A16 MoE only supports synchronous eager execution.')
