# Copyright (c) Shanghai AI Lab. All rights reserved.
"""TurboMind grouped MXFP4 routed experts over checkpoint-native parameters.

The module accepts the gpt-oss mxfp4 layout exactly as the checkpoint
stores it: e2m1 weight blocks (two codes per byte along the input dim), one
e8m0 byte scale per group of 32, and per-projection bias. The loader
deinterleaves the fused gate/up output dim before feeding the module, so
parameters hold the gate half first and the up half second. The TurboMind
channel packs them once in ``update_weights``; forward consumes only
activations plus routing and applies bias python-side at the blocked-slot
granularity (the grouped kernels have no bias epilogue).

Scope mirrors the W4A16 route: eager default TP (moe-TP slices every expert
along the expert width) and plain expert parallelism; EPLB, microbatch,
EP+DP and EP+TP are rejected loudly.
"""

from itertools import product

import torch
import torch.distributed
from torch import nn

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.backends.cuda.tm_moe_modules import FusedMoETmMXFP4BuildSpec, turbomind_mxfp4_support_reason
from lmdeploy.pytorch.backends.cuda.tm_weight_release import TorchWeightSpill
from lmdeploy.pytorch.backends.tm_moe_modules import FusedMoETmMXFP4Impl
from lmdeploy.pytorch.config import TPMode
from lmdeploy.pytorch.distributed import get_dist_manager, get_ep_world_rank, get_tp_world_rank
from lmdeploy.pytorch.models.patch import get_build_model_context

from .base import FusedMoEBase, MoeType, moe_gather_inputs, moe_reduce, update_dims

_PARTS = ('blocks', 'scales')


class TmMoeMXFP4Weights(nn.Module):
    """Rank-local MXFP4 parameters for one fused MoE projection.

    Parameters keep the checkpoint-native orientation: ``blocks``
    ``[E_local, out_features, in_features // 2]`` uint8, ``scales``
    ``[E_local, out_features, in_features // 32]`` uint8, ``bias``
    ``[E_local, out_features]``. ``MXFP4Format.normalize`` transposes into
    the TurboMind layout at pack time.
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int, bias: bool, weight_type: str,
                 dtype: torch.dtype, device: torch.device):
        super().__init__()
        if weight_type not in ('gate_up', 'down'):
            raise ValueError(f'Unknown weight type: {weight_type!r}')
        self.weight_type = weight_type
        self.with_bias = bias
        self.in_features = in_features
        self.out_features = out_features
        ep_world, ep_rank = get_ep_world_rank()
        if num_experts % ep_world != 0:
            raise ValueError(f'num_experts={num_experts} is not divisible by EP={ep_world}')
        # Contiguous expert blocks: this rank stores experts
        # [first_local_expert, first_local_expert + num_local_experts).
        num_local_experts = num_experts // ep_world
        first_local_expert = ep_rank * num_local_experts
        expert_list = list(range(first_local_expert, first_local_expert + num_local_experts))
        # Keep ``num_experts`` as the global router domain for compatibility.
        # Parameters, loader destinations and completeness are rank-local.
        self.global_num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.expert_list = expert_list
        self.expert_map = {global_expert_id: local_slot for local_slot, global_expert_id in enumerate(expert_list)}
        # gate_up stores the gate half first, the up half second; the split
        # marks the per-projection width inside the fused storage. The
        # loader deinterleaves the checkpoint's interleaved gate/up rows.
        self.split = out_features // 2 if weight_type == 'gate_up' else None

        shapes = {
            'blocks': (num_local_experts, out_features, in_features // 2),
            'scales': (num_local_experts, out_features, in_features // 32),
        }
        if bias:
            shapes['bias'] = (num_local_experts, out_features)
        self._loaded_parts: set[tuple[int, str, str]] = set()
        for name, shape in shapes.items():
            dtype_ = dtype if name == 'bias' else torch.uint8
            parameter = nn.Parameter(torch.empty(shape, dtype=dtype_, device=device), requires_grad=False)
            parameter.weight_loader = self.weight_loader
            self.register_parameter(name, parameter)

    def _expected_shards(self) -> tuple[str, ...]:
        if self.weight_type == 'gate_up':
            return ('gate', 'up')
        return ('down', )

    def _expected_parts(self) -> tuple[str, ...]:
        return _PARTS + (('bias', ) if self.with_bias else ())

    def _expected_keys(self) -> set:
        """The complete loaded-part ledger of a fully loaded module."""
        return set(product(range(self.blocks.shape[0]), self._expected_shards(), self._expected_parts()))

    def _half(self, shard_id: str) -> tuple[int, int]:
        """Row range of one logical projection inside the fused gate_up
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
        """Validate and copy one checkpoint tensor owned by this rank.

        The loaded tensor is one expert's already-deinterleaved projection
        (the gpt-oss loader strips the interleaved gate/up rows before
        calling): ``[out_features, in_features // 2]`` blocks, its scales,
        or the per-projection bias.
        """
        if not 0 <= expert_id < self.global_num_experts or shard_id not in self._expected_shards():
            raise ValueError(f'MXFP4 routed-expert loading requires a global expert_id in '
                             f'[0, {self.global_num_experts}) and a {self._expected_shards()} '
                             f'shard, got expert_id={expert_id} shard_id={shard_id!r}')
        if param is self.blocks:
            part = 'blocks'
        elif param is self.scales:
            part = 'scales'
        elif param is self.bias:
            part = 'bias'
        else:
            raise ValueError(f'Unknown MXFP4 routed-expert part for {param!r}')
        # Foreign experts belong to another EP rank: skip them silently, the
        # completeness ledger only tracks local slots.
        local_expert_id = self.expert_map.get(expert_id)
        if local_expert_id is None:
            return
        key = (local_expert_id, shard_id, part)
        if key in self._loaded_parts:
            if self._loaded_parts == self._expected_keys():
                # A complete module re-fed from the start is a hot update:
                # reset the ledger and accept the new stream.
                self._loaded_parts.clear()
            else:
                raise RuntimeError(f'Duplicate MXFP4 checkpoint tensor for expert={expert_id}, shard={shard_id}, '
                                   f'part={part}')

        world_size, rank = get_tp_world_rank('moe')
        # gate/up slice the expert output width (first axis of the fed
        # tensor); down slices the packed input width (second axis). The
        # moe-TP grid keeps every sliced side on the group-32 grid (guarded
        # by the post-slice probe), so chunks stay scale-coherent. Bias has
        # no input dim: gate/up bias rides the expert width slice, down bias
        # spans the (unsliced) hidden width.
        if part == 'bias' and self.weight_type == 'down':
            shard = loaded_weight
        else:
            chunk_axis = 0 if self.weight_type == 'gate_up' else 1
            shard = loaded_weight.chunk(world_size, dim=chunk_axis)[rank]

        begin, end = self._half(shard_id)
        if part == 'bias':
            destination = param.data[local_expert_id, begin:end]
        else:
            destination = param.data[local_expert_id, begin:end]
        if tuple(shard.shape) != tuple(destination.shape):
            raise RuntimeError(f'MXFP4 routed-expert shard shape mismatch for expert={expert_id}, '
                               f'shard={shard_id}, part={part}: source={tuple(shard.shape)}, '
                               f'destination={tuple(destination.shape)}; check the moe-TP slicing grid')
        destination.copy_(shard.to(destination.device))
        self._loaded_parts.add(key)

    def validate_complete(self):
        """Reject inference unless every local projection was loaded."""
        expected = self._expected_keys()
        missing = sorted(expected - self._loaded_parts)
        unexpected = sorted(self._loaded_parts - expected)
        if missing or unexpected:
            missing_preview = ', '.join(map(str, missing[:8]))
            unexpected_preview = ', '.join(map(str, unexpected[:8]))
            raise RuntimeError(f'Incomplete MXFP4 {self.weight_type} weights: missing={len(missing)} '
                               f'[{missing_preview}], unexpected={len(unexpected)} [{unexpected_preview}]')


class FusedMoETmMXFP4(FusedMoEBase):
    """TurboMind grouped MXFP4 MoE over checkpoint-native expert weights."""

    # The routed reduce matches the compressed-tensors contract: FP32
    # products and one outer TP reduction.
    tp_reduce_dtype = torch.float32

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
        bias: bool = True,
        renormalize: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        all_reduce: bool = True,
        group_size: int = 32,
        layer_idx: int = 0,
        act_func=None,
    ):
        device = device or torch.device('cpu')
        dtype = dtype or torch.bfloat16
        if dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f'TurboMind MXFP4 requires a 16-bit floating dtype, got {dtype}')

        self.init_dist_args(all_reduce)
        dist_ctx = get_dist_manager().current_context()
        dist_config = dist_ctx.dist_config
        if self.tp_mode != TPMode.DEFAULT:
            raise RuntimeError('TurboMind MXFP4 MoE only supports eager default TP mode.')
        if dist_config.enable_eplb:
            # EPLB permutes experts away from contiguous blocks, which the
            # rank-local expert_map below cannot express; re-evaluate it
            # together with EP before lifting this.
            raise RuntimeError('TurboMind MXFP4 MoE does not support EPLB.')
        if dist_config.enable_microbatch:
            raise RuntimeError('TurboMind MXFP4 MoE does not support microbatch execution.')
        if num_experts % self.ep != 0:
            raise RuntimeError(f'TurboMind MXFP4 MoE requires num_experts={num_experts} to be divisible by '
                               f'EP={self.ep}.')
        if top_k > num_experts:
            # The only router constraint that survives EP: the global router
            # may exceed the rank-local expert count (surplus foreign routes
            # park in the skip block), but more routes than experts would
            # silently produce all-zero outputs.
            raise RuntimeError(f'TurboMind MXFP4 MoE requires top_k={top_k} <= num_experts={num_experts}: '
                               'more routes than experts would silently produce all-zero outputs.')
        if self.ep > 1 and dist_config.dp > 1:
            # Plain EP serves the whole token batch per rank; DP attention
            # partitions tokens, which needs all-to-all dispatch (DeepEP) —
            # outside this route.
            raise RuntimeError('TurboMind MXFP4 MoE EP requires dp=1: every rank must see the full token '
                               'batch, but DP attention partitions tokens per rank.')
        if self.ep > 1 and self.tp > 1:
            # Expert blocks (EP) and moe-TP width shards cannot co-exist:
            # each expert would materialize at partial width on one rank.
            raise RuntimeError(f'TurboMind MXFP4 MoE EP requires moe_tp=1, got EP={self.ep}, '
                               f'moe_tp={self.tp}.')
        if self.ep > 1:
            # init_dist_args disables the outer reduce at moe_tp==1; expert
            # parallelism re-enables exactly one group-wide sum of the
            # disjoint per-rank expert partials (see ``combine``).
            self.all_reduce = all_reduce
            self.ep_group = dist_ctx.ep_gpu_group
        else:
            self.ep_group = None

        super().__init__(tp=self.tp, tp_mode=self.tp_mode, do_renormalize=renormalize)

        hidden_dim, local_ffn_dim = update_dims(hidden_dim, ffn_dim)
        # The builder checked the global shapes; the moe-TP slice must sit on
        # the same registry alignment grid or the pack would miscompute.
        reason = turbomind_mxfp4_support_reason(hidden_dim, local_ffn_dim, dtype)
        if reason is not None:
            raise RuntimeError(f'TurboMind MXFP4 MoE rejected the TP-sliced expert geometry: {reason}')

        ep_world, ep_rank = get_ep_world_rank()
        num_local_experts = num_experts // ep_world
        first_local_expert = ep_rank * num_local_experts

        self.impl: FusedMoETmMXFP4Impl = get_backend().build_op(
            FusedMoETmMXFP4BuildSpec(
                top_k=top_k,
                # The kernel sees only the rank-local expert list; the module
                # remaps global router ids to local slots before routing.
                num_experts=num_local_experts,
                hidden_dim=hidden_dim,
                ffn_dim=local_ffn_dim,
                renormalize=renormalize,
                group_size=group_size,
                output_dtype=dtype,
                act_func=act_func,
                prefix=f'layer_{layer_idx}',
            ),
            enable_deterministic=get_build_model_context().enable_deterministic,
        )

        self.gate_up = TmMoeMXFP4Weights(num_experts, hidden_dim, 2 * local_ffn_dim, bias, 'gate_up', dtype,
                                         device)
        self.down = TmMoeMXFP4Weights(num_experts, local_ffn_dim, hidden_dim, bias, 'down', dtype, device)
        self.hidden_dim = hidden_dim
        self.ffn_dim = local_ffn_dim
        # Keep ``num_experts`` as the global router domain for compatibility.
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.first_local_expert = first_local_expert
        self.expert_list = self.gate_up.expert_list if self.ep > 1 else None
        self.release_torch_weights = True
        self.dtype = dtype
        self.device = device
        self._spill: TorchWeightSpill | None = None

    def _pack(self):
        self.gate_up.validate_complete()
        self.down.validate_complete()
        self.impl.update_weights(
            self.gate_up.blocks.data,
            self.gate_up.scales.data,
            self.gate_up.bias.data if self.gate_up.bias is not None else None,
            self.down.blocks.data,
            self.down.scales.data,
            self.down.bias.data if self.down.bias is not None else None,
        )

    def update_weights(self):
        """Pack the loaded MXFP4 experts into the TurboMind channel, then spill
        the torch weights."""
        if self.gate_up.blocks.numel() == 0:
            # The module was not part of this update bucket; its storages
            # are spilled and packing would read the emptied tensors.
            return
        self._pack()
        if not self.release_torch_weights:
            # Keep the torch copies resident (mirrors the dense route's MLA
            # exemption).
            return
        self._spill_torch_weights()

    def _spill_torch_weights(self):
        """Copy the torch expert weights to CPU and free their GPU storages."""
        if self._spill is None:
            self._spill = TorchWeightSpill()
        # Bias stays behind only inside the implementation's own device
        # copies; the registered parameters all travel to CPU.
        spill = {
            'gate_up.blocks': self.gate_up.blocks,
            'gate_up.scales': self.gate_up.scales,
            'down.blocks': self.down.blocks,
            'down.scales': self.down.scales,
        }
        if self.gate_up.bias is not None:
            spill['gate_up.bias'] = self.gate_up.bias
        if self.down.bias is not None:
            spill['down.bias'] = self.down.bias
        self._spill.spill(spill)

    def sleep_release(self):
        """Free the packed expert handles ahead of ``Module.to``.

        The packed weights are not registered tensors, so the model move cannot free them. Modules with spilled torch
        weights keep their CPU copies authoritative.
        """
        self.impl.sleep_release()
        if self.gate_up.blocks.numel() == 0:
            # Already spilled; the CPU copies stay authoritative.
            return
        if not self.release_torch_weights:
            return
        self._spill_torch_weights()

    def sleep_wakeup(self):
        """Restore spilled torch weights, re-pack, and re-spill.

        The re-spill keeps the steady state after a sleep/wakeup cycle
        identical to after a hot update: empty GPU storages, authoritative
        CPU copies, and only the packed handles (plus the impl's bias
        copies) resident.
        """
        if self._spill is not None:
            self._spill.restore()
        if self.gate_up.blocks.numel() == 0:
            raise RuntimeError('TurboMind MXFP4 MoE weights are still spilled after wakeup; '
                               'the module was never loaded.')
        self._pack()
        if not self.release_torch_weights:
            return
        self._spill_torch_weights()

    def dispatch(self, state: dict):
        """Gather eager default-TP inputs.

        Under EP (moe_tp==1) the gather is a no-op: every rank keeps the full token batch and serves only its local
        experts.
        """
        moe_type = state['moe_type']
        if moe_type != MoeType.Default:
            raise NotImplementedError(f'TurboMind MXFP4 MoE does not support MoE mode: {moe_type}')
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
        topk_idx = state['topk_idx']
        if self.ep > 1:
            # Global router ids -> local blocked slots. ``blocked_route``
            # indexes prefix sums by raw id and the packed weights are
            # (num_local_experts, ...), so the remap must happen before
            # routing. Foreign experts fall outside [0, num_local_experts)
            # and are parked in the skip block (zeroed before the reduce),
            # which is exactly the per-rank partial this route reduces.
            topk_idx = topk_idx - self.first_local_expert
        output = self.impl.forward(
            state['hidden_states'],
            state['topk_weights'],
            topk_idx,
        )
        return {'hidden_states': output, 'moe_type': state['moe_type']}

    def combine(self, state: dict):
        """Apply the existing MoE TP reduction contract exactly once."""
        moe_type = state['moe_type']
        if moe_type != MoeType.Default:
            raise NotImplementedError(f'TurboMind MXFP4 MoE does not support MoE mode: {moe_type}')
        if self.ep > 1:
            # Dispatch kept the full token batch on every rank (the
            # moe-TP gather is a no-op at moe_tp==1) and each rank reduced
            # only its local experts, so the per-rank outputs are disjoint
            # partials: one all-reduce across the EP group (== the whole
            # world in every EP topology) completes the sum. The FP32
            # expert-reduction contract lives inside the impl; this is the
            # single outer reduction. DeepEP combine is out of scope.
            if self.all_reduce:
                torch.distributed.all_reduce(state['hidden_states'], group=self.ep_group)
            return {'hidden_states': state['hidden_states'], 'moe_type': moe_type}
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
        raise NotImplementedError('TurboMind MXFP4 MoE only supports synchronous eager execution.')
