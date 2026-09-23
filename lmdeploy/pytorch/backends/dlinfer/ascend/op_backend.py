# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import math
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.pytorch.distributed import get_dist_manager
from lmdeploy.utils import get_logger

from ..moe import DlinferMoECommType, DlinferMoeMetadata
from ..op_backend import DlinferOpsBackend

logger = get_logger('lmdeploy')

# V4 FULL-graph: cached x_active_mask = ones(num_reqs) per captured batch size.
# On the graph path every request is active, so the mask is invariant per
# num_reqs -> reuse the address-stable tensor (no per-step torch.ones eager
# aclnn op between replays). Per-rank (not a cross-rank collective), so caching
# by local num_reqs is sound regardless of cross-DP imbalance.
_XACTIVE_MASK_CACHE: dict[int, torch.Tensor] = {}


class SocVersion:
    Ascend310P: str = 'Ascend310P'
    Ascend910: str = 'Ascend910'

    @classmethod
    @lru_cache(maxsize=1)
    def device_name(cls) -> str:
        try:
            return torch.npu.get_device_name()
        except ImportError:
            logger.warning('Failed to import torch_npu. Please make sure torch_npu is installed correctly.')
        except Exception as e:
            logger.warning(f'Error during Ascend get device name: {str(e)}. '
                           'Please check your Ascend environment configuration.')

    @classmethod
    def is_Ascend310P(cls) -> bool:
        return cls.device_name().startswith(cls.Ascend310P)

    @classmethod
    def is_Ascend910(cls) -> bool:
        return cls.device_name().startswith(cls.Ascend910)

    @classmethod
    @lru_cache(maxsize=1)
    def soc_version(cls) -> int:
        return torch.npu.get_soc_version()

    @classmethod
    def is_A2(cls) -> bool:
        return 220 <= cls.soc_version() <= 225

    @classmethod
    def is_A3(cls) -> bool:
        return 250 <= cls.soc_version() <= 255


@dataclass
class DistMeta:
    dp_size: int
    tp_size: int
    ep_size: int
    tp_rank: int
    ep_rank: int
    tp_group: torch.distributed.ProcessGroup
    ep_group: torch.distributed.ProcessGroup


class AscendKVQuantMeta:
    has_set_value: bool = False
    quant_meta: dict = {}

    @classmethod
    def set_value(cls, device: str, dtype: torch.dtype, record_file: str, total_layers: int):
        with open(record_file) as file:
            data = file.read()
        scale_offset_pairs = re.findall(r'scale:\s*([\d\.\-]+)\s*offset:\s*(-?\d+)', data)
        scale_offset_pairs = [(float(scale), float(offset)) for scale, offset in scale_offset_pairs]
        k_scales, v_scales, kv_scales = [], [], []
        k_zeros, v_zeros, kv_zeros = [], [], []
        if len(scale_offset_pairs) == total_layers:
            for scale, offset in scale_offset_pairs:
                k_scales.append(torch.tensor([scale], device=device, dtype=dtype))
                v_scales.append(torch.tensor([scale], device=device, dtype=dtype))
                kv_scales.append(torch.tensor([scale, scale], device=device, dtype=dtype))
                k_zeros.append(torch.tensor([offset], device=device, dtype=dtype))
                v_zeros.append(torch.tensor([offset], device=device, dtype=dtype))
                kv_zeros.append(torch.tensor([offset, offset], device=device, dtype=dtype))
        elif len(scale_offset_pairs) == total_layers * 2:
            for i in range(total_layers):
                scale_k, offset_k = scale_offset_pairs[2 * i]
                scale_v, offset_v = scale_offset_pairs[2 * i + 1]
                k_scales.append(torch.tensor([scale_k], device=device, dtype=dtype))
                v_scales.append(torch.tensor([scale_v], device=device, dtype=dtype))
                kv_scales.append(torch.tensor([scale_k, scale_v], device=device, dtype=dtype))
                k_zeros.append(torch.tensor([offset_k], device=device, dtype=dtype))
                v_zeros.append(torch.tensor([offset_v], device=device, dtype=dtype))
                kv_zeros.append(torch.tensor([offset_k, offset_v], device=device, dtype=dtype))
        else:
            raise ValueError(f'num of scale_offset_pairs({len(scale_offset_pairs)}) '
                             f'must match num of total_layers({total_layers})')

        cls.quant_meta.update({
            'k_scales': itertools.cycle(k_scales),
            'k_zeros': itertools.cycle(k_zeros),
            'v_scales': itertools.cycle(v_scales),
            'v_zeros': itertools.cycle(v_zeros),
            'kv_scales': itertools.cycle(kv_scales),
            'kv_zeros': itertools.cycle(kv_zeros)
        })
        cls.has_set_value = True


class AscendOpsBackend(DlinferOpsBackend):
    """Ascend layer backend."""
    enable_graph: bool = False
    total_slots = None
    max_batches = None
    dist_meta: DistMeta = None
    # DeepSeek V4: per-layer heterogeneous 6-tuple DSA caches (allocated once)
    v4_caches = None
    # one-shot guard for the V4_BUILD_OPCOUNT build-phase op counter
    _V4_BUILD_OPC_DONE = [False]
    _v4_build_n = 0

    @staticmethod
    def get_name() -> str:
        """Backend name."""
        return 'ascend'

    @staticmethod
    def get_k_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[int, ...]:
        if SocVersion.is_Ascend910():
            return (block_size, num_heads, head_size)
        else:
            raise ValueError(f'dlinfer does not support {SocVersion.device_name()} device currently.')

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> tuple[int, ...]:
        if SocVersion.is_Ascend910():
            return (block_size, num_heads, head_size)
        else:
            raise ValueError(f'dlinfer does not support {SocVersion.device_name()} device currently.')

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""

        block_num, block_size, *_ = step_context.kv_caches[0][0].shape
        is_prefill_no_cache = False
        if not step_context.is_decoding:
            is_prefill_no_cache = all((step_context.q_seqlens == step_context.kv_seqlens).tolist())
        if step_context.block_offsets.dtype != torch.int32:
            step_context.block_offsets = step_context.block_offsets.to(torch.int32)
        if step_context.kv_seqlens.dtype != torch.int32:
            step_context.kv_seqlens = step_context.kv_seqlens.to(torch.int32)
        if step_context.q_seqlens.dtype != torch.int32:
            step_context.q_seqlens = step_context.q_seqlens.to(torch.int32)

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(block_num * block_size,
                                               dtype=torch.int32,
                                               device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        def get_cpu_seqlens(is_decoding, is_prefill_no_cache):
            """Get sequence lengths on CPU.

            Returns:
                q_seqlens_cpu: query sequence lengths (per sequence).
                kv_seqlens_cpu: kv sequence lengths (per sequence), used for
                    list/max seqlens calculation.
                kv_seqlens_expanded: kv sequence lengths expanded per token via
                    repeat_interleave, used for attention metadata.
            """
            if is_decoding:
                q_seqlens_cpu = None
                # V4 FULL-graph: skip the device->CPU .cpu() host sync (the
                # per-step 507018 race surface -- the sync catches an
                # async-corrupted prior replay). V4's custom
                # sparse_attn_sharedkv reads kv length from the DEVICE
                # seqused_kv tensor (via dsa_inputs), not this CPU copy; and
                # for decode get_list_seqlens returns (None,None) (no tolist)
                # and get_kv_start_indices computes from device
                # step_context.kv_seqlens directly. So passing the DEVICE
                # kv_seqlens (-> attn_metadata.kv_seqlens) is safe for V4 and
                # also makes the model-forward in-graph gate fire on
                # forward_eager mixed-decode steps (no EZ1001). Non-V4 keeps
                # the CPU copy (standard attention reads it).
                if getattr(getattr(step_context.model_config, 'hf_config',
                                   None), 'model_type', None) == 'deepseek_v4':
                    kv_seqlens_cpu = step_context.kv_seqlens   # device, no sync
                else:
                    kv_seqlens_cpu = step_context.kv_seqlens.cpu()
            elif is_prefill_no_cache:
                q_seqlens_cpu = step_context.q_seqlens.cpu()
                kv_seqlens_cpu = q_seqlens_cpu
            else:
                q_seqlens_cpu = step_context.q_seqlens.cpu()
                kv_seqlens_cpu = step_context.kv_seqlens.cpu()
            return q_seqlens_cpu, kv_seqlens_cpu

        def get_list_seqlens(is_decoding, is_prefill_no_cache, q_seqlens_cpu=None, kv_seqlens_cpu=None):
            if is_decoding:
                q_seqlens_list, kv_seqlens_list = None, None
            elif is_prefill_no_cache:
                q_seqlens_list = kv_seqlens_list = q_seqlens_cpu.tolist()
            else:
                q_seqlens_list, kv_seqlens_list = q_seqlens_cpu.tolist(), kv_seqlens_cpu.tolist()
            return q_seqlens_list, kv_seqlens_list

        def get_max_seqlens(is_decoding, is_prefill_no_cache, q_seqlens_list=None, kv_seqlens_list=None):
            if is_decoding:
                max_q_seq_len, max_kv_seq_len = 1, None
            elif is_prefill_no_cache:
                max_q_seq_len = max_kv_seq_len = max(q_seqlens_list)
            else:
                max_q_seq_len = max(q_seqlens_list)
                max_kv_seq_len = max(kv_seqlens_list)
            return max_q_seq_len, max_kv_seq_len

        def update_q_seqlens(is_decoding, is_prefill_no_cache, q_seqlens_cpu=None):
            if is_decoding:
                batch_size = step_context.q_seqlens.size(0)
                return torch.arange(1, batch_size + 1, dtype=torch.int32)
            elif is_prefill_no_cache:
                return q_seqlens_cpu
            return q_seqlens_cpu.cumsum(dim=0)

        def get_kv_start_indices_and_attention_mask(is_decoding, is_prefill_no_cache, q_seqlens_list, kv_seqlens_list,
                                                    max_q_seq_len, max_kv_seq_len):
            kv_start_indices, attention_mask = [], []
            if is_decoding:
                idx = (step_context.kv_seqlens - 1) % block_size
                block_num = (step_context.kv_seqlens - 1) // block_size
                last_block = step_context.block_offsets.gather(1, block_num.view(-1, 1)).view(-1)
                kv_start_indices = last_block * block_size + idx
            else:
                for i in range(step_context.q_start_loc.size(0)):
                    q_seq_len = q_seqlens_list[i]
                    kv_seq_len = kv_seqlens_list[i]

                    history_length = kv_seq_len - q_seq_len
                    total_slots = get_total_slots()
                    slot_tables = total_slots[step_context.block_offsets[i]].view(-1)
                    slots = slot_tables[history_length:kv_seq_len]
                    kv_start_indices.append(slots)

                if is_prefill_no_cache:
                    attention_mask.append(
                        torch.triu(torch.ones(max_q_seq_len,
                                              max_kv_seq_len,
                                              dtype=step_context.kv_caches[0][0].dtype,
                                              device=step_context.block_offsets.device),
                                   diagonal=max_kv_seq_len - max_q_seq_len + 1))
                else:
                    attention_mask.append(
                        torch.triu(torch.ones(2048, 2048, dtype=torch.bool, device=step_context.block_offsets.device),
                                   diagonal=1))

                kv_start_indices = torch.cat(kv_start_indices)

            return kv_start_indices, attention_mask

        def get_dist_meta():
            if cls.dist_meta is not None:
                return cls.dist_meta
            dist_ctx = get_dist_manager().current_context()
            dp_size, tp_size, ep_size = dist_ctx.dist_config.dp, dist_ctx.dist_config.tp, dist_ctx.dist_config.ep
            tp_rank, ep_rank = dist_ctx.attn_tp_group.rank, dist_ctx.ep_rank
            tp_group = dist_ctx.attn_tp_group.gpu_group
            ep_group = dist_ctx.ep_gpu_group
            cls.dist_meta = DistMeta(dp_size=dp_size,
                                     tp_size=tp_size,
                                     ep_size=ep_size,
                                     tp_rank=tp_rank,
                                     ep_rank=ep_rank,
                                     tp_group=tp_group,
                                     ep_group=ep_group)
            return cls.dist_meta

        def get_tokens_info(dp_size, tp_size, ep_size, ep_group):
            if ep_size <= 1:
                return 0, 0, 0, False
            # get padded_tokens_current_rank
            is_graph = cls.enable_graph and step_context.is_decoding
            if is_graph:
                from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import get_ascend_compatible_size
                actual_tokens_current_rank = step_context.q_seqlens.shape[0]
                padded_tokens_current_rank = min(get_ascend_compatible_size(actual_tokens_current_rank),
                                                 cls.max_batches)
            else:
                actual_tokens_current_rank = step_context.q_seqlens.sum().item()
                padded_tokens_current_rank = actual_tokens_current_rank
            # get max_tokens_across_dp
            # FULL-graph: on the replay path (support_cuda_graph ==
            # global_is_decoding) the probe confirms num_reqs is uniform across
            # DP ranks (both padded to the same capture size), so the cross-DP
            # max == the local padded count. Skip the HCCL all_gather +
            # `.item()` -- the last eager aclnn compute op + host sync between
            # replays (mirrors vllm-ascend FULL graph, which bakes the count and
            # does no per-step gather). The stream-drain role of `.item()` is
            # covered by torch.npu.current_stream().synchronize() before replay
            # (ascend_cudagraph.forward). Still run the gather on eager/mixed
            # steps (global_is_decoding=False) where one DP group may be
            # prefilling with a different count and a host sync is safe.
            _uniform_decode = is_graph and step_context.global_is_decoding()
            if dp_size > 1 and not _uniform_decode:
                runtime_tokens_tensor = torch.tensor([padded_tokens_current_rank],
                                                     dtype=step_context.q_seqlens.dtype,
                                                     device=torch.npu.current_device())
                world_size = dp_size * tp_size
                runtime_tokens_buffer = torch.zeros([world_size],
                                                    dtype=step_context.q_seqlens.dtype,
                                                    device=torch.npu.current_device())
                dist.all_gather_into_tensor(runtime_tokens_buffer, runtime_tokens_tensor, ep_group)
                max_tokens_across_dp = torch.max(runtime_tokens_buffer).item()
            else:
                max_tokens_across_dp = padded_tokens_current_rank
            return actual_tokens_current_rank, padded_tokens_current_rank, max_tokens_across_dp, is_graph

        @lru_cache
        def init_mc2_token_capacity(tp_size):
            max_num_tokens = min(cls.max_batches, 512)
            num_tokens_per_tp_rank = (max_num_tokens + tp_size - 1) // tp_size
            return num_tokens_per_tp_rank * tp_size

        def select_moe_comm_type(max_tokens_across_dp, dp_size, tp_size, ep_size):
            if ep_size <= 1:
                return DlinferMoECommType.ALLGATHER
            mc2_token_capacity = init_mc2_token_capacity(tp_size)
            is_graph = cls.enable_graph and step_context.is_decoding
            if is_graph:
                max_tokens_across_dp = math.ceil(max_tokens_across_dp / tp_size) * tp_size
            if SocVersion.is_A2():
                if max_tokens_across_dp <= mc2_token_capacity and dp_size * tp_size >= 16:
                    return DlinferMoECommType.MC2
                else:
                    return DlinferMoECommType.ALLGATHER
            elif SocVersion.is_A3():
                if max_tokens_across_dp <= mc2_token_capacity:
                    return DlinferMoECommType.MC2
                else:
                    return DlinferMoECommType.ALLTOALL
            else:
                raise ValueError(f'Unsupported soc_version: {SocVersion.soc_version()}')

        def get_pad_info(actual_tokens_current_rank, padded_tokens_current_rank, max_tokens_across_dp, tp_size,
                         moe_comm_type, is_graph=False):
            x_active_mask = None
            if moe_comm_type == DlinferMoECommType.MC2:
                padded_size = math.ceil(max_tokens_across_dp / tp_size) * tp_size
                pad_size = padded_size - padded_tokens_current_rank
                if is_graph:
                    key = actual_tokens_current_rank
                    cached = _XACTIVE_MASK_CACHE.get(key)
                    if cached is None:
                        cached = torch.ones(actual_tokens_current_rank,
                                            dtype=torch.bool,
                                            device=torch.npu.current_device())
                        _XACTIVE_MASK_CACHE[key] = cached
                    x_active_mask = cached
                else:
                    x_active_mask = torch.ones(actual_tokens_current_rank,
                                               dtype=torch.bool,
                                               device=torch.npu.current_device())
            elif moe_comm_type == DlinferMoECommType.ALLTOALL:
                pad_size = tp_size - padded_tokens_current_rank
            elif moe_comm_type == DlinferMoECommType.ALLGATHER:
                pad_size = max_tokens_across_dp - padded_tokens_current_rank
            else:
                pad_size = 0
            return pad_size, x_active_mask

        @lru_cache(maxsize=1)
        def get_moe_group_name(group):
            if group is None:
                return None
            local_rank = torch.distributed.get_rank(group=group)
            backend = group._get_backend(torch.device('npu'))
            group_name = backend.get_hccl_comm_name(local_rank)
            return group_name

        q_seqlens_cpu, kv_seqlens_cpu = get_cpu_seqlens(step_context.is_decoding, is_prefill_no_cache)
        q_seqlens_list, kv_seqlens_list = get_list_seqlens(step_context.is_decoding, is_prefill_no_cache, q_seqlens_cpu,
                                                           kv_seqlens_cpu)
        max_q_seq_len, max_kv_seq_len = get_max_seqlens(step_context.is_decoding, is_prefill_no_cache, q_seqlens_list,
                                                        kv_seqlens_list)
        kv_start_indices, attention_mask = get_kv_start_indices_and_attention_mask(step_context.is_decoding,
                                                                                   is_prefill_no_cache, q_seqlens_list,
                                                                                   kv_seqlens_list, max_q_seq_len,
                                                                                   max_kv_seq_len)
        q_seqlens_cpu = update_q_seqlens(step_context.is_decoding, is_prefill_no_cache, q_seqlens_cpu)

        if not cls.enable_graph and step_context.kv_quant_policy == 8:
            record_file = os.getenv('ASCEND_QUANT_RECORD_FILE')
            assert record_file, 'please specify valid ASCEND_QUANT_RECORD_FILE'
            path = Path(record_file)
            is_path = path.is_absolute() or path.is_relative_to('/')
            exists = path.exists()
            if not (is_path and exists):
                raise ValueError('please specify valid ASCEND_QUANT_RECORD_FILE')
            if not AscendKVQuantMeta.has_set_value:
                total_layers = len(step_context.kv_caches)
                AscendKVQuantMeta.set_value(step_context.block_offsets.device, step_context.model_config.dtype,
                                            record_file, total_layers)

        cu_seqlens = None
        has_initial_state = None

        is_gated_delta = step_context.model_config.is_gated_delta
        if is_gated_delta:
            q_start_loc = step_context.q_start_loc.to(dtype=step_context.q_seqlens.dtype,
                                                      device=step_context.q_seqlens.device)
            cu_seqlens = torch.cat((q_start_loc, step_context.q_seqlens.sum().unsqueeze(0))).int()
            if not step_context.is_decoding:
                has_initial_state = ~(step_context.q_seqlens == step_context.kv_seqlens)

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            # cu_seqlens is only used in GDN and is passed down via q_start_loc.
            # Otherwise, q_start_loc is None.
            q_start_loc=cu_seqlens,
            q_seqlens=q_seqlens_cpu,
            kv_seqlens=kv_seqlens_cpu,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=attention_mask,
            is_prefill_no_cache=is_prefill_no_cache,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            quant_policy=step_context.kv_quant_policy,
            quant_meta=AscendKVQuantMeta.quant_meta,
            has_initial_state=has_initial_state,
        )
        step_context.attn_metadata = attn_metadata

        # ---- DeepSeek V4: build per-layer 6-tuple DSA caches + metadata ----
        cls._maybe_build_v4_dsa(step_context)

        cls.dist_meta = get_dist_meta()
        actual_tokens_current_rank, padded_tokens_current_rank, max_tokens_across_dp, is_graph = get_tokens_info(
            cls.dist_meta.dp_size, cls.dist_meta.tp_size, cls.dist_meta.ep_size, cls.dist_meta.ep_group)
        moe_comm_type = select_moe_comm_type(max_tokens_across_dp, cls.dist_meta.dp_size, cls.dist_meta.tp_size,
                                             cls.dist_meta.ep_size)
        pad_size, x_active_mask = get_pad_info(actual_tokens_current_rank, padded_tokens_current_rank,
                                               max_tokens_across_dp, cls.dist_meta.tp_size, moe_comm_type,
                                               is_graph=is_graph)
        moe_group_name = get_moe_group_name(cls.dist_meta.ep_group)

        moe_metadata = DlinferMoeMetadata(
            max_tokens_across_dp=max_tokens_across_dp,
            pad_size=pad_size,
            dp_size=cls.dist_meta.dp_size,
            tp_size=cls.dist_meta.tp_size,
            ep_size=cls.dist_meta.ep_size,
            tp_rank=cls.dist_meta.tp_rank,
            ep_rank=cls.dist_meta.ep_rank,
            tp_group=cls.dist_meta.tp_group,
            ep_group=cls.dist_meta.ep_group,
            moe_comm_type=moe_comm_type,
            x_active_mask=x_active_mask,
            moe_group_name=moe_group_name,
        )
        step_context.moe_metadata = moe_metadata
        return step_context

    @classmethod
    def _maybe_build_v4_dsa(cls, step_context):
        """DeepSeek V4 6-tuple DSA cache allocation + per-layer metadata.

        For V4 the engine-paged k_cache is reused as swa_kv; the remaining 5
        DSA caches (compress_kv / state / indexer_state / indexer_k /
        indexer_scale) are allocated here once (persisted on the class) with
        the per-layer heterogeneous shapes the dlinfer ops require, and the
        per-layer DSA metadata (block tables, slot mappings, compressor /
        lightning-indexer metadata) is built from the step context.  The
        resulting per-layer input dicts are stashed on
        ``step_context.v4_dsa_inputs`` and consumed by
        ``DeepseekV4Model.forward``.
        """
        model_config = step_context.model_config
        hf_config = getattr(model_config, 'hf_config', None)
        if getattr(hf_config, 'model_type', None) != 'deepseek_v4':
            return
        from .v4_dsa import allocate_v4_caches, build_v4_dsa_inputs, \
            reset_active_request
        # allocate the persistent per-layer extra caches once
        if cls.v4_caches is None:
            block_num = step_context.kv_caches[0][0].shape[0]
            device = step_context.kv_caches[0][0].device
            cache_cfg = step_context.cache_config
            # Size the recurrent state pool for the FULL engine max_batches
            # (not a DP-halved per-rank ceiling). An earlier optimization
            # assumed internal DP always splits the running batch evenly
            # (ceil(max_batches/dp) per rank) and sized the pool -- plus the
            # graph capture set (ascend_cudagraph.get_capture_batch_sizes) --
            # for that per-rank ceiling. That assumption is FALSE: the DP
            # scheduler can route more than ceil(max_batches/dp) requests to
            # one group (e.g. with max_batches=2/dp=2 the engine admits 2
            # concurrent decodes, and both can land on the same group while
            # the other runs a dummy). The per-rank ceiling (1) then can't
            # host the 2nd slot -> _V4StateAlloc.assign writes row r=1 of a
            # [1, ...] cpu table -> IndexError -> whole-serve ENGINE_STOP
            # crash. Sizing for the full max_batches makes the pool robust to
            # any DP routing, and matches the (likewise full-sized) capture
            # set so capture-warmup state_bt ranges stay <= pool. HBM cost is
            # the state pool only (compress_kv/indexer_k/indexer_scale share
            # the main paged block table, unchanged); it is acceptable on the
            # 16-card A3 box (~61 GB free/chip).
            _per_rank = cache_cfg.max_batches
            cls.v4_caches = allocate_v4_caches(
                block_num, hf_config, device, model_config.dtype,
                max_num_seqs=_per_rank,
                max_prefill_token_num=cache_cfg.max_prefill_token_num,
                session_len=getattr(cache_cfg, 'session_len', 65536))
        # a fresh prefill (no history) starts a new request -> reset state
        if not step_context.is_decoding:
            q_seqlens = step_context.q_seqlens
            kv_seqlens = step_context.kv_seqlens
            if q_seqlens.numel() and bool((q_seqlens == kv_seqlens).all()):
                reset_active_request()
        # V4_BUILD_OPCOUNT: one-shot pure-Python TorchFunctionMode op-count
        # over a single DECODE build_v4_dsa_inputs call (the build_host ~5ms
        # eager phase, NOT the captured forward). Attributes the remaining
        # build_host to torch ops (arange/copy_/to/empty/...) after the SAS
        # call-once cache eliminated the SAS op. Fires once at decode step 50
        # (steady state: graph warmup done, SAS cache warm -> 0 SAS ops), so the
        # count reflects the REAL per-step eager build cost, not first-step
        # cache-population. rank0; mirrors the V4_OPCOUNT forward counter.
        _build_opc = __import__('os').environ.get('V4_BUILD_OPCOUNT', '0') == '1'
        _is_dec = getattr(step_context, 'is_decoding', False)
        _bo_mode = None
        if _build_opc and _is_dec:
            cls._v4_build_n = cls._v4_build_n + 1
            if cls._v4_build_n == 50 and not cls._V4_BUILD_OPC_DONE[0]:
                from collections import Counter as _Ctr
                import torch.overrides as _ov

                class _Boc(_ov.TorchFunctionMode):
                    def __init__(self):
                        super().__init__()
                        self.counts = _Ctr()

                    def __torch_function__(self, func, types,
                                           args=(), kwargs=None):
                        self.counts[func.__name__] += 1
                        return func(*args, **(kwargs or {}))
                _bo_mode = _Boc()
                _bo_mode.__enter__()
        step_context.v4_dsa_inputs = build_v4_dsa_inputs(
            step_context, cls.v4_caches, model_config)
        if _bo_mode is not None:
            _bo_mode.__exit__(None, None, None)
            cls._V4_BUILD_OPC_DONE[0] = True
            _tot = sum(_bo_mode.counts.values())
            _top = sorted(_bo_mode.counts.items(), key=lambda kv: kv[1],
                          reverse=True)
            print(f'[V4-BOC] build_v4_dsa_inputs total_torch_ops={_tot}',
                  flush=True)
            for _n, _c in _top[:30]:
                print(f'[V4-BOC] {_c:5d}  {_n}', flush=True)

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        AscendOpsBackend.enable_graph = not backend_config.eager_mode
        AscendOpsBackend.max_batches = cache_config.max_batches
        from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import AscendGraphRunner
        return AscendGraphRunner(model, model_config, cache_config, backend_config, device)

    @staticmethod
    def init():
        """Initialize Ascend backend.

        Note: triton device properties initialization is only required for models that use
        linear attention (e.g. Qwen3.5 35B). If triton-ascend is not installed, a warning
        is emitted but non-linear-attention models are unaffected.
        """

        try:
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
        except ImportError:
            logger.warning('Failed to import torch_npu. Please make sure torch_npu is installed correctly. '
                           'Ascend initialization skipped.')
        except Exception as e:
            logger.warning(f'Error during Ascend initialization: {str(e)}. '
                           'Please check your Ascend environment configuration.')

        try:
            from dlinfer.vendor.ascend.triton_ops.triton_utils import init_device_properties_triton
            init_device_properties_triton()
        except ImportError:
            logger.warning('triton-ascend is not installed. Only linear attention models (e.g. Qwen3.5 35B) '
                           'require triton-ascend. Please install it with: pip install triton-ascend==3.2.0')
        except Exception as e:
            logger.warning(f'Error during Ascend initialization: {str(e)}. '
                           'Please check your Ascend environment configuration.')

    @staticmethod
    def ccl_backend():
        return 'hccl'

    @staticmethod
    def device_count():
        """Get num available devices."""
        return torch.npu.device_count()

    @staticmethod
    def support_ray():
        """Support ray."""
        if not _envs.ascend_set_rt_visable_devices_by_ray:
            os.environ['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] = '1'
        return True
