# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import torch

from lmdeploy.pytorch import envs as _envs
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, ModelConfig
from lmdeploy.utils import get_logger

from ..op_backend import DlinferOpsBackend
from .utils import nd_to_nz_spec

logger = get_logger('lmdeploy')


class SocVersion:
    Ascend310P: str = 'Ascend310P'
    Ascend910: str = 'Ascend910'

    @classmethod
    @lru_cache(maxsize=1)
    def device_name(cls) -> str:
        try:
            import torch_npu
            return torch_npu.npu.get_device_name()
        except ImportError:
            logger.warning('Failed to import torch_npu. Please make sure torch_npu is installed correctly. ')
        except Exception as e:
            logger.warning(f'Error during Ascend get device name: {str(e)}. '
                           'Please check your Ascend environment configuration.')

    @classmethod
    def is_Ascend310P(cls) -> bool:
        return cls.device_name().startswith(cls.Ascend310P)

    @classmethod
    def is_Ascend910(cls) -> bool:
        return cls.device_name().startswith(cls.Ascend910)


class AscendKVQuantMeta:
    has_set_value: bool = False
    quant_meta: Dict = {}

    @classmethod
    def set_value(cls, device: str, dtype: torch.dtype, record_file: str, total_layers: int):
        with open(record_file, 'r') as file:
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
    enable_graph = False
    half_negative_inf = torch.finfo(torch.float16).min
    total_slots = None

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
    ) -> Tuple[int, ...]:
        if SocVersion.is_Ascend910():
            return (block_size, num_heads, head_size)
        elif SocVersion.is_Ascend310P():
            return (
                (num_heads * head_size + 15) // 16,
                block_size,
                16,
            )
        else:
            raise ValueError(f'dlinfer does not support {SocVersion.device_name()} device currently.')

    @staticmethod
    def get_v_block_shape(
        block_size: int,
        num_heads: int,
        head_size: int,
        dtype: torch.dtype,
    ) -> Tuple[int, ...]:
        if SocVersion.is_Ascend910():
            return (block_size, num_heads, head_size)
        elif SocVersion.is_Ascend310P():
            return (
                (num_heads * head_size + 15) // 16,
                block_size,
                16,
            )
        else:
            raise ValueError(f'dlinfer does not support {SocVersion.device_name()} device currently.')

    @classmethod
    @lru_cache(maxsize=1)
    def enable_aclgraph(cls) -> bool:
        if os.getenv('ASCEND_GRAPH_MODE', 'aclgraph') == 'aclgraph' and not SocVersion.is_Ascend310P():
            return True
        elif os.getenv('ASCEND_GRAPH_MODE', 'aclgraph') == 'atbgraph' or SocVersion.is_Ascend310P():
            return False
        else:
            raise ValueError(f"unsupported ASCEND_GRAPH_MODE: {os.getenv('ASCEND_GRAPH_MODE')}")

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""

        def get_total_slots():
            if cls.total_slots is None:
                cls.total_slots = torch.arange(block_num * block_size,
                                               dtype=torch.long,
                                               device=step_context.block_offsets.device)
                cls.total_slots = cls.total_slots.view(block_num, block_size)
            return cls.total_slots

        kv_start_indices, attention_mask = [], []
        if SocVersion.is_Ascend910():
            block_num, block_size, *_ = step_context.kv_caches[0][0].shape
        elif SocVersion.is_Ascend310P():
            block_num, _, block_size, _ = step_context.kv_caches[0][0].shape

        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = \
                all((step_context.q_seqlens ==
                     step_context.kv_seqlens).tolist())
        q_seqlens_list = step_context.q_seqlens.tolist()
        kv_seqlens_list = step_context.kv_seqlens.tolist()
        max_q_seq_len = max(q_seqlens_list)
        max_kv_seq_len = max(kv_seqlens_list)

        if step_context.is_decoding:
            # collect kv_start_indices without using a for-loop,
            # (fill kv-cache for just ONE token during the decoding phase)
            idx = (step_context.kv_seqlens - 1) % block_size
            block_num = (step_context.kv_seqlens - 1) // block_size
            last_block = step_context.block_offsets.gather(1, block_num.view(-1, 1)).view(-1)
            kv_start_indices = last_block * block_size + idx
        else:
            for i in range(step_context.q_start_loc.size(0)):
                q_seq_len = q_seqlens_list[i]
                kv_seq_len = kv_seqlens_list[i]

                # collect kv start indices during the prefill phase.
                history_length = kv_seq_len - q_seq_len
                total_slots = get_total_slots()
                slot_tables = total_slots[step_context.block_offsets[i]].view(-1)
                slots = slot_tables[history_length:kv_seq_len]
                kv_start_indices.append(slots)

                # collect attention mask of paged_prefill attention stage.
                if not is_unpaged_prefill:
                    single_attention_mask = torch.logical_not(
                        torch.tril(
                            torch.ones(q_seq_len,
                                       step_context.block_offsets.shape[1] * block_size,
                                       dtype=torch.bool,
                                       device=step_context.block_offsets.device),
                            diagonal=kv_seq_len - q_seq_len,
                        ))
                    attention_mask.append(single_attention_mask)

            kv_start_indices = torch.cat(kv_start_indices)

        if step_context.is_decoding:
            # prepare some params of paged_decode attention stage.
            q_start_loc_cpu, q_seqlens_cpu = None, None
        elif is_unpaged_prefill:
            # prepare some params of unpaged_prefill attention stage.
            q_start_loc_cpu, kv_seqlens_cpu = None, None
            q_seqlens_cpu = step_context.q_seqlens.cpu()
            if SocVersion.is_Ascend910():
                single_attention_mask = torch.logical_not(
                    torch.tril(
                        torch.ones(max_q_seq_len, max_kv_seq_len, dtype=torch.bool).cuda(),
                        diagonal=max_kv_seq_len - max_q_seq_len,
                    ))
                attention_mask.append(single_attention_mask)
            elif SocVersion.is_Ascend310P():
                if not cls.enable_graph:
                    for i in range(q_seqlens_cpu.size(0)):
                        single_attention_mask = torch.zeros(q_seqlens_cpu[i],
                                                            q_seqlens_cpu[i]).fill_(-float('inf')).cuda()
                        single_attention_mask = torch.triu(single_attention_mask, diagonal=1)
                        attention_mask.append(single_attention_mask)
                else:
                    # Transdata needs dtype to be float16 or int8
                    single_attention_mask = torch.triu(
                        torch.ones(max_q_seq_len, max_kv_seq_len, dtype=torch.float16).fill_(-float('inf')).cuda(),
                        diagonal=max_kv_seq_len - max_q_seq_len + 1,
                    )
                    # Convert to NZ format
                    attention_mask.append(nd_to_nz_spec(single_attention_mask))
            else:
                raise ValueError(f"dlinfer doesn't support {SocVersion.device_name()} device currently.")
        else:
            # prepare some params of paged_prefill attention stage.
            q_start_loc_cpu, q_seqlens_cpu = None, None
            attention_mask = [torch.cat([mask for mask in attention_mask])]

        if cls.enable_graph:
            kv_start_indices = kv_start_indices.view(-1).to(torch.int32)
            import torch._dynamo as dynamo
            if not is_unpaged_prefill:
                step_context.block_offsets = step_context.block_offsets.to(torch.int32)
                if not step_context.is_decoding:
                    step_context.block_offsets = step_context.block_offsets\
                        .repeat_interleave(step_context.q_seqlens, 0)
            dynamo.mark_dynamic(step_context.block_offsets, [0, 1])
            kv_seqlens = step_context.kv_seqlens.to(torch.int32)
            if not step_context.is_decoding:
                if is_unpaged_prefill:
                    if SocVersion.is_Ascend910():
                        attention_mask = [mask.half() for mask in attention_mask]
                else:
                    if SocVersion.is_Ascend910():
                        attention_mask = [
                            torch.cat([mask.half() * cls.half_negative_inf for mask in attention_mask]).unsqueeze(1)
                        ]
                    elif SocVersion.is_Ascend310P():
                        # Convert mask to NZ format.
                        attention_mask = [
                            nd_to_nz_spec(torch.cat([mask.half() * cls.half_negative_inf for mask in attention_mask]))
                        ]
                    else:
                        raise ValueError(f"dlinfer doesn't support {SocVersion.device_name()} device currently.")
                    kv_seqlens = kv_seqlens.repeat_interleave(step_context.q_seqlens, 0)
            if not is_unpaged_prefill and AscendOpsBackend.enable_aclgraph():
                kv_seqlens = kv_seqlens.cpu().tolist()
        else:
            if step_context.is_decoding:
                kv_seqlens_cpu = step_context.kv_seqlens.cpu()
            elif is_unpaged_prefill:
                pass
            else:
                kv_seqlens_cpu = step_context.kv_seqlens.repeat_interleave(step_context.q_seqlens, 0).cpu()
                block_offsets_int32 = step_context.block_offsets.to(torch.int32)
                step_context.block_offsets = block_offsets_int32\
                    .repeat_interleave(step_context.q_seqlens, 0)
            kv_seqlens = kv_seqlens_cpu

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

        attn_meta_cls = cls.get_attention_metadata_cls()
        attn_metadata = attn_meta_cls(
            step_context.is_decoding,
            step_context.block_offsets,
            q_start_loc=q_start_loc_cpu,
            q_seqlens=q_seqlens_cpu,
            kv_seqlens=kv_seqlens,
            kv_start_indices=kv_start_indices,
            block_size=block_size,
            attention_mask=attention_mask,
            is_unpaged_prefill=is_unpaged_prefill,
            max_q_seq_len=max_q_seq_len,
            max_kv_seq_len=max_kv_seq_len,
            quant_policy=step_context.kv_quant_policy,
            quant_meta=AscendKVQuantMeta.quant_meta,
        )

        step_context.attn_metadata = attn_metadata
        return step_context

    @staticmethod
    def build_graph_runner(model: torch.nn.Module, model_config: ModelConfig, cache_config: CacheConfig,
                           backend_config: BackendConfig, device: torch.device):
        """Build graph runner."""
        if AscendOpsBackend.enable_aclgraph():
            from lmdeploy.pytorch.backends.cuda.graph_runner import CUDAGraphRunner
            return CUDAGraphRunner(model, model_config, cache_config, backend_config, device)
        else:
            from .graph_runner import AscendGraphRunner
            ascend_graph_runner = AscendGraphRunner(model, model_config, cache_config, backend_config, device)
            AscendOpsBackend.enable_graph = ascend_graph_runner.enable_graph
            return ascend_graph_runner

    @staticmethod
    def init():
        """Initialize Ascend backend."""
        try:
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
            if SocVersion.is_Ascend310P():
                # NOTE: Ascend310P has a bug with InternVL vision embedding using interpolate.
                torch.npu.set_compile_mode(jit_compile=False)
        except ImportError:
            logger.warning('Failed to import torch_npu. Please make sure torch_npu is installed correctly. '
                           'Ascend initialization skipped.')
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
