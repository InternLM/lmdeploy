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

logger = get_logger('lmdeploy')


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
    max_batches = None

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
        else:
            raise ValueError(f'dlinfer does not support {SocVersion.device_name()} device currently.')

    @classmethod
    def update_step_context(cls, step_context):
        """Update step context."""

        block_num, block_size, *_ = step_context.kv_caches[0][0].shape
        is_unpaged_prefill = False
        if not step_context.is_decoding:
            is_unpaged_prefill = all((step_context.q_seqlens == step_context.kv_seqlens).tolist())
        if step_context.block_offsets.dtype != torch.int32:
            step_context.block_offsets = step_context.block_offsets.to(torch.int32)
        if not (step_context.is_decoding or is_unpaged_prefill):
            step_context.block_offsets = step_context.block_offsets.repeat_interleave(step_context.q_seqlens, 0)
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

        def get_cpu_seqlens(is_decoding, is_unpaged_prefill):
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
                kv_seqlens_cpu = kv_seqlens_expanded = step_context.kv_seqlens.cpu()
            elif is_unpaged_prefill:
                q_seqlens_cpu = step_context.q_seqlens.cpu()
                kv_seqlens_cpu = kv_seqlens_expanded = q_seqlens_cpu
            else:
                q_seqlens_cpu = step_context.q_seqlens.cpu()
                kv_seqlens_cpu = step_context.kv_seqlens.cpu()
                # Expand kv_seqlens to per-token for paged prefill attention
                kv_seqlens_expanded = kv_seqlens_cpu.repeat_interleave(q_seqlens_cpu, 0)
            return q_seqlens_cpu, kv_seqlens_cpu, kv_seqlens_expanded

        def get_list_seqlens(is_decoding, is_unpaged_prefill, q_seqlens_cpu=None, kv_seqlens_cpu=None):
            if is_decoding:
                q_seqlens_list, kv_seqlens_list = None, None
            elif is_unpaged_prefill:
                q_seqlens_list = kv_seqlens_list = q_seqlens_cpu.tolist()
            else:
                q_seqlens_list, kv_seqlens_list = q_seqlens_cpu.tolist(), kv_seqlens_cpu.tolist()
            return q_seqlens_list, kv_seqlens_list

        def get_max_seqlens(is_decoding, is_unpaged_prefill, q_seqlens_list=None, kv_seqlens_list=None):
            if is_decoding:
                max_q_seq_len, max_kv_seq_len = 1, None
            elif is_unpaged_prefill:
                max_q_seq_len = max_kv_seq_len = max(q_seqlens_list)
            else:
                max_q_seq_len = max(q_seqlens_list)
                max_kv_seq_len = max(kv_seqlens_list)
            return max_q_seq_len, max_kv_seq_len

        def get_kv_start_indices_and_attention_mask(is_decoding, is_unpaged_prefill, q_seqlens_list, kv_seqlens_list,
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

                    if not is_unpaged_prefill:
                        single_attention_mask = torch.triu(
                            torch.ones(q_seq_len,
                                       step_context.block_offsets.shape[1] * block_size,
                                       dtype=torch.bool,
                                       device=step_context.block_offsets.device),
                            diagonal=kv_seq_len - q_seq_len + 1,
                        )
                        attention_mask.append(single_attention_mask)

                if is_unpaged_prefill:
                    attention_mask.append(
                        torch.triu(torch.ones(max_q_seq_len,
                                              max_kv_seq_len,
                                              dtype=step_context.kv_caches[0][0].dtype,
                                              device=step_context.block_offsets.device),
                                   diagonal=max_kv_seq_len - max_q_seq_len + 1))
                else:
                    attention_mask = [torch.cat(attention_mask)]

                kv_start_indices = torch.cat(kv_start_indices)

            return kv_start_indices, attention_mask

        q_seqlens_cpu, kv_seqlens_cpu, kv_seqlens_expanded = get_cpu_seqlens(step_context.is_decoding,
                                                                             is_unpaged_prefill)
        q_seqlens_list, kv_seqlens_list = get_list_seqlens(step_context.is_decoding, is_unpaged_prefill, q_seqlens_cpu,
                                                           kv_seqlens_cpu)
        max_q_seq_len, max_kv_seq_len = get_max_seqlens(step_context.is_decoding, is_unpaged_prefill, q_seqlens_list,
                                                        kv_seqlens_list)
        kv_start_indices, attention_mask = get_kv_start_indices_and_attention_mask(step_context.is_decoding,
                                                                                   is_unpaged_prefill, q_seqlens_list,
                                                                                   kv_seqlens_list, max_q_seq_len,
                                                                                   max_kv_seq_len)

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
            q_start_loc=None,
            q_seqlens=q_seqlens_cpu,
            # kv_seqlens_expanded is only expanded in paged prefill,
            # otherwise it equals kv_seqlens_cpu
            kv_seqlens=kv_seqlens_expanded,
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
        AscendOpsBackend.enable_graph = not backend_config.eager_mode
        AscendOpsBackend.max_batches = cache_config.max_batches
        from dlinfer.framework.lmdeploy_ext.cudagraph.ascend_cudagraph import AscendGraphRunner
        return AscendGraphRunner(model, model_config, cache_config, backend_config, device)

    @staticmethod
    def init():
        """Initialize Ascend backend."""
        try:
            from torch_npu.contrib import transfer_to_npu  # noqa: F401
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
