# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch

from ..config import CacheConfig, ModelConfig
from .layout_convert import continuous_tensor, page_cache


def make_model_inputs(input_ids: torch.Tensor,
                      block_offsets: torch.Tensor,
                      seq_length: torch.Tensor = None,
                      history_length: torch.Tensor = None):
    """make model inputs."""
    from lmdeploy.pytorch.engine.model_agent import ModelInputs
    batch_size = input_ids.size(0)
    max_seq_len = input_ids.size(1)
    if seq_length is None:
        max_seq_len = input_ids.size(1)
        seq_length = torch.full((batch_size, ), max_seq_len)
    input_ids = continuous_tensor(input_ids, seq_length)
    if history_length is None:
        history_length = torch.zeros_like(seq_length)
    else:
        assert len(history_length) == len(seq_length)
    is_decoding = max_seq_len == 1

    num_ignored_history = torch.zeros_like(seq_length)
    return ModelInputs(input_ids=input_ids,
                       seq_length=seq_length,
                       history_lengths=history_length,
                       block_offsets=block_offsets,
                       is_decoding=is_decoding,
                       num_ignored_history=num_ignored_history)


def make_step_context(
    input_ids: torch.Tensor,
    seq_length: torch.Tensor = None,
    history_length: torch.Tensor = None,
    past_key_values: List[Tuple] = None,
    model_config: ModelConfig = None,
    cache_config: CacheConfig = None,
    block_size: int = 64,
    world_size: int = 1,
):
    """make step context."""
    device = input_ids.device
    from torch.nn.utils.rnn import pad_sequence

    from lmdeploy.pytorch.engine.cache_engine import CacheEngine
    from lmdeploy.pytorch.model_inputs import StepContext

    if model_config is None:
        model_config = ModelConfig(hidden_size=4096,
                                   num_layers=1,
                                   num_attention_heads=8,
                                   num_key_value_heads=8,
                                   bos_token_id=0,
                                   eos_token_id=0,
                                   head_dim=128)

    batch_size = input_ids.size(0)
    max_seq_len = input_ids.size(1)
    if seq_length is None:
        max_seq_len = input_ids.size(1)
        seq_length = torch.full((batch_size, ), max_seq_len)

    if history_length is None:
        history_length = torch.zeros_like(seq_length)
    else:
        assert len(history_length) == len(seq_length)
    if past_key_values is not None:
        assert len(past_key_values) == model_config.num_layers

    if cache_config is None:
        total_length = seq_length + history_length
        num_blocks_per_seq = (total_length + block_size - 1) // block_size
        num_blocks = sum(num_blocks_per_seq).item()
        cache_config = CacheConfig(
            max_batches=128,
            block_size=block_size,
            num_cpu_blocks=0,
            num_gpu_blocks=num_blocks,
        )

    def __create_kv_caches():
        """create kv caches."""
        total_length = seq_length + history_length
        num_blocks_per_seq = (total_length + block_size - 1) // block_size
        num_blocks = sum(num_blocks_per_seq)

        cache_engine = CacheEngine(
            cache_config=cache_config,
            model_config=model_config,
            world_size=world_size,
        )
        kv_caches = cache_engine.gpu_cache

        block_offsets_1d = torch.arange(0, num_blocks)
        block_end_loc = num_blocks_per_seq.cumsum(0)
        block_start_loc = block_end_loc - num_blocks_per_seq
        block_offsets = [
            block_offsets_1d[sloc:eloc]
            for sloc, eloc in zip(block_start_loc, block_end_loc)
        ]
        num_blocks_offs = torch.tensor([len(boff) for boff in block_offsets])
        block_offsets = pad_sequence(block_offsets, batch_first=True)

        return kv_caches, block_offsets, num_blocks_offs

    def __fill_kv_caches(kv_caches, past_key_values, block_offsets):
        """fill kv caches."""
        if past_key_values is None:
            return

        if all(hlen == 0 for hlen in history_length):
            return

        num_layers = len(past_key_values)
        for layer_idx in range(num_layers):
            k_cache, v_cache = kv_caches[layer_idx]
            past_k, past_v = past_key_values[layer_idx]
            page_cache(k_cache, past_k, history_length, block_offsets)
            page_cache(v_cache, past_v, history_length, block_offsets)

    kv_caches, block_offsets, _ = __create_kv_caches()
    __fill_kv_caches(kv_caches, past_key_values, block_offsets)

    model_inputs = make_model_inputs(input_ids,
                                     block_offsets=block_offsets,
                                     seq_length=seq_length,
                                     history_length=history_length)
    model_inputs = model_inputs.to_device(device)

    return StepContext.new(
        inputs=model_inputs,
        world_size=world_size,
        kv_caches=kv_caches,
    )


class ExtractorFound(Exception):
    pass


class ModuleIOExtractor:
    """Extract input and output of target sub module."""

    def __init__(self, model: torch.nn.Module, target_module: torch.nn.Module):

        def __check_target_exist():
            for mod in model.modules():
                if mod == target_module:
                    return True
            return False

        if not __check_target_exist():
            raise RuntimeError(f'{type(target_module)} is not a sub module'
                               f' of {type(model)}')

        self._model = model
        self._target_module = target_module

    @torch.inference_mode()
    def extract(self, *args, **kwargs):
        """extract."""
        target_args = None
        target_kwargs = None
        target_output = None

        def __forward_hook(module, args, kwargs, output):
            """hook."""
            nonlocal target_args, target_kwargs, target_output
            target_args = args
            target_kwargs = kwargs
            target_output = output
            raise ExtractorFound()

        handle = self._target_module.register_forward_hook(__forward_hook,
                                                           with_kwargs=True)

        try:
            self._model(*args, **kwargs)
        except ExtractorFound:
            pass
        finally:
            handle.remove()

        return target_args, target_kwargs, target_output
