# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Tuple

import torch

from .layout_convert import continuous_tensor, page_cache


def make_model_inputs(input_ids: torch.Tensor,
                      block_offsets: torch.Tensor,
                      seq_length: torch.Tensor = None,
                      history_length: List[int] = None):
    """make model inputs."""
    from lmdeploy.pytorch.engine.model_agent import ModelInputs
    batch_size = input_ids.size(0)
    max_seq_len = input_ids.size(1)
    if seq_length is None:
        max_seq_len = input_ids.size(1)
        seq_length = torch.full((batch_size, ), max_seq_len)
    input_ids = continuous_tensor(input_ids, seq_length)
    if history_length is None:
        history_length = [0] * batch_size
    else:
        assert len(history_length) == len(seq_length)
    is_decoding = input_ids.size(0) == batch_size
    q_start_loc = seq_length.cumsum(0) - seq_length
    mask_range = torch.arange(max_seq_len)[None, :]
    attention_mask = (mask_range < seq_length[:, None]).long()
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids += position_ids.new_tensor(history_length).unsqueeze(-1)

    if isinstance(history_length, torch.Tensor):
        history_length = history_length.tolist()

    return ModelInputs(input_ids=input_ids,
                       seq_length=seq_length,
                       attention_mask=attention_mask,
                       block_offsets=block_offsets,
                       position_ids=position_ids,
                       q_start_loc=q_start_loc,
                       history_lengths=history_length,
                       is_decoding=is_decoding)


def make_step_context(
    input_ids: torch.Tensor,
    seq_length: torch.Tensor = None,
    history_length: List[int] = None,
    past_key_values: List[Tuple] = None,
    world_size: int = 1,
    device: str = 'cuda',
    block_size: int = 64,
    num_key_value_heads: int = 32,
    head_size: int = 128,
    kv_cache_dtype: torch.dtype = torch.float16,
    json_config: Any = None,
):
    """make step context."""
    from torch.nn.utils.rnn import pad_sequence

    from lmdeploy.pytorch.engine.model_agent import StepContext

    batch_size = input_ids.size(0)
    max_seq_len = input_ids.size(1)
    if seq_length is None:
        max_seq_len = input_ids.size(1)
        seq_length = torch.full((batch_size, ), max_seq_len)

    if history_length is None:
        history_length = [0] * batch_size
    else:
        assert len(history_length) == len(seq_length)
    history_length = torch.tensor(history_length)

    def __create_kv_caches(past_key_values):
        """create kv caches."""
        total_length = seq_length + history_length
        num_blocks_per_seq = (total_length + block_size - 1) // block_size
        num_blocks = sum(num_blocks_per_seq)
        num_caches = 1 if past_key_values is None else len(past_key_values)
        cache_shape = [num_blocks, block_size, num_key_value_heads, head_size]

        block_offsets_1d = torch.arange(0, num_blocks)
        block_end_loc = num_blocks_per_seq.cumsum(0)
        block_start_loc = block_end_loc - num_blocks_per_seq
        block_offsets = [
            block_offsets_1d[sloc:eloc]
            for sloc, eloc in zip(block_start_loc, block_end_loc)
        ]
        block_offsets = pad_sequence(block_offsets, batch_first=True)

        kv_caches = []
        for _ in range(num_caches):
            k_cache = torch.empty(cache_shape,
                                  dtype=kv_cache_dtype,
                                  device=device)
            v_cache = torch.empty_like(k_cache)
            kv_caches.append((k_cache, v_cache))
        return kv_caches, block_offsets

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

    kv_caches, block_offsets = __create_kv_caches(past_key_values)
    __fill_kv_caches(kv_caches, past_key_values, block_offsets)

    history_length = history_length.tolist()
    model_inputs = make_model_inputs(input_ids,
                                     block_offsets=block_offsets,
                                     seq_length=seq_length,
                                     history_length=history_length)

    model_inputs = model_inputs.to_device(device)

    return StepContext.new(
        inputs=model_inputs,
        world_size=world_size,
        device=device,
        json_config=json_config,
        kv_caches=kv_caches,
    )


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

        handle = self._target_module.register_forward_hook(__forward_hook,
                                                           with_kwargs=True)

        self._model(*args, **kwargs)
        handle.remove()

        return target_args, target_kwargs, target_output
