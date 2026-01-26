# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional

import torch
from mmengine import Registry
from torch.profiler import record_function

from lmdeploy.utils import get_logger

from ...config import ModelConfig, SpecDecodeConfig
from ...engine.cache_engine import CacheEngine
from ...model_inputs import ModelInputs, step_ctx_manager
from ...models.patch import build_patched_model, update_custom_module_map
from ...strategies.base.model_agent import ExtraInputs
from ...weight_loader.model_weight_loader import load_model_weights

SPEC_PROPOSERS = Registry('spec_proposers')

logger = get_logger('lmdeploy')


@torch.inference_mode()
def draft_model_forward(
    model: torch.nn.Module,
    inputs: ModelInputs,
    model_config: Optional[ModelConfig] = None,
    cache_engine: Optional[CacheEngine] = None,
):
    """Perform model forward."""
    stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
        # forward
        ctx_mgr = model.ctx_mgr
        kv_caches = None if cache_engine is None else cache_engine.gpu_cache
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=model_config,
            cache_config=cache_engine.cache_config,
            kv_caches=kv_caches,
        )
        with ctx_mgr.context(context):
            model_metas = None
            model_metas = model.update_model_metas(
                past_key_values=kv_caches,
                context=context,
            )
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=kv_caches,
                context=context,
            )
            outputs = model(**input_dict)
            if not isinstance(outputs, dict):
                outputs = dict(hidden_states=outputs)
            outputs.update(dict(model_metas=model_metas))
    return outputs


class BaseSpecProposer:

    def __init__(self, specdecode_config: SpecDecodeConfig, device: torch.device = None):
        self.specdecode_config = specdecode_config
        self.model = None
        self.device = device
        self.lm_head = None
        self.num_speculative_tokens = specdecode_config.num_speculative_tokens
        self.target_model = None

    def build_model(self,
                    empty_init: bool,
                    target_model: torch.nn.Module = None,
                    model_format=None,
                    build_model_ctx=None):
        if self.specdecode_config is None:
            return
        model_path = self.specdecode_config.model
        model_config = self.specdecode_config.model_config
        custom_module_map = model_config.custom_module_map
        if custom_module_map is not None:
            update_custom_module_map(custom_module_map)
        logger.debug('build draft model')
        patched_model = build_patched_model(
            model_config,
            device=self.device,
            model_format=model_format,
            build_model_ctx=build_model_ctx,
        )
        logger.debug('loading weights for draft model.')
        if not empty_init:
            load_model_weights(patched_model, model_path, device=self.device)
        self.model = patched_model
        self.target_model = target_model

    def get_outputs(self,
                    model_outputs: Dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ExtraInputs = None):
        """Get outputs."""
        raise NotImplementedError()

    @record_function('draft_model_forward')
    def _forward(self, model_inputs: ModelInputs, cache_engine: CacheEngine = None):
        """Forward."""
        return draft_model_forward(
            self.model,
            model_inputs,
            model_config=self.specdecode_config.model_config,
            cache_engine=cache_engine,
        )

    def update_inputs_decoding(self, model_inputs: ModelInputs, extra_inputs: ExtraInputs, next_input_ids: torch.Tensor,
                               target_hidden_states: torch.Tensor, model_metas: List[Any]):
        """Update to decoding inputs."""
        model_inputs.is_decoding = True
        batch_size = model_inputs.seq_length.size(0)
        model_inputs.input_ids = next_input_ids
        model_inputs.max_q_seqlen = 1
        model_inputs.max_kv_seqlen += 1
        model_inputs.sum_kv_seqlen += model_inputs.seq_length.numel()
        model_inputs.history_lengths += model_inputs.seq_length
        if extra_inputs.num_rejected_tokens is not None:
            model_inputs.history_lengths -= extra_inputs.num_rejected_tokens
        model_inputs.seq_length = model_inputs.seq_length.new_ones(batch_size)
        model_inputs.target_position_ids = model_inputs.history_lengths.unsqueeze(0).clone()
        model_inputs.model_metas = model_metas
        model_inputs.target_hidden_states = target_hidden_states
        return model_inputs

    @record_function('draft_get_logits')
    def get_logits(self, hidden_states: torch.Tensor):
        """Get logits of model output."""
        draft_model = self.model
        if not isinstance(draft_model, torch.nn.Module):
            draft_model = draft_model.model

        if hasattr(draft_model, 'get_logits'):
            logits = draft_model.get_logits(hidden_states)
        else:
            logits = self.target_model.get_logits(hidden_states)
        return logits

    def get_target_hidden_size(self, model_config: ModelConfig):
        """Get target hidden size."""
        return model_config.hidden_size


def build_specdecode_proposer(specdecode_config: SpecDecodeConfig, device: str = 'cuda'):
    """Build spec decoding proposer."""
    method = specdecode_config.method
    if method in SPEC_PROPOSERS.module_dict:
        spec_cls = SPEC_PROPOSERS.module_dict[method]
        obj = spec_cls(specdecode_config, device=device)
        return obj
    raise ValueError(f'{method} not found in {SPEC_PROPOSERS.module_dict.keys()}')
