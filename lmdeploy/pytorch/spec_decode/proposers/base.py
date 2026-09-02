# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from mmengine import Registry
from torch.profiler import record_function

from lmdeploy.utils import get_logger

from ...config import CacheConfig, ModelConfig, SpecDecodeConfig
from ...engine.cache_engine import CacheEngine
from ...model_inputs import ModelInputs, step_ctx_manager
from ...models.patch import build_patched_model, update_custom_module_map
from ...strategies.base.model_agent import ExtraInputs
from ...weight_loader.model_weight_loader import load_model_weights
from ..guided_spec_helper import GuidedSpecHelper

SPEC_PROPOSERS = Registry('spec_proposers')

logger = get_logger('lmdeploy')


class ProposalMethod(str, Enum):
    """How the agent should prepare and execute draft proposal."""

    AUTOREGRESSIVE = 'autoregressive'
    DIFFUSION = 'diffusion'


@dataclass(frozen=True)
class ProposalContext:
    """Explicit runtime dependencies for a non-autoregressive proposer."""

    cache_engine: CacheEngine | None


@dataclass(frozen=True)
class ProposalWarmupCase:
    """One declarative draft warmup input shape."""

    batch_size: int
    is_decoding: bool
    max_q_seqlen: int
    target_hidden_size: int


@dataclass(frozen=True)
class ProposalWarmupPlan:
    """Ordered proposer-specific cases executed by the agent."""

    cases: tuple[ProposalWarmupCase, ...]


@torch.inference_mode()
def draft_model_forward(
    model: torch.nn.Module,
    inputs: ModelInputs,
    cache_engine: CacheEngine,
    model_config: ModelConfig | None = None,
):
    """Perform model forward."""
    stream = torch.cuda.current_stream()
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
        # forward
        ctx_mgr = model.ctx_mgr
        kv_caches = cache_engine.gpu_cache
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=model_config,
            cache_config=cache_engine.cache_config,
            kv_caches=kv_caches,
        )
        # Attach named cache views for models that declare block_cache_specs.
        context.block_caches = cache_engine.block_caches
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

    proposal_method = ProposalMethod.AUTOREGRESSIVE

    def __init__(self, specdecode_config: SpecDecodeConfig, device: torch.device = None):
        self.specdecode_config = specdecode_config
        self.model = None
        self.device = device
        self.lm_head = None
        self.num_speculative_tokens = specdecode_config.num_speculative_tokens
        self.target_model = None
        # Set by SpecModelAgent after construction
        self.guided_helper = GuidedSpecHelper()

    def build_model(self, empty_init: bool, target_model: torch.nn.Module = None, build_model_ctx=None):
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
            build_model_ctx=build_model_ctx,
        )
        logger.debug('loading weights for draft model.')
        if not empty_init:
            load_model_weights(patched_model, model_path, device=self.device)
        self.model = patched_model
        self.target_model = target_model

    async def get_outputs(self,
                    model_outputs: dict[str, torch.Tensor],
                    model_inputs: ModelInputs,
                    extra_inputs: ExtraInputs = None,
                    guided_processors: dict | None = None):
        """Get outputs."""
        raise NotImplementedError()

    async def propose(self,
                      model_inputs: ModelInputs,
                      extra_inputs: ExtraInputs,
                      sampling_inputs,
        proposal_ctx: ProposalContext | None = None):
        """Run a non-autoregressive proposal method."""
        raise NotImplementedError(f'{type(self).__name__} does not implement its proposal method.')

    def get_warmup_plan(self,
                        max_batches: int,
                        target_model_config: ModelConfig,
                        capture_batch_sizes: list[int],
                        cache_config: CacheConfig) -> ProposalWarmupPlan | None:
        """Return custom warmup shapes, or ``None`` for generic AR warmup."""
        return None

    def prepare_warmup_forward(self, inputs: ModelInputs, cache_engine: CacheEngine) -> ModelInputs | None:
        """Prepare one declarative case for forwarding by the agent."""
        return inputs

    @record_function('draft_model_forward')
    def _forward(self, model_inputs: ModelInputs, cache_engine: CacheEngine):
        """Forward."""
        return draft_model_forward(
            self.model,
            model_inputs,
            model_config=self.specdecode_config.model_config,
            cache_engine=cache_engine,
        )

    def update_inputs_decoding(self, model_inputs: ModelInputs, extra_inputs: ExtraInputs, next_input_ids: torch.Tensor,
                               target_hidden_states: torch.Tensor, model_metas: list[Any]):
        """Update to decoding inputs."""
        batch_size = model_inputs.seq_length.size(0)
        history_lengths = model_inputs.history_lengths + model_inputs.seq_length
        if extra_inputs.num_rejected_tokens is not None:
            history_lengths = history_lengths - extra_inputs.num_rejected_tokens

        mrope_pos_ids = None
        if model_inputs.mrope_pos_ids is not None:
            mrope_pos_ids = model_inputs.mrope_pos_ids[:, extra_inputs.last_token_indices] + 1

        return model_inputs.clone(
            input_ids=next_input_ids,
            seq_length=model_inputs.seq_length.new_ones(batch_size),
            history_lengths=history_lengths,
            is_decoding=True,
            num_ignored_history=model_inputs.num_ignored_history,
            max_q_seqlen=1,
            max_kv_seqlen=model_inputs.max_kv_seqlen + 1,
            sum_kv_seqlen=model_inputs.sum_kv_seqlen + model_inputs.seq_length.numel(),
            target_position_ids=history_lengths.unsqueeze(0).clone(),
            target_inputs_embeds=None,
            mrope_pos_ids=mrope_pos_ids,
            target_hidden_states=target_hidden_states,
            model_metas=model_metas,
        )

    def embed_input_ids(self, input_ids: torch.Tensor):
        """embed_input_ids."""
        if hasattr(self.model, 'get_input_embeddings'):
            input_embeds = self.model.get_input_embeddings()(input_ids)
        else:
            input_embeds = self.target_model.get_input_embeddings()(input_ids)
        return input_embeds

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
