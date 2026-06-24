# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import contextmanager
from dataclasses import replace

import torch

from lmdeploy.pytorch.backends import get_backend
from lmdeploy.pytorch.config import BackendConfig, CacheConfig, MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine
from lmdeploy.pytorch.model_inputs import ModelInputs, step_ctx_manager
from lmdeploy.pytorch.models.patch import BuildModelContext, build_patched_model, update_custom_module_map
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_model_weights


@torch.inference_mode()
def memory_model_forward(
    model: torch.nn.Module,
    inputs: ModelInputs,
    model_config: ModelConfig,
    cache_engine: CacheEngine,
    state_cache_engine: StateCacheEngine = None,
):
    """Perform memory model forward."""
    state_caches = None
    if state_cache_engine is not None:
        state_caches = state_cache_engine.state_caches

    with step_ctx_manager(model.ctx_mgr):
        ctx_mgr = model.ctx_mgr
        context = ctx_mgr.build_context(
            inputs=inputs,
            model_config=model_config,
            cache_config=cache_engine.cache_config,
            kv_caches=cache_engine.gpu_cache,
            state_caches=state_caches,
            kv_quant_policy=cache_engine.cache_config.quant_policy,
        )

        with ctx_mgr.context(context):
            model_metas = model.update_model_metas(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            output = model(**input_dict)
            if not isinstance(output, dict):
                output = dict(hidden_states=output)
            if getattr(context, 'is_model_meta_updated', False):
                model_metas = context.model_metas
            output['model_metas'] = model_metas
            output['seq_length'] = context.q_seqlens[:len(inputs.seq_length)]
            output['position_ids'] = context.position_ids
            return output


class BaseMemDecodeAgent:
    """Disabled MemDecode memory model agent."""

    def __init__(
        self,
        memdecode_config: MemDecodeConfig | None,
        backend_config: BackendConfig,
        dist_ctx: DistContext,
        device: str = 'cuda',
    ):
        self.memdecode_config = memdecode_config
        self.backend_config = backend_config
        self.dist_ctx = dist_ctx
        self.device = device
        self.model_config = memdecode_config.memory_model_config if memdecode_config is not None else None
        self.cache_config = None
        self.cache_engine = None
        self.state_cache_engine = None
        self.model = None

    def is_enabled(self):
        """Return whether MemDecode is enabled."""
        return False

    @contextmanager
    def memory_context(self):
        """Memory model dist context."""
        yield

    def set_cache_config(self, cache_config: CacheConfig):
        """Set cache config."""
        pass

    def build_model(self, empty_init: bool = False):
        """Build memory model."""
        pass

    def build_graph_runner(self):
        """Build graph runner."""
        pass

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        pass

    async def async_forward(self, inputs: ModelInputs):
        """Run memory model forward."""
        return None

    def get_logits(self, hidden_states: torch.Tensor):
        """Get memory model logits."""
        raise RuntimeError('MemDecode is disabled.')

    def reset_graph_runner(self):
        """Reset graph runner."""
        pass

    def release(self):
        """Release memory model resources."""
        pass

    def get_model(self):
        """Get memory model."""
        return None


class MemDecodeAgent(BaseMemDecodeAgent):
    """Enabled MemDecode memory model agent."""

    def is_enabled(self):
        """Return whether MemDecode is enabled."""
        return True

    @contextmanager
    def memory_context(self):
        """Memory model dist context."""
        with get_dist_manager().context(self.dist_ctx):
            yield

    def set_cache_config(self, cache_config: CacheConfig):
        """Set cache config."""
        self.cache_config = cache_config

    def build_model(self, empty_init: bool = False):
        """Build memory model."""
        with self.memory_context():
            custom_module_map = self.model_config.custom_module_map
            if custom_module_map is not None:
                update_custom_module_map(custom_module_map)

            build_model_ctx = BuildModelContext(
                quant_config=self.model_config.quant_config,
                fp32_lm_head=self.model_config.fp32_lm_head,
                tie_word_embeddings=self.model_config.tie_word_embeddings,
            )
            self.model = build_patched_model(self.model_config, device=self.device, build_model_ctx=build_model_ctx)
            if not empty_init:
                load_model_weights(self.model, self.memdecode_config.memory_model_path, device=self.device)

    def build_graph_runner(self):
        """Build graph runner."""
        with self.memory_context():
            backend = get_backend()
            self.model = backend.build_graph_runner(self.model,
                                                    model_config=self.model_config,
                                                    cache_config=self.cache_config,
                                                    backend_config=self.backend_config,
                                                    device=self.device)

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        """Build cache engine."""
        with self.memory_context():
            dist_ctx = get_dist_manager().current_context()
            dist_config = dist_ctx.dist_config
            tp = dist_config.attn_tp
            self.cache_engine = CacheEngine(
                self.cache_config,
                self.model_config,
                rank=self.dist_ctx.rank,
                tp_rank=dist_ctx.attn_tp_group.rank,
                world_size=tp,
                cache_stream=cache_stream,
            )
            if len(self.model_config.states_shapes) > 0:
                state_cache_config = replace(
                    self.cache_config,
                    states_shapes=list(self.model_config.states_shapes),
                )
                self.state_cache_engine = StateCacheEngine(state_cache_config)
            else:
                self.state_cache_engine = None

    async def async_forward(self, inputs: ModelInputs):
        """Run memory model forward."""
        output = memory_model_forward(
            self.model,
            inputs,
            self.model_config,
            self.cache_engine,
            state_cache_engine=self.state_cache_engine,
        )
        await asyncio.sleep(0)
        return output

    def get_logits(self, hidden_states: torch.Tensor):
        """Get memory model logits."""
        return self.model.get_logits(hidden_states)

    def reset_graph_runner(self):
        """Reset graph runner."""
        if self.model is None:
            return
        with self.memory_context():
            if hasattr(self.model, 'reset'):
                self.model.reset()

    def release(self):
        """Release memory model resources."""
        self.model = None
        self.cache_engine = None
        self.state_cache_engine = None

    def get_model(self):
        """Get memory model."""
        if self.model is None:
            return None
        if hasattr(self.model, 'get_model'):
            return self.model.get_model()
        return self.model


def build_memdecode_agent(
    memdecode_config: MemDecodeConfig | None,
    backend_config: BackendConfig,
    dist_ctx: DistContext,
    device: str = 'cuda',
):
    """Build MemDecode memory model agent."""
    if memdecode_config is None:
        return BaseMemDecodeAgent(memdecode_config, backend_config, dist_ctx, device=device)
    return MemDecodeAgent(memdecode_config, backend_config, dist_ctx, device=device)
