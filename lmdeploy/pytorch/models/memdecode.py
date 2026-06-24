# Copyright (c) OpenMMLab. All rights reserved.

import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from lmdeploy.pytorch.model_inputs import StepContext, get_step_ctx_manager
from lmdeploy.pytorch.models.patch import build_model_from_hf_config
from lmdeploy.pytorch.models.utils.cudagraph import CudaGraphMixin
from lmdeploy.pytorch.utils import get_logger
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_model_weights

logger = get_logger('lmdeploy')


class RouterNetwork(nn.Module):
    """Router network to produce per-token log mixing weights."""

    def __init__(self,
                 base_hidden_size: int,
                 mem_hidden_size: int,
                 num_layers: int = 2,
                 input_mode: str = 'both',
                 use_scalars: bool = True,
                 scalar_proj_dim: int = 64,
                 hidden_dim: int = 128,
                 dropout: float = 0.2):
        super().__init__()
        self.input_mode = input_mode
        self.use_scalars = use_scalars
        self.scalar_proj_dim = scalar_proj_dim

        input_dim = 0
        if input_mode == 'both':
            input_dim += base_hidden_size + mem_hidden_size
        elif input_mode in {'memory_only', 'mem_hidden_both_scalars'}:
            input_dim += mem_hidden_size
        else:
            raise ValueError(f'Unknown input_mode: {input_mode}')

        self.num_scalars = 4 if input_mode in {'both', 'mem_hidden_both_scalars'} else 2

        if use_scalars:
            self.scalar_projectors = nn.ModuleList([
                nn.Sequential(nn.Linear(1, scalar_proj_dim), nn.ReLU()) for _ in range(self.num_scalars)
            ])
            input_dim += self.num_scalars * scalar_proj_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(max(num_layers - 2, 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)

    @staticmethod
    def _get_metrics(logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        confidence, _ = torch.max(probs, dim=-1, keepdim=True)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1, keepdim=True)
        return confidence, entropy

    def forward(self, base_hs=None, mem_hs=None, base_logits=None, mem_logits=None):
        features = []
        if self.input_mode == 'both':
            if base_hs is None or mem_hs is None:
                raise ValueError('router in `both` mode requires both base_hs and mem_hs.')
            features.append(base_hs)
            features.append(mem_hs)
        elif self.input_mode in {'memory_only', 'mem_hidden_both_scalars'}:
            if mem_hs is None:
                raise ValueError('router in memory-only mode requires mem_hs.')
            features.append(mem_hs)

        if self.use_scalars:
            scalars = []
            if self.input_mode in {'both', 'mem_hidden_both_scalars'}:
                b_conf, b_ent = self._get_metrics(base_logits)
                m_conf, m_ent = self._get_metrics(mem_logits)
                scalars.extend([b_conf, b_ent, m_conf, m_ent])
            else:
                m_conf, m_ent = self._get_metrics(mem_logits)
                scalars.extend([m_conf, m_ent])

            for idx, scalar in enumerate(scalars):
                features.append(self.scalar_projectors[idx](scalar))

        router_input = torch.cat(features, dim=-1)
        return F.log_softmax(self.mlp(router_input), dim=-1)


DEFAULT_ROUTER_CONFIG = {
    'num_layers': 2,
    'input_mode': 'both',
    'use_scalars': True,
    'scalar_proj_dim': 64,
    'hidden_dim': 128,
}


def get_hidden_size(config) -> int:
    """Resolve hidden size from nested HF configs."""
    if hasattr(config, 'hidden_size') and config.hidden_size is not None:
        return int(config.hidden_size)
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
        return int(config.text_config.hidden_size)
    if hasattr(config, 'llm_config') and hasattr(config.llm_config, 'hidden_size'):
        return int(config.llm_config.hidden_size)
    raise ValueError('Cannot resolve hidden_size from config.')


def get_vocab_size(config) -> int:
    """Resolve vocab size from nested HF configs."""
    if hasattr(config, 'vocab_size') and config.vocab_size is not None:
        return int(config.vocab_size)
    if hasattr(config, 'text_config') and hasattr(config.text_config, 'vocab_size'):
        return int(config.text_config.vocab_size)
    if hasattr(config, 'llm_config') and hasattr(config.llm_config, 'vocab_size'):
        return int(config.llm_config.vocab_size)
    raise ValueError('Cannot resolve vocab_size from config.')


class MemDecodeForCausalLM(nn.Module, CudaGraphMixin):
    """Dual-model wrapper that fuses base and memory logits."""

    def __init__(self, config, ctx_mgr, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__()
        self.config = config
        self.ctx_mgr = ctx_mgr
        self.dtype = dtype
        self.base_model_path = getattr(config, 'base_model_path', None)
        self.memory_model_path = getattr(config, 'memory_model_path', None)
        self.lambda_value = float(getattr(config, 'lambda_value', 1.0))
        if not 0.0 <= self.lambda_value <= 1.0:
            raise ValueError(f'lambda_value must be in [0, 1], got {self.lambda_value}')
        self.adaptive_router = bool(getattr(config, 'adaptive_router', False))
        threshold = getattr(config, 'lambda_base_only_threshold', -1.0)
        self.lambda_base_only_threshold = None if threshold is None else float(threshold)
        if self.lambda_base_only_threshold is not None and self.lambda_base_only_threshold < 0:
            self.lambda_base_only_threshold = None
        self.router_path = getattr(config, 'router_path', None)
        trust_remote_code = getattr(config, 'trust_remote_code', False)

        if self.base_model_path is None:
            raise ValueError('`base_model_path` is required for MemDecodeForCausalLM')

        logger.warning(
            'Initializing MemDecodeForCausalLM: base=%s memory=%s adaptive_router=%s lambda_value=%s',
            self.base_model_path,
            self.memory_model_path,
            self.adaptive_router,
            self.lambda_value,
        )

        base_hf_config = self._load_hf_config(self.base_model_path, trust_remote_code=trust_remote_code)
        base_dtype = self._resolve_build_dtype(base_hf_config, dtype)

        # build true base model (do not recursively build MemDecodeForCausalLM)
        self.base_model = build_model_from_hf_config(base_hf_config,
                                                    dtype=base_dtype,
                                                    device=device,
                                                    ctx_mgr=self.ctx_mgr,
                                                    build_model_ctx=self.ctx_mgr.build_ctx)
        self.base_vocab_size = get_vocab_size(self.base_model.config)
        self.memory_model = None
        self.memory_model_config = None
        if self.memory_model_path is not None:
            memory_hf_config = self._load_hf_config(self.memory_model_path, trust_remote_code=trust_remote_code)
            memory_dtype = self._resolve_build_dtype(memory_hf_config, base_dtype)
            memory_build_ctx = self.ctx_mgr.build_ctx
            if memory_build_ctx is not None:
                memory_build_ctx = replace(memory_build_ctx, quant_config=None)
            self.memory_model = build_model_from_hf_config(memory_hf_config,
                                                          dtype=memory_dtype,
                                                          device=device,
                                                          ctx_mgr=self.ctx_mgr,
                                                          build_model_ctx=memory_build_ctx)

        self.router = None
        self.router_config = None
        if self.adaptive_router:
            if self.router_path is None:
                raise ValueError('`router_path` is required when adaptive_router is enabled')
            if self.memory_model is None:
                raise RuntimeError('adaptive router mode requires memory_model_path.')
            self.router_config = self._load_router_config(self.router_path)
            router_config = self.router_config or DEFAULT_ROUTER_CONFIG
            self.router = RouterNetwork(
                base_hidden_size=get_hidden_size(self.base_model.config),
                mem_hidden_size=get_hidden_size(self.memory_model.config),
                **router_config,
            )
            self._load_router(self.router_path)
            self._to_router_device()

    @staticmethod
    def _load_hf_config(model_path: str, trust_remote_code: bool = False):
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if getattr(hf_config, 'model_type', None) in ['phi3']:
            hf_config = AutoConfig.from_pretrained(model_path)
        return hf_config

    @staticmethod
    def _get_hf_dtype(config):
        hf_dtype = getattr(config, 'torch_dtype', None)
        if hf_dtype is None:
            return None
        if isinstance(hf_dtype, str):
            return getattr(torch, hf_dtype)
        return hf_dtype

    @classmethod
    def _resolve_build_dtype(cls, hf_config, fallback: torch.dtype | None):
        dtype = cls._get_hf_dtype(hf_config)
        if dtype is None and hasattr(hf_config, 'text_config'):
            dtype = cls._get_hf_dtype(hf_config.text_config)
        if dtype is None:
            dtype = fallback
        return dtype

    @staticmethod
    def _unwrap_hidden_states(model_output: torch.Tensor | dict) -> torch.Tensor:
        if isinstance(model_output, dict):
            return model_output['hidden_states']
        return model_output

    @staticmethod
    def _align_vocab_to_base(logits: torch.Tensor, base_vocab_size: int) -> torch.Tensor:
        vocab_size = logits.size(-1)
        if vocab_size == base_vocab_size:
            return logits
        if vocab_size > base_vocab_size:
            return logits[..., :base_vocab_size]
        pad = logits.new_full((*logits.shape[:-1], base_vocab_size - vocab_size), float('-inf'))
        return torch.cat([logits, pad], dim=-1)

    def _align_fusion_logits(self, base_logits: torch.Tensor, mem_logits: torch.Tensor):
        base_vocab_size = base_logits.size(-1)
        if base_vocab_size != self.base_vocab_size:
            logger.warning(
                f'Base logits vocab ({base_vocab_size}) differs from config '
                f'base_vocab_size ({self.base_vocab_size}); using logits size for fusion.',
            )
        base_logits = self._align_vocab_to_base(base_logits, base_vocab_size)
        mem_logits = self._align_vocab_to_base(mem_logits, base_vocab_size)
        return base_logits, mem_logits

    def _get_fixed_log_mixing_weights(self) -> tuple[float, float]:
        lam = self.lambda_value
        if lam <= 0.0:
            return float('-inf'), 0.0
        if lam >= 1.0:
            return 0.0, float('-inf')
        return math.log(lam), math.log1p(-lam)

    def _log_fusion_debug(
        self,
        base_logits: torch.Tensor,
        mem_logits: torch.Tensor,
        fused_logits: torch.Tensor,
    ) -> None:
        """Log fusion tensor shapes.

        Visible on prefill / cudagraph capture only.
        """
        ctx = get_step_ctx_manager().current_context()
        is_decoding = ctx.global_is_decoding() if ctx is not None else None
        logger.warning(
            '[memdecode] is_decoding=%s adaptive_router=%s lambda_value=%s '
            'base_logits=%s mem_logits=%s fused_logits=%s',
            is_decoding,
            self.adaptive_router,
            self.lambda_value,
            tuple(base_logits.shape),
            tuple(mem_logits.shape),
            tuple(fused_logits.shape),
        )

    def get_logits(self, hidden_states: torch.Tensor):
        """MemDecode already outputs logits in hidden_states field."""
        return hidden_states

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_input_processor(self):
        return self.base_model.get_input_processor()

    def prepare_inputs_for_generation(
        self,
        past_key_values: list[list[torch.Tensor]],
        inputs_embeds: torch.Tensor | None = None,
        context: StepContext | None = None,
        **kwargs,
    ):
        if context is None:
            raise ValueError('context is required for MemDecodeForCausalLM.prepare_inputs_for_generation')

        memory_past_key_values = context.memory_kv_caches

        base_inputs = self.base_model.prepare_inputs_for_generation(
            past_key_values,
            inputs_embeds=inputs_embeds,
            context=context,
        )
        if memory_past_key_values is None:
            base_inputs['memory_past_key_values'] = None
            return base_inputs

        memory_state_caches = context.memory_state_caches
        if memory_state_caches is None:
            memory_state_caches = context.state_caches
        memory_context = replace(context, state_caches=memory_state_caches)

        memory_inputs = self.memory_model.prepare_inputs_for_generation(
            memory_past_key_values,
            inputs_embeds=inputs_embeds,
            context=memory_context,
        )
        base_inputs['memory_past_key_values'] = memory_inputs['past_key_values']
        # keep any embedding patching done by either side
        if memory_inputs.get('inputs_embeds') is not None:
            base_inputs['inputs_embeds'] = memory_inputs['inputs_embeds']
        return base_inputs

    def update_model_metas(self,
                          past_key_values: list[list[torch.Tensor]],
                          inputs_embeds: torch.Tensor = None,
                          context: StepContext = None):
        return None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: list[list[torch.Tensor]],
        memory_past_key_values: list[list[torch.Tensor]] = None,
        attn_metadata: Any = None,
        inputs_embeds: torch.Tensor = None,
        **kwargs,
    ):
        base_output = self.base_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        base_hidden_states = self._unwrap_hidden_states(base_output)
        base_logits = self.base_model.get_logits(base_hidden_states)

        if self.memory_model is None:
            return {
                'hidden_states': base_logits,
            }

        mem_output = self.memory_model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=memory_past_key_values,
            attn_metadata=attn_metadata,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        mem_hidden_states = self._unwrap_hidden_states(mem_output)
        mem_logits = self.memory_model.get_logits(mem_hidden_states)

        if self.adaptive_router:
            fused_logits, all_routed_experts = self._adaptive_fuse(
                base_logits,
                mem_logits,
                base_hidden_states,
                mem_hidden_states,
            )
        else:
            fused_logits, all_routed_experts = self._fixed_fuse(base_logits, mem_logits)

        self._log_fusion_debug(base_logits, mem_logits, fused_logits)

        output = {
            'hidden_states': fused_logits,
            'all_routed_experts': all_routed_experts,
        }
        return output

    def _fixed_fuse(self, base_logits: torch.Tensor, mem_logits: torch.Tensor):
        base_logits, mem_logits = self._align_fusion_logits(base_logits, mem_logits)
        log_lambda, log_one_minus_lambda = self._get_fixed_log_mixing_weights()

        logp_joint = torch.logaddexp(
            F.log_softmax(base_logits, dim=-1) + log_one_minus_lambda,
            F.log_softmax(mem_logits, dim=-1) + log_lambda,
        )
        return logp_joint, None

    def _get_router_device(self):
        device = None
        dtype = None
        for p in self.base_model.parameters():
            device = p.device
            dtype = p.dtype
            break
        if device is None:
            device = torch.device('cpu')
            dtype = torch.float32
        return device, dtype

    def _to_router_device(self):
        if self.router is None:
            return
        device, dtype = self._get_router_device()
        self.router.to(device=device, dtype=dtype)

    @staticmethod
    def _extract_epoch_from_name(filename: str) -> int:
        matches = re.findall(r'\d+', filename)
        return int(matches[-1]) if matches else -1

    @staticmethod
    def _load_router_config(router_path: str):
        if router_path is None:
            return None

        config_file = None
        if os.path.isdir(router_path):
            candidate = os.path.join(router_path, 'router_config.json')
            if os.path.exists(candidate):
                config_file = candidate
        else:
            candidate = os.path.join(os.path.dirname(router_path), 'router_config.json')
            if os.path.exists(candidate):
                config_file = candidate

        if config_file is None:
            return None

        with open(config_file, encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _find_latest_checkpoint(router_dir: str) -> str:
        pt_files = [f for f in os.listdir(router_dir) if f.endswith('.pt')]
        if not pt_files:
            raise FileNotFoundError(f'No .pt checkpoint found in {router_dir}')
        pt_files.sort(key=lambda name: (MemDecodeForCausalLM._extract_epoch_from_name(name), name))
        return os.path.join(router_dir, pt_files[-1])

    def _load_router(self, router_path: str | None):
        if router_path is None:
            return

        if not hasattr(self, 'router') or self.router is None:
            return

        route_cfg = None
        if os.path.isdir(router_path):
            config = self._load_router_config(router_path)
            if config is not None:
                self.router_config = config
                route_cfg = config
            ckpt_path = self._find_latest_checkpoint(router_path)
        else:
            ckpt_path = router_path
            route_cfg = self._load_router_config(router_path)

        if route_cfg is not None:
            self.router_config = route_cfg

        state_dict = torch.load(ckpt_path, map_location=self._get_router_device()[0])
        if isinstance(state_dict, dict):
            for key in ['state_dict', 'model_state_dict', 'router_state_dict']:
                if isinstance(state_dict.get(key, None), dict):
                    state_dict = state_dict[key]
                    break
        self.router.load_state_dict(state_dict)

    def _adaptive_fuse(
        self,
        base_logits: torch.Tensor,
        mem_logits: torch.Tensor,
        base_hidden_states: torch.Tensor,
        mem_hidden_states: torch.Tensor,
    ):
        if self.router is None:
            return self._fixed_fuse(base_logits, mem_logits)

        router_device = self._get_router_device()[0]
        base_hs = base_hidden_states.to(router_device)
        mem_hs = mem_hidden_states.to(router_device)

        log_mixing = self.router(
            base_hs=base_hs,
            mem_hs=mem_hs,
            base_logits=base_logits.to(router_device),
            mem_logits=mem_logits.to(router_device),
        )
        log_one_minus_lambda = log_mixing[:, :, 0].unsqueeze(-1)
        log_lambda = log_mixing[:, :, 1].unsqueeze(-1)

        if self.lambda_base_only_threshold is not None:
            lambda_probs = torch.exp(log_lambda)
            gate = lambda_probs < self.lambda_base_only_threshold
            log_one_minus_lambda = torch.where(gate, torch.zeros_like(log_one_minus_lambda), log_one_minus_lambda)
            log_lambda = torch.where(gate, torch.full_like(log_lambda, float('-inf')), log_lambda)

        base_logits, mem_logits = self._align_fusion_logits(base_logits, mem_logits)

        logp_base = F.log_softmax(base_logits.to(self._get_router_device()[0]), dim=-1)
        logp_mem = F.log_softmax(mem_logits.to(self._get_router_device()[0]), dim=-1)

        fused_logits = torch.logaddexp(logp_base + log_one_minus_lambda, logp_mem + log_lambda)
        return fused_logits, log_mixing.detach()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        self.base_model.load_weights(weights)

    def load_auxiliary_weights(self, checkpoint_path: str, device: torch.device = None):
        if self.memory_model is None:
            return
        if checkpoint_path is None:
            return
        load_model_weights(self.memory_model, checkpoint_path, device=device)

    def update_weights(self):
        self.base_model.update_weights()
        if self.memory_model is not None:
            self.memory_model.update_weights()
