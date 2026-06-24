# Copyright (c) OpenMMLab. All rights reserved.
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.config import MemDecodeConfig, ModelConfig


def align_logits_to_base(logits: torch.Tensor, base_vocab_size: int) -> torch.Tensor:
    """Align logits to the base model vocabulary size."""
    vocab_size = logits.size(-1)
    if vocab_size == base_vocab_size:
        return logits
    if vocab_size > base_vocab_size:
        return logits[..., :base_vocab_size]

    pad_shape = (*logits.shape[:-1], base_vocab_size - vocab_size)
    padding = logits.new_full(pad_shape, -math.inf)
    return torch.cat((logits, padding), dim=-1)


class RouterNetwork(nn.Module):
    """Small router that predicts base/memory log mixing weights per token."""

    def __init__(self, hidden_size: int, intermediate_size: int | None = None):
        super().__init__()
        self.hidden_size = int(hidden_size)
        if intermediate_size is None:
            self.network = nn.Linear(self.hidden_size, 2)
        else:
            self.network = nn.Sequential(
                nn.Linear(self.hidden_size, int(intermediate_size)),
                nn.GELU(),
                nn.Linear(int(intermediate_size), 2),
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.network(hidden_states), dim=-1)


class MemDecodeFusion(nn.Module):
    """Fuse base and memory model logits into log probabilities."""

    def __init__(self, model_config: ModelConfig, memdecode_config: MemDecodeConfig):
        super().__init__()
        self.model_config = model_config
        self.memdecode_config = memdecode_config
        self.base_vocab_size = int(model_config.vocab_size)
        self.lambda_value = float(memdecode_config.lambda_value)
        self.adaptive_router = bool(memdecode_config.adaptive_router)
        self.lambda_base_only_threshold = float(memdecode_config.lambda_base_only_threshold)
        self.router: RouterNetwork | nn.Module | None = None

        if self.adaptive_router:
            if memdecode_config.router_path is None:
                raise ValueError('router_path is required when adaptive_router is enabled.')
            self.router = self._load_router(memdecode_config.router_path)

    def forward(
        self,
        base_logits: torch.Tensor,
        memory_logits: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        base_logits = align_logits_to_base(base_logits, self.base_vocab_size)
        memory_logits = align_logits_to_base(memory_logits, self.base_vocab_size)
        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        memory_log_probs = torch.log_softmax(memory_logits, dim=-1)

        if self.adaptive_router:
            return self._adaptive_fusion(base_log_probs, memory_log_probs, hidden_states)

        if self.lambda_value == 0.0:
            return base_log_probs, None
        if self.lambda_value == 1.0:
            return memory_log_probs, None

        base_log_weight = base_log_probs.new_tensor(math.log1p(-self.lambda_value))
        memory_log_weight = memory_log_probs.new_tensor(math.log(self.lambda_value))
        fused = torch.logaddexp(base_log_probs + base_log_weight, memory_log_probs + memory_log_weight)
        return fused, None

    def _adaptive_fusion(
        self,
        base_log_probs: torch.Tensor,
        memory_log_probs: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden_states is None:
            raise ValueError('hidden_states are required when adaptive_router is enabled.')

        assert self.router is not None
        self.router.to(device=base_log_probs.device)
        router_dtype = self._router_dtype()
        router_hidden_states = hidden_states.to(device=base_log_probs.device, dtype=router_dtype)
        log_weights = self.router(router_hidden_states)
        log_weights = log_weights.to(dtype=base_log_probs.dtype)
        if log_weights.shape[-1] != 2:
            raise ValueError(f'router must produce 2 log mixing weights, got {log_weights.shape[-1]}')

        routing_info: dict[str, torch.Tensor] = {'log_weights': log_weights.detach()}
        if self.lambda_base_only_threshold >= 0.0:
            memory_weight = log_weights[..., 1].exp()
            base_only_mask = memory_weight <= self.lambda_base_only_threshold
            log_weights = log_weights.clone()
            log_weights[..., 0] = torch.where(
                base_only_mask,
                torch.zeros((), device=log_weights.device, dtype=log_weights.dtype),
                log_weights[..., 0],
            )
            log_weights[..., 1] = torch.where(
                base_only_mask,
                torch.full((), -math.inf, device=log_weights.device, dtype=log_weights.dtype),
                log_weights[..., 1],
            )
            routing_info['base_only_mask'] = base_only_mask.detach()
            routing_info['thresholded_log_weights'] = log_weights.detach()

        fused = torch.logaddexp(base_log_probs + log_weights[..., 0:1], memory_log_probs + log_weights[..., 1:2])
        return fused, routing_info

    def _router_dtype(self) -> torch.dtype:
        assert self.router is not None
        for parameter in self.router.parameters():
            return parameter.dtype
        return torch.float32

    def _load_router(self, router_path: str) -> RouterNetwork | nn.Module:
        checkpoint = torch.load(Path(router_path), map_location='cpu')
        if isinstance(checkpoint, nn.Module):
            return checkpoint
        if not isinstance(checkpoint, dict):
            raise ValueError('router checkpoint must be a dict or nn.Module.')

        config = self._router_config_from_checkpoint(checkpoint)
        hidden_size = int(config.get('hidden_size', self.model_config.hidden_size))
        intermediate_size = config.get('intermediate_size')
        router = RouterNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size)

        state_dict = self._router_state_dict_from_checkpoint(checkpoint)
        if state_dict:
            router.load_state_dict(state_dict)
        return router

    @staticmethod
    def _router_config_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
        config = checkpoint.get('config', checkpoint.get('router_config', {}))
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError('router config must be a dict.')
        return config

    @staticmethod
    def _router_state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
        for key in ('state_dict', 'model_state_dict', 'router_state_dict'):
            state_dict = checkpoint.get(key)
            if state_dict is not None:
                if not isinstance(state_dict, dict):
                    raise ValueError(f'{key} must be a state dict.')
                return state_dict

        if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            return checkpoint
        return {}
