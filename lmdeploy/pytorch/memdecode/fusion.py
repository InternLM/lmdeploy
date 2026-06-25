# Copyright (c) OpenMMLab. All rights reserved.
import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lmdeploy.pytorch.config import MemDecodeConfig

DEFAULT_ROUTER_CONFIG = {
    'num_layers': 2,
    'input_mode': 'both',
    'use_scalars': True,
    'scalar_proj_dim': 64,
    'hidden_dim': 128,
    'dropout': 0.2,
}


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
    """Router that predicts per-token base/memory log mixing weights."""

    _INPUT_MODES = {'both', 'memory_only', 'mem_hidden_both_scalars'}

    def __init__(self, config: dict[str, Any], base_hidden_size: int, memory_hidden_size: int):
        super().__init__()
        self.config = dict(config)
        self.input_mode = self.config.get('input_mode', DEFAULT_ROUTER_CONFIG['input_mode'])
        if self.input_mode not in self._INPUT_MODES:
            raise ValueError(f'unsupported router input_mode: {self.input_mode}')

        self.use_scalars = bool(self.config.get('use_scalars', DEFAULT_ROUTER_CONFIG['use_scalars']))
        self.num_scalars = self._num_scalars_for_input_mode()
        scalar_proj_dim = int(self.config.get('scalar_proj_dim', DEFAULT_ROUTER_CONFIG['scalar_proj_dim']) or 0)
        self.scalar_projectors = None
        if self.use_scalars and scalar_proj_dim > 0:
            self.scalar_projectors = nn.ModuleList([
                nn.Sequential(nn.Linear(1, scalar_proj_dim), nn.ReLU()) for _ in range(self.num_scalars)
            ])

        hidden_dim = int(self.config.get('hidden_dim', DEFAULT_ROUTER_CONFIG['hidden_dim']))
        num_layers = max(int(self.config.get('num_layers', DEFAULT_ROUTER_CONFIG['num_layers'])), 1)
        dropout = float(self.config.get('dropout', DEFAULT_ROUTER_CONFIG['dropout']))
        input_dim = self._hidden_input_dim(base_hidden_size, memory_hidden_size)
        if self.use_scalars:
            input_dim += self.num_scalars * scalar_proj_dim if self.scalar_projectors is not None else self.num_scalars

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, 2))
        else:
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.GELU()])
            layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, 2))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        base_hidden_states: torch.Tensor,
        memory_hidden_states: torch.Tensor,
        scalar_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        router_input = self._hidden_input(base_hidden_states, memory_hidden_states)
        if self.use_scalars:
            if scalar_features is None:
                raise ValueError('scalar_features are required for this router configuration.')
            if self.scalar_projectors is not None:
                projected_scalars = [
                    projector(scalar_features[..., idx:idx + 1])
                    for idx, projector in enumerate(self.scalar_projectors)
                ]
                scalar_features = torch.cat(projected_scalars, dim=-1)
            router_input = torch.cat((router_input, scalar_features), dim=-1)
        return torch.log_softmax(self.mlp(router_input), dim=-1)

    def _hidden_input_dim(self, base_hidden_size: int, memory_hidden_size: int) -> int:
        if self.input_mode == 'both':
            return int(base_hidden_size) + int(memory_hidden_size)
        if self.input_mode in {'memory_only', 'mem_hidden_both_scalars'}:
            return int(memory_hidden_size)
        raise AssertionError('unreachable')

    def _hidden_input(self, base_hidden_states: torch.Tensor, memory_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.input_mode == 'both':
            return torch.cat((base_hidden_states, memory_hidden_states), dim=-1)
        if self.input_mode in {'memory_only', 'mem_hidden_both_scalars'}:
            return memory_hidden_states
        raise AssertionError('unreachable')

    def _num_scalars_for_input_mode(self) -> int:
        if self.input_mode == 'memory_only':
            return 2
        return 4


class MemDecodeFusion(nn.Module):
    """Fuse base and memory model logits into log probabilities."""

    def __init__(
        self,
        config: MemDecodeConfig,
        base_hidden_size: int,
        memory_hidden_size: int,
        base_vocab_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.config = config
        self.base_hidden_size = int(base_hidden_size)
        self.memory_hidden_size = int(memory_hidden_size)
        self.base_vocab_size = int(base_vocab_size)
        self.lambda_value = float(config.lambda_value)
        self.adaptive_router = bool(config.adaptive_router)
        self.lambda_base_only_threshold = float(config.lambda_base_only_threshold)
        self.router: RouterNetwork | None = None
        self.router_config: dict[str, Any] | None = None

        if self.adaptive_router:
            if config.router_path is None:
                raise ValueError('router_path is required when adaptive_router is enabled.')
            self.router_config, checkpoint_path = self._resolve_router_config_and_checkpoint(config.router_path)
            self.router = RouterNetwork(self.router_config, self.base_hidden_size, self.memory_hidden_size)
            state_dict = self._load_router_state_dict(checkpoint_path)
            self.router.load_state_dict(state_dict)
            self.router.eval()
            self.router.to(device=device, dtype=dtype)

    def forward(
        self,
        base_logits: torch.Tensor,
        memory_logits: torch.Tensor,
        base_hidden_states: torch.Tensor | None = None,
        memory_hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        base_logits = align_logits_to_base(base_logits, self.base_vocab_size)
        memory_logits = align_logits_to_base(memory_logits, self.base_vocab_size)
        base_log_probs = torch.log_softmax(base_logits, dim=-1)
        memory_log_probs = torch.log_softmax(memory_logits, dim=-1)

        if self.adaptive_router:
            return self._adaptive_fusion(
                base_log_probs,
                memory_log_probs,
                base_hidden_states=base_hidden_states,
                memory_hidden_states=memory_hidden_states,
            )

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
        base_hidden_states: torch.Tensor | None,
        memory_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if base_hidden_states is None or memory_hidden_states is None:
            raise ValueError(
                'base_hidden_states and memory_hidden_states are required when adaptive_router is enabled.'
            )

        assert self.router is not None
        router_dtype = next(self.router.parameters()).dtype
        router_device = base_log_probs.device
        self.router.to(device=router_device)
        base_hidden_states = base_hidden_states.to(device=router_device, dtype=router_dtype)
        memory_hidden_states = memory_hidden_states.to(device=router_device, dtype=router_dtype)
        scalar_features = self._scalar_features(base_log_probs, memory_log_probs, self.router.input_mode)
        scalar_features = scalar_features.to(dtype=router_dtype)

        log_weights = self.router(base_hidden_states, memory_hidden_states, scalar_features)
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

    @staticmethod
    def _scalar_features(
        base_log_probs: torch.Tensor,
        memory_log_probs: torch.Tensor,
        input_mode: str,
    ) -> torch.Tensor:
        base_probs = base_log_probs.exp()
        memory_probs = memory_log_probs.exp()
        memory_confidence = memory_probs.max(dim=-1).values
        memory_entropy = MemDecodeFusion._entropy(memory_probs, memory_log_probs)
        if input_mode == 'memory_only':
            return torch.stack((memory_confidence, memory_entropy), dim=-1)

        base_confidence = base_probs.max(dim=-1).values
        base_entropy = MemDecodeFusion._entropy(base_probs, base_log_probs)
        return torch.stack((base_confidence, base_entropy, memory_confidence, memory_entropy), dim=-1)

    @staticmethod
    def _entropy(probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        finite_log_probs = torch.where(torch.isfinite(log_probs), log_probs, torch.zeros_like(log_probs))
        return -(probs * finite_log_probs).sum(dim=-1)

    @classmethod
    def _resolve_router_config_and_checkpoint(cls, router_path: str) -> tuple[dict[str, Any], Path]:
        path = Path(router_path)
        if path.is_dir():
            checkpoint_path = cls._latest_checkpoint(path)
            config_path = path / 'router_config.json'
        else:
            checkpoint_path = path
            config_path = path.parent / 'router_config.json'

        checkpoint_config = cls._router_config_from_checkpoint(checkpoint_path)
        file_config = cls._router_config_from_file(config_path)
        config = dict(DEFAULT_ROUTER_CONFIG)
        config.update(checkpoint_config)
        config.update(file_config)
        return config, checkpoint_path

    @staticmethod
    def _latest_checkpoint(router_dir: Path) -> Path:
        checkpoints = sorted(
            router_dir.glob('*.pt'),
            key=lambda path: (MemDecodeFusion._checkpoint_number(path), path.name),
        )
        if not checkpoints:
            raise ValueError(f'no .pt router checkpoints found in {router_dir}')
        return checkpoints[-1]

    @staticmethod
    def _checkpoint_number(path: Path) -> int:
        matches = re.findall(r'\d+', path.stem)
        return int(matches[-1]) if matches else -1

    @staticmethod
    def _router_config_from_file(config_path: Path) -> dict[str, Any]:
        if not config_path.exists():
            return {}
        with config_path.open() as f:
            config = json.load(f)
        if not isinstance(config, dict):
            raise ValueError('router_config.json must contain a JSON object.')
        return config

    @staticmethod
    def _router_config_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            return {}
        config = checkpoint.get('router_config', checkpoint.get('config', {}))
        if config is None:
            return {}
        if not isinstance(config, dict):
            raise ValueError('router config must be a dict.')
        return config

    @staticmethod
    def _load_router_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise ValueError('router checkpoint must be a state-dict checkpoint.')

        state_dict = None
        for key in ('state_dict', 'router_state_dict', 'model_state_dict'):
            value = checkpoint.get(key)
            if value is not None:
                state_dict = value
                break
        if state_dict is None and checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            state_dict = checkpoint

        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError('router checkpoint must contain a non-empty state dict.')
        if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
            raise ValueError('router checkpoint state dict must contain only tensors.')
        return state_dict
