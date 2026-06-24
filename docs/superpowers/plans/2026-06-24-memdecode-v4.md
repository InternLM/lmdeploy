# MemDecode v4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current MemDecode model wrapper with an agent-owned memory model path that fuses sliced base and memory logits.

**Architecture:** Base model execution stays in `BaseModelAgent` as a normal patched model. A new `MemDecodeAgent` owns memory model build, cache, graph runner, and forward lifecycle. A separate fusion unit aligns memory logits to the base vocab and applies fixed or adaptive fusion only after hidden states are sliced to needed positions.

**Tech Stack:** Python dataclasses, PyTorch inference modules, lmdeploy PyTorch engine/model-agent/cache abstractions, pytest.

______________________________________________________________________

## File Structure

- Create `lmdeploy/pytorch/memdecode/__init__.py`: exports `MemDecodeConfig`, `MemDecodeAgent`, `BaseMemDecodeAgent`, `build_memdecode_agent`, and `MemDecodeFusion`.
- Create `lmdeploy/pytorch/memdecode/fusion.py`: owns vocab alignment, fixed log-prob fusion, adaptive router loading, and routed-weight output.
- Create `lmdeploy/pytorch/memdecode/agent.py`: owns memory model build/load, graph runner, cache engines, state cache engines, forward, warmup, reset, and release.
- Modify `lmdeploy/pytorch/config.py`: replace flat MemDecode fields on `ModelConfig` with `memdecode_config: MemDecodeConfig | None`, normalize flat `hf_overrides` into that object, validate lambda and SSM matching.
- Modify `lmdeploy/pytorch/engine/model_agent/agent.py`: remove wrapper-specific memory cache plumbing, create/use `MemDecodeAgent`, fuse sliced logits after base and memory forwards, reject MemDecode sleep/wakeup.
- Modify `lmdeploy/pytorch/engine/executor/base.py`: reserve memory for memory KV/state through `memdecode_config`, validate spec-decode mutual exclusion and SSM matching.
- Modify `lmdeploy/pytorch/models/module_map.py`: remove the `MemDecodeForCausalLM` architecture mapping.
- Delete or retire `lmdeploy/pytorch/models/memdecode.py`: move the useful router/fusion logic into `lmdeploy/pytorch/memdecode/fusion.py`; do not keep wrapper as an active model class.
- Modify `examples/memdec/main.py`: stop setting `architectures=["MemDecodeForCausalLM"]`; pass flat MemDecode overrides that normalize into `ModelConfig.memdecode_config`.
- Create `tests/pytorch/memdecode/test_fusion.py`: high-value CPU tests for vocab alignment and lambda endpoints.
- Create `tests/pytorch/memdecode/test_agent.py`: lightweight tests for memory agent disabled/enabled behavior and forward input usage.
- Create `tests/pytorch/config/test_memdecode_config.py`: tests override normalization, lambda validation, and SSM mismatch validation.
- Modify `tests/pytorch/engine/test_executor_base.py`: tests memory cache block sizing through `memdecode_config`.
- Modify `tests/pytorch/engine/test_model_agent.py`: tests MemDecode sleep/wakeup rejection and graph reset/release propagation.

## Task 1: Normalize MemDecode Configuration

**Files:**

- Modify: `lmdeploy/pytorch/config.py`

- Create: `tests/pytorch/config/test_memdecode_config.py`

- [ ] **Step 1: Write failing config normalization tests**

Create `tests/pytorch/config/test_memdecode_config.py` with:

```python
from types import SimpleNamespace

import pytest
import torch

from lmdeploy.pytorch.config import DistConfig, MemDecodeConfig, ModelConfig


def _hf_config(vocab_size=32000):
    return SimpleNamespace(
        architectures=['Qwen3ForCausalLM'],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=1024,
        model_type='qwen3',
        num_attention_heads=8,
        num_hidden_layers=2,
        num_key_value_heads=2,
        torch_dtype='float16',
        vocab_size=vocab_size,
    )


def _model_config_from_hf(hf_config, model_path, tp=1, states_shapes=None):
    return ModelConfig(
        hidden_size=hf_config.hidden_size,
        num_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        bos_token_id=hf_config.bos_token_id,
        eos_token_id=[hf_config.eos_token_id],
        head_dim=128,
        vocab_size=hf_config.vocab_size,
        hf_config=hf_config,
        dist_config=DistConfig(tp=tp),
        states_shapes=list(states_shapes or []),
    )


def test_memdecode_config_dataclass_keeps_fusion_policy_off_model_config():
    memory_config = _model_config_from_hf(_hf_config(), 'memory')
    memdecode_config = MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=memory_config,
        lambda_value=0.25,
        adaptive_router=True,
        router_path='router.pt',
        lambda_base_only_threshold=0.1,
    )

    base_config = _model_config_from_hf(_hf_config(), 'base')
    base_config.memdecode_config = memdecode_config

    assert base_config.memdecode_config.lambda_value == 0.25
    assert not hasattr(base_config, 'lambda_value')
    assert not hasattr(memory_config, 'lambda_value')


def test_memdecode_config_validates_lambda_range():
    memory_config = _model_config_from_hf(_hf_config(), 'memory')

    with pytest.raises(ValueError, match='lambda_value must be in \\[0, 1\\]'):
        MemDecodeConfig(memory_model_path='memory', memory_model_config=memory_config, lambda_value=1.25)


def test_validate_memdecode_config_rejects_ssm_mismatch():
    base_config = _model_config_from_hf(_hf_config(), 'base', states_shapes=[((1, 2), torch.float16)])
    memory_config = _model_config_from_hf(_hf_config(), 'memory')
    base_config.memdecode_config = MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=memory_config,
    )

    with pytest.raises(ValueError, match='Base and memory model must both use SSM state caches'):
        base_config.validate_memdecode_config()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/pytorch/config/test_memdecode_config.py -q
```

Expected: FAIL because `MemDecodeConfig`, `ModelConfig.memdecode_config`, and `validate_memdecode_config()` do not exist yet.

- [ ] **Step 3: Add `MemDecodeConfig` and `ModelConfig.memdecode_config`**

In `lmdeploy/pytorch/config.py`, add this dataclass before `ModelConfig`:

```python
@dataclass
class MemDecodeConfig:
    """Configuration for MemDecode auxiliary memory model and fusion."""

    memory_model_path: str
    memory_model_config: 'ModelConfig'
    lambda_value: float = 1.0
    adaptive_router: bool = False
    router_path: str | None = None
    lambda_base_only_threshold: float = -1.0

    def __post_init__(self):
        self.lambda_value = float(self.lambda_value)
        self.lambda_base_only_threshold = float(self.lambda_base_only_threshold)
        if not 0.0 <= self.lambda_value <= 1.0:
            raise ValueError(f'lambda_value must be in [0, 1], got {self.lambda_value}')
        if self.adaptive_router and self.router_path is None:
            raise ValueError('router_path is required when adaptive_router is enabled.')
```

Replace the current flat MemDecode fields on `ModelConfig` with:

```python
    # memdecode
    memdecode_config: MemDecodeConfig | None = None
```

Add this method to `ModelConfig`:

```python
    def validate_memdecode_config(self):
        """Validate base and memory model compatibility for MemDecode."""
        memdecode_config = self.memdecode_config
        if memdecode_config is None:
            return

        memory_model_config = memdecode_config.memory_model_config
        base_has_state = len(self.states_shapes) > 0
        memory_has_state = len(memory_model_config.states_shapes) > 0
        if base_has_state != memory_has_state:
            raise ValueError('Base and memory model must both use SSM state caches or both not use them.')
```

- [ ] **Step 4: Normalize flat `hf_overrides` into `MemDecodeConfig`**

In `ModelConfig.from_pretrained()`, replace the current `memory_model_path`, `lambda_value`, `adaptive_router`, `router_path`, and `lambda_base_only_threshold` field assignment block with:

```python
        hf_overrides = dict(hf_overrides or {})
        memory_model_path = hf_overrides.pop('memory_model_path', None)
        lambda_value = hf_overrides.pop('lambda_value', 1.0)
        adaptive_router = hf_overrides.pop('adaptive_router', False)
        router_path = hf_overrides.pop('router_path', None)
        lambda_base_only_threshold = hf_overrides.pop('lambda_base_only_threshold', -1.0)

        if memory_model_path is not None:
            memory_model_config = cls.from_pretrained(
                memory_model_path,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                dist_config=dist_config,
                model_format=None,
                device_type=device_type,
                block_size=block_size,
            )
            memdecode_config = MemDecodeConfig(
                memory_model_path=memory_model_path,
                memory_model_config=memory_model_config,
                lambda_value=lambda_value,
                adaptive_router=adaptive_router,
                router_path=router_path,
                lambda_base_only_threshold=lambda_base_only_threshold,
            )
        else:
            memdecode_config = None
```

After `model_config = cls.from_hf_config(...)`, assign and validate:

```python
        model_config.memdecode_config = memdecode_config
        model_config.validate_memdecode_config()
```

Keep the existing vocab-size warning, but read from:

```python
        if memdecode_config is not None:
            memory_vocab_size = memdecode_config.memory_model_config.vocab_size
```

Remove writes such as `hf_config.memory_model_path`, `hf_config.lambda_value`, and `model_config.memory_model_path`; the wrapper should no longer consume HF config fields.

- [ ] **Step 5: Run config tests**

Run:

```bash
pytest tests/pytorch/config/test_memdecode_config.py tests/pytorch/config/test_model_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit config normalization**

```bash
git add lmdeploy/pytorch/config.py tests/pytorch/config/test_memdecode_config.py
git commit -m "refactor: normalize memdecode config"
```

## Task 2: Extract MemDecode Fusion

**Files:**

- Create: `lmdeploy/pytorch/memdecode/__init__.py`

- Create: `lmdeploy/pytorch/memdecode/fusion.py`

- Create: `tests/pytorch/memdecode/test_fusion.py`

- [ ] **Step 1: Write failing fusion tests**

Create `tests/pytorch/memdecode/test_fusion.py` with:

```python
import torch

from lmdeploy.pytorch.config import MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.memdecode.fusion import MemDecodeFusion, align_logits_to_base


def _model_config(vocab_size):
    return ModelConfig(
        hidden_size=4,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=4,
        vocab_size=vocab_size,
    )


def _fusion(lambda_value, memory_vocab_size=3, base_vocab_size=3):
    memory_config = _model_config(memory_vocab_size)
    return MemDecodeFusion(
        MemDecodeConfig(
            memory_model_path='memory',
            memory_model_config=memory_config,
            lambda_value=lambda_value,
        ),
        base_hidden_size=4,
        memory_hidden_size=4,
        base_vocab_size=base_vocab_size,
    )


def test_align_logits_truncates_larger_memory_vocab():
    logits = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])

    aligned = align_logits_to_base(logits, base_vocab_size=2)

    torch.testing.assert_close(aligned, torch.tensor([[[1.0, 2.0]]]))


def test_align_logits_pads_smaller_memory_vocab_with_negative_infinity():
    logits = torch.tensor([[[1.0, 2.0]]])

    aligned = align_logits_to_base(logits, base_vocab_size=4)

    assert aligned.shape == (1, 1, 4)
    torch.testing.assert_close(aligned[..., :2], logits)
    assert torch.isneginf(aligned[..., 2:]).all()


def test_fixed_fusion_lambda_zero_is_base_only_logprob():
    fusion = _fusion(lambda_value=0.0)
    base_logits = torch.tensor([[[3.0, 1.0, 0.0]]])
    memory_logits = torch.tensor([[[0.0, 1.0, 3.0]]])

    fused, routed = fusion(
        base_logits=base_logits,
        memory_logits=memory_logits,
        base_hidden_states=None,
        memory_hidden_states=None,
    )

    torch.testing.assert_close(fused, torch.log_softmax(base_logits, dim=-1))
    assert routed is None


def test_fixed_fusion_lambda_one_is_memory_only_logprob():
    fusion = _fusion(lambda_value=1.0)
    base_logits = torch.tensor([[[3.0, 1.0, 0.0]]])
    memory_logits = torch.tensor([[[0.0, 1.0, 3.0]]])

    fused, routed = fusion(
        base_logits=base_logits,
        memory_logits=memory_logits,
        base_hidden_states=None,
        memory_hidden_states=None,
    )

    torch.testing.assert_close(fused, torch.log_softmax(memory_logits, dim=-1))
    assert routed is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/pytorch/memdecode/test_fusion.py -q
```

Expected: FAIL because `lmdeploy.pytorch.memdecode.fusion` does not exist.

- [ ] **Step 3: Implement fusion module**

Create `lmdeploy/pytorch/memdecode/fusion.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import re

import torch
import torch.nn.functional as F
from torch import nn

from lmdeploy.pytorch.config import MemDecodeConfig


DEFAULT_ROUTER_CONFIG = {
    'num_layers': 2,
    'input_mode': 'both',
    'use_scalars': True,
    'scalar_proj_dim': 64,
    'hidden_dim': 128,
}


def align_logits_to_base(logits: torch.Tensor, base_vocab_size: int) -> torch.Tensor:
    """Align logits to base vocab size by truncating or padding with -inf."""
    vocab_size = logits.size(-1)
    if vocab_size == base_vocab_size:
        return logits
    if vocab_size > base_vocab_size:
        return logits[..., :base_vocab_size]
    pad = logits.new_full((*logits.shape[:-1], base_vocab_size - vocab_size), float('-inf'))
    return torch.cat([logits, pad], dim=-1)


class RouterNetwork(nn.Module):
    """Router network that emits per-token log mixing weights."""

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

        input_dim = 0
        if input_mode == 'both':
            input_dim += base_hidden_size + mem_hidden_size
        elif input_mode in {'memory_only', 'mem_hidden_both_scalars'}:
            input_dim += mem_hidden_size
        else:
            raise ValueError(f'Unknown input_mode: {input_mode}')

        self.num_scalars = 4 if input_mode in {'both', 'mem_hidden_both_scalars'} else 2
        if use_scalars:
            self.scalar_projectors = nn.ModuleList(
                [nn.Sequential(nn.Linear(1, scalar_proj_dim), nn.ReLU()) for _ in range(self.num_scalars)])
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
                raise ValueError('router in both mode requires base_hs and mem_hs.')
            features.extend([base_hs, mem_hs])
        else:
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

        return F.log_softmax(self.mlp(torch.cat(features, dim=-1)), dim=-1)


class MemDecodeFusion:
    """Fuse sliced base and memory logits."""

    def __init__(self,
                 config: MemDecodeConfig,
                 base_hidden_size: int,
                 memory_hidden_size: int,
                 base_vocab_size: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None):
        self.config = config
        self.base_vocab_size = base_vocab_size
        self.router = None
        if config.adaptive_router:
            router_config = self._load_router_config(config.router_path) or DEFAULT_ROUTER_CONFIG
            self.router = RouterNetwork(base_hidden_size, memory_hidden_size, **router_config)
            if device is not None:
                self.router.to(device=device, dtype=dtype)
            self._load_router(config.router_path, device=device)

    def __call__(self, base_logits, memory_logits, base_hidden_states=None, memory_hidden_states=None):
        memory_logits = align_logits_to_base(memory_logits, self.base_vocab_size)
        base_logits = align_logits_to_base(base_logits, self.base_vocab_size)
        if self.router is None:
            return self._fixed_fuse(base_logits, memory_logits), None
        return self._adaptive_fuse(base_logits, memory_logits, base_hidden_states, memory_hidden_states)

    def _fixed_fuse(self, base_logits, memory_logits):
        lam = self.config.lambda_value
        if lam <= 0.0:
            return F.log_softmax(base_logits, dim=-1)
        if lam >= 1.0:
            return F.log_softmax(memory_logits, dim=-1)
        return torch.logaddexp(
            F.log_softmax(base_logits, dim=-1) + math.log1p(-lam),
            F.log_softmax(memory_logits, dim=-1) + math.log(lam),
        )

    def _adaptive_fuse(self, base_logits, memory_logits, base_hidden_states, memory_hidden_states):
        if base_hidden_states is None or memory_hidden_states is None:
            raise ValueError('adaptive MemDecode fusion requires sliced base and memory hidden states.')
        log_mixing = self.router(
            base_hs=base_hidden_states,
            mem_hs=memory_hidden_states,
            base_logits=base_logits,
            mem_logits=memory_logits,
        )
        log_one_minus_lambda = log_mixing[..., 0:1]
        log_lambda = log_mixing[..., 1:2]
        if self.config.lambda_base_only_threshold >= 0:
            lambda_probs = torch.exp(log_lambda)
            gate = lambda_probs < self.config.lambda_base_only_threshold
            log_one_minus_lambda = torch.where(gate, torch.zeros_like(log_one_minus_lambda), log_one_minus_lambda)
            log_lambda = torch.where(gate, torch.full_like(log_lambda, float('-inf')), log_lambda)
        fused = torch.logaddexp(
            F.log_softmax(base_logits, dim=-1) + log_one_minus_lambda,
            F.log_softmax(memory_logits, dim=-1) + log_lambda,
        )
        return fused, log_mixing.detach()

    @staticmethod
    def _extract_epoch_from_name(filename: str) -> int:
        matches = re.findall(r'\d+', filename)
        return int(matches[-1]) if matches else -1

    @classmethod
    def _find_latest_checkpoint(cls, router_dir: str) -> str:
        pt_files = [f for f in os.listdir(router_dir) if f.endswith('.pt')]
        if not pt_files:
            raise FileNotFoundError(f'No .pt checkpoint found in {router_dir}')
        pt_files.sort(key=lambda name: (cls._extract_epoch_from_name(name), name))
        return os.path.join(router_dir, pt_files[-1])

    @staticmethod
    def _load_router_config(router_path: str | None):
        if router_path is None:
            return None
        import json
        config_file = os.path.join(router_path, 'router_config.json') if os.path.isdir(router_path) else \
            os.path.join(os.path.dirname(router_path), 'router_config.json')
        if not os.path.exists(config_file):
            return None
        with open(config_file, encoding='utf-8') as f:
            return json.load(f)

    def _load_router(self, router_path: str | None, device: torch.device | None = None):
        if self.router is None or router_path is None:
            return
        ckpt_path = self._find_latest_checkpoint(router_path) if os.path.isdir(router_path) else router_path
        state_dict = torch.load(ckpt_path, map_location=device or 'cpu')
        if isinstance(state_dict, dict):
            for key in ['state_dict', 'model_state_dict', 'router_state_dict']:
                if isinstance(state_dict.get(key, None), dict):
                    state_dict = state_dict[key]
                    break
        self.router.load_state_dict(state_dict)
```

Create `lmdeploy/pytorch/memdecode/__init__.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import MemDecodeConfig

from .fusion import MemDecodeFusion, align_logits_to_base

__all__ = ['MemDecodeConfig', 'MemDecodeFusion', 'align_logits_to_base']
```

- [ ] **Step 4: Run fusion tests**

Run:

```bash
pytest tests/pytorch/memdecode/test_fusion.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit fusion extraction**

```bash
git add lmdeploy/pytorch/memdecode tests/pytorch/memdecode/test_fusion.py
git commit -m "feat: add memdecode fusion unit"
```

## Task 3: Add MemDecodeAgent

**Files:**

- Modify: `lmdeploy/pytorch/memdecode/__init__.py`

- Create: `lmdeploy/pytorch/memdecode/agent.py`

- Create: `tests/pytorch/memdecode/test_agent.py`

- [ ] **Step 1: Write focused agent tests**

Create `tests/pytorch/memdecode/test_agent.py` with:

```python
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.config import CacheConfig, MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.memdecode.agent import BaseMemDecodeAgent, MemDecodeAgent


def _model_config():
    return ModelConfig(
        hidden_size=4,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=4,
        vocab_size=8,
    )


def _memdecode_config():
    return MemDecodeConfig(memory_model_path='memory', memory_model_config=_model_config())


def test_disabled_memdecode_agent_is_noop():
    agent = BaseMemDecodeAgent(None, backend_config=None, dist_ctx=None, device='cpu')

    assert not agent.is_enabled()
    assert agent.get_model() is None


def test_memdecode_agent_release_clears_resources():
    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = object()
    agent.cache_engine = object()
    agent.state_cache_engine = object()

    agent.release()

    assert agent.model is None
    assert agent.cache_engine is None
    assert agent.state_cache_engine is None


def test_memdecode_agent_reset_graph_runner_resets_model():
    events = []

    class _Model:

        def reset(self):
            events.append('reset')

    agent = MemDecodeAgent.__new__(MemDecodeAgent)
    agent.model = _Model()

    @contextmanager
    def _ctx():
        events.append('enter')
        yield
        events.append('exit')

    agent.memory_context = _ctx

    agent.reset_graph_runner()

    assert events == ['enter', 'reset', 'exit']
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/pytorch/memdecode/test_agent.py -q
```

Expected: FAIL because `lmdeploy.pytorch.memdecode.agent` does not exist.

- [ ] **Step 3: Implement memory forward and agent lifecycle**

Create `lmdeploy/pytorch/memdecode/agent.py` with:

```python
# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from contextlib import contextmanager
from dataclasses import replace

import torch

from lmdeploy.pytorch.config import BackendConfig, CacheConfig, MemDecodeConfig
from lmdeploy.pytorch.distributed import DistContext, get_dist_manager
from lmdeploy.pytorch.engine.cache_engine import CacheEngine, StateCacheEngine
from lmdeploy.pytorch.model_inputs import ModelInputs, step_ctx_manager
from lmdeploy.pytorch.models.patch import build_patched_model, update_custom_module_map
from lmdeploy.pytorch.weight_loader.model_weight_loader import load_model_weights


@torch.inference_mode()
def memory_model_forward(model: torch.nn.Module,
                         inputs: ModelInputs,
                         model_config,
                         cache_engine: CacheEngine,
                         state_cache_engine: StateCacheEngine | None = None):
    """Run memory model on accepted-token inputs."""
    stream = torch.cuda.current_stream()
    state_caches = None if state_cache_engine is None else state_cache_engine.state_caches
    with torch.cuda.stream(stream), step_ctx_manager(model.ctx_mgr):
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
            model_metas = model.update_model_metas(past_key_values=cache_engine.gpu_cache, context=context)
            input_dict = model.prepare_inputs_for_generation(
                past_key_values=cache_engine.gpu_cache,
                context=context,
            )
            output = model(**input_dict)
            if not isinstance(output, dict):
                output = dict(hidden_states=output)
            output['model_metas'] = model_metas
            output['seq_length'] = context.q_seqlens[:len(inputs.seq_length)]
            output['position_ids'] = context.position_ids
            return output


class BaseMemDecodeAgent:
    """Disabled MemDecode agent."""

    def __init__(self,
                 memdecode_config: MemDecodeConfig | None,
                 backend_config: BackendConfig | None,
                 dist_ctx: DistContext | None,
                 device: str = 'cuda'):
        self.memdecode_config = memdecode_config
        self.backend_config = backend_config
        self.dist_ctx = dist_ctx
        self.device = device
        self.model_config = None if memdecode_config is None else memdecode_config.memory_model_config
        self.cache_engine = None
        self.state_cache_engine = None
        self.model = None

    def is_enabled(self):
        return False

    def set_cache_config(self, cache_config: CacheConfig):
        pass

    def build_model(self, empty_init: bool, build_model_ctx=None):
        pass

    def build_graph_runner(self):
        pass

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        pass

    async def async_forward(self, inputs: ModelInputs):
        return None

    def get_logits(self, hidden_states: torch.Tensor):
        raise RuntimeError('MemDecode is disabled.')

    def reset_graph_runner(self):
        pass

    def release(self):
        pass

    def get_model(self):
        return None


class MemDecodeAgent(BaseMemDecodeAgent):
    """Owns memory model execution for MemDecode."""

    def is_enabled(self):
        return True

    @contextmanager
    def memory_context(self):
        dist_mgr = get_dist_manager()
        with dist_mgr.context(self.dist_ctx):
            yield

    def set_cache_config(self, cache_config: CacheConfig):
        self.cache_config = cache_config

    def build_model(self, empty_init: bool, build_model_ctx=None):
        with self.memory_context():
            custom_module_map = self.model_config.custom_module_map
            if custom_module_map is not None:
                update_custom_module_map(custom_module_map)
            model = build_patched_model(self.model_config, device=self.device, build_model_ctx=build_model_ctx)
            if not empty_init:
                load_model_weights(model, self.memdecode_config.memory_model_path, device=self.device)
            self.model = model

    def build_graph_runner(self):
        with self.memory_context():
            from lmdeploy.pytorch.backends import get_backend
            backend = get_backend()
            self.model = backend.build_graph_runner(
                self.model,
                model_config=self.model_config,
                cache_config=self.cache_config,
                backend_config=self.backend_config,
                device=self.device,
            )

    def build_cache_engine(self, cache_stream: torch.cuda.Stream):
        with self.memory_context():
            dist_ctx = get_dist_manager().current_context()
            tp = dist_ctx.dist_config.attn_tp
            self.cache_engine = CacheEngine(
                self.cache_config,
                self.model_config,
                rank=dist_ctx.rank,
                tp_rank=dist_ctx.attn_tp_group.rank,
                world_size=tp,
                cache_stream=cache_stream,
            )
            self.state_cache_engine = None
            if len(self.model_config.states_shapes) > 0:
                state_cache_config = replace(self.cache_config, states_shapes=list(self.model_config.states_shapes))
                self.state_cache_engine = StateCacheEngine(state_cache_config)

    async def async_forward(self, inputs: ModelInputs):
        output = memory_model_forward(self.model, inputs, self.model_config, self.cache_engine,
                                      self.state_cache_engine)
        await asyncio.sleep(0)
        return output

    def get_logits(self, hidden_states: torch.Tensor):
        return self.model.get_logits(hidden_states)

    def reset_graph_runner(self):
        with self.memory_context():
            if self.model is not None and hasattr(self.model, 'reset'):
                self.model.reset()

    def release(self):
        self.model = None
        self.cache_engine = None
        self.state_cache_engine = None

    def get_model(self):
        if self.model is None:
            return None
        return self.model.get_model() if hasattr(self.model, 'get_model') else self.model


def build_memdecode_agent(memdecode_config: MemDecodeConfig | None,
                          backend_config: BackendConfig,
                          dist_ctx: DistContext,
                          device: str = 'cuda'):
    if memdecode_config is None:
        return BaseMemDecodeAgent(memdecode_config, backend_config, dist_ctx, device=device)
    return MemDecodeAgent(memdecode_config, backend_config, dist_ctx, device=device)
```

- [ ] **Step 4: Export agent symbols**

Update `lmdeploy/pytorch/memdecode/__init__.py` to:

```python
# Copyright (c) OpenMMLab. All rights reserved.
from lmdeploy.pytorch.config import MemDecodeConfig

from .agent import BaseMemDecodeAgent, MemDecodeAgent, build_memdecode_agent
from .fusion import MemDecodeFusion, align_logits_to_base

__all__ = [
    'BaseMemDecodeAgent',
    'MemDecodeAgent',
    'MemDecodeConfig',
    'MemDecodeFusion',
    'align_logits_to_base',
    'build_memdecode_agent',
]
```

- [ ] **Step 5: Run agent tests**

Run:

```bash
pytest tests/pytorch/memdecode/test_agent.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit memory agent**

```bash
git add lmdeploy/pytorch/memdecode tests/pytorch/memdecode/test_agent.py
git commit -m "feat: add memdecode memory agent"
```

## Task 4: Wire MemDecode Into BaseModelAgent

**Files:**

- Modify: `lmdeploy/pytorch/engine/model_agent/agent.py`

- Modify: `tests/pytorch/engine/test_model_agent.py`

- [ ] **Step 1: Write focused lifecycle tests**

Append to `tests/pytorch/engine/test_model_agent.py`:

```python
class TestMemDecodeModelAgentLifecycle:

    def test_memdecode_sleep_is_rejected(self, event_loop):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.state = SleepWakeupState()
        agent.memdecode_agent = SimpleNamespace(is_enabled=lambda: True)

        with pytest.raises(NotImplementedError, match='MemDecode sleep/wakeup is not supported'):
            event_loop.run_until_complete(agent.sleep(level=1))

    def test_memdecode_wakeup_is_rejected(self):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent, SleepWakeupState

        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.state = SleepWakeupState()
        agent.memdecode_agent = SimpleNamespace(is_enabled=lambda: True)

        with pytest.raises(NotImplementedError, match='MemDecode sleep/wakeup is not supported'):
            agent.wakeup(['kv_cache'])

    def test_release_releases_memdecode_agent(self):
        from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent

        events = []
        agent = BaseModelAgent.__new__(BaseModelAgent)
        agent.patched_model = None
        agent.cache_engine = object()
        agent.state_cache_engine = object()
        agent.spec_agent = SimpleNamespace(reset_graph_runner=lambda: events.append('spec_reset'))
        agent.memdecode_agent = SimpleNamespace(release=lambda: events.append('memdecode_release'))
        agent.reset_graph_runner = lambda: events.append('reset_graph_runner')

        agent.release()

        assert events == ['reset_graph_runner', 'memdecode_release']
        assert agent.cache_engine is None
        assert agent.state_cache_engine is None
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/pytorch/engine/test_model_agent.py::TestMemDecodeModelAgentLifecycle -q
```

Expected: FAIL because `BaseModelAgent` does not yet own `memdecode_agent` or reject sleep/wakeup.

- [ ] **Step 3: Add agent ownership**

In `lmdeploy/pytorch/engine/model_agent/agent.py`, import:

```python
from lmdeploy.pytorch.memdecode import MemDecodeFusion, build_memdecode_agent
```

In `BaseModelAgent.__init__`, after `self.spec_agent = build_spec_agent(...)`, add:

```python
        if model_config.memdecode_config is not None and specdecode_config is not None:
            raise ValueError('MemDecode and speculative decoding cannot be enabled together.')
        self.memdecode_agent = build_memdecode_agent(
            model_config.memdecode_config,
            backend_config,
            dist_ctx,
            device=device,
        )
        self.memdecode_fusion = None
        if model_config.memdecode_config is not None:
            mem_cfg = model_config.memdecode_config
            self.memdecode_fusion = MemDecodeFusion(
                mem_cfg,
                base_hidden_size=model_config.hidden_size,
                memory_hidden_size=mem_cfg.memory_model_config.hidden_size,
                base_vocab_size=model_config.vocab_size,
                dtype=model_config.dtype,
            )
```

- [ ] **Step 4: Move memory build/cache lifecycle out of the wrapper path**

In `_build_model()`, remove:

```python
            if hasattr(patched_model, 'load_auxiliary_weights'):
                patched_model.load_auxiliary_weights(self.model_config.memory_model_path, device=device)
```

In `build_model()`, after `self._build_model()`, add:

```python
            self.memdecode_agent.build_model(
                self.misc_config.empty_init,
                build_model_ctx=self.build_model_ctx,
            )
```

In `build_graph_runner()`, after `self.spec_agent.build_graph_runner()`, add:

```python
            self.memdecode_agent.build_graph_runner()
```

In `build_cache_engine()`, remove `self.memory_cache_engine` and `self.memory_state_cache_engine` creation. After `self.spec_agent.build_cache_engine(self.cache_stream)`, add:

```python
            self.memdecode_agent.set_cache_config(self.cache_config)
            self.memdecode_agent.build_cache_engine(self.cache_stream)
```

In `_forward_impl()`, call `model_forward()` without memory cache arguments:

```python
        output = model_forward(
            self.patched_model,
            inputs,
            self.cache_engine,
            state_cache_engine=self.state_cache_engine,
            stream=self.stream,
        )
```

- [ ] **Step 5: Fuse sliced logits in `_async_model_forward()`**

Replace `_async_model_forward()` with this shape:

```python
    async def _async_model_forward(self, inputs: ModelInputs, return_logits: bool):
        """Model forward."""
        if self.memdecode_agent.is_enabled():
            if return_logits and inputs.is_chunk:
                raise RuntimeError('MemDecode does not support full-prompt returned logits for chunked prefill.')

            base_output = await self.async_forward(inputs)
            base_output = self._postprocess_forward_output(base_output, inputs)
            base_hidden_states = base_output['hidden_states']
            base_logits = self.get_logits(base_hidden_states)

            memory_output = await self.memdecode_agent.async_forward(inputs)
            memory_output = self._postprocess_forward_output(memory_output, inputs)
            memory_hidden_states = memory_output['hidden_states']
            memory_logits = self.memdecode_agent.get_logits(memory_hidden_states)

            fused_logits, routed = self.memdecode_fusion(
                base_logits=base_logits,
                memory_logits=memory_logits,
                base_hidden_states=base_hidden_states,
                memory_hidden_states=memory_hidden_states,
            )
            base_output['logits'] = fused_logits
            if routed is not None:
                base_output['all_routed_experts'] = routed
            return base_output

        ret = await self.async_forward(inputs)

        if not return_logits:
            ret = self._postprocess_forward_output(ret, inputs)

        hidden_states, ret = self.spec_agent.update_main_model_outputs(ret, inputs)

        logits = self.get_logits(hidden_states)
        ret['logits'] = logits
        return ret
```

This intentionally computes MemDecode logits only after `_postprocess_forward_output()` slices hidden states.

- [ ] **Step 6: Reject sleep/wakeup and release memory resources**

At the top of `sleep()` and `wakeup()`, add:

```python
        if self.memdecode_agent.is_enabled():
            raise NotImplementedError('MemDecode sleep/wakeup is not supported yet.')
```

In `reset_graph_runner()`, after `self.spec_agent.reset_graph_runner()`, add:

```python
            self.memdecode_agent.reset_graph_runner()
```

In `release()`, add:

```python
        self.memdecode_agent.release()
```

Remove assignments to `self.memory_cache_engine` and `self.memory_state_cache_engine` from `__init__`, `sleep()`, `release()`, and any other local lifecycle branch.

- [ ] **Step 7: Run model-agent tests**

Run:

```bash
pytest tests/pytorch/engine/test_model_agent.py::TestMemDecodeModelAgentLifecycle tests/pytorch/engine/test_model_agent.py::TestResetGraphRunner -q
```

Expected: PASS.

- [ ] **Step 8: Commit BaseModelAgent integration**

```bash
git add lmdeploy/pytorch/engine/model_agent/agent.py tests/pytorch/engine/test_model_agent.py
git commit -m "feat: wire memdecode agent into model agent"
```

## Task 5: Update Executor Cache Sizing And Validation

**Files:**

- Modify: `lmdeploy/pytorch/engine/executor/base.py`

- Modify: `tests/pytorch/engine/test_executor_base.py`

- [ ] **Step 1: Write cache sizing tests**

Add `torch` to the top-level imports in `tests/pytorch/engine/test_executor_base.py`:

```python
import torch
```

Append these tests to `tests/pytorch/engine/test_executor_base.py`:

```python
def test_get_cache_block_sizes_includes_memdecode_memory(monkeypatch):
    executor = object.__new__(ExecutorBase)
    executor.dist_config = SimpleNamespace(attn_tp=4)
    executor.cache_config = object()
    memory_model_config = object()
    executor.model_config = SimpleNamespace(memdecode_config=SimpleNamespace(memory_model_config=memory_model_config))
    executor.specdecode_config = None
    calls = []

    def fake_get_cache_block_size(cache_config, model_config, world_size):
        calls.append((model_config, world_size))
        return 512 if model_config is executor.model_config else 128

    monkeypatch.setattr(CacheEngine, 'get_cache_block_size', fake_get_cache_block_size)

    cache_block_size = executor._get_cache_block_sizes(None, None)

    assert cache_block_size == _CacheBlockSize(target=512, spec=0, memory=128)
    assert calls == [(executor.model_config, 4), (memory_model_config, 4)]


def test_update_configs_rejects_memdecode_and_specdecode_together():
    executor = object.__new__(ExecutorBase)
    executor.model_config = SimpleNamespace(memdecode_config=SimpleNamespace(memory_model_config=object()),
                                            states_shapes=[])
    executor.specdecode_config = SimpleNamespace()

    with pytest.raises(ValueError, match='MemDecode and speculative decoding cannot be enabled together'):
        executor._validate_memdecode_configs()


def test_update_configs_rejects_memdecode_ssm_mismatch():
    executor = object.__new__(ExecutorBase)
    memory_config = SimpleNamespace(states_shapes=[])
    executor.model_config = SimpleNamespace(
        states_shapes=[((1, 2), torch.float16)],
        memdecode_config=SimpleNamespace(memory_model_config=memory_config),
    )
    executor.specdecode_config = None

    with pytest.raises(ValueError, match='Base and memory model must both use SSM state caches'):
        executor._validate_memdecode_configs()
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest tests/pytorch/engine/test_executor_base.py -q
```

Expected: FAIL because executor still reads `model_config.memory_model_config` and lacks `_validate_memdecode_configs()`.

- [ ] **Step 3: Add executor helpers**

In `lmdeploy/pytorch/engine/executor/base.py`, add:

```python
    def _get_memdecode_config(self):
        return getattr(self.model_config, 'memdecode_config', None)

    def _get_memory_model_config(self):
        memdecode_config = self._get_memdecode_config()
        if memdecode_config is None:
            return None
        return memdecode_config.memory_model_config

    def _validate_memdecode_configs(self):
        memdecode_config = self._get_memdecode_config()
        if memdecode_config is None:
            return
        if self.specdecode_config is not None:
            raise ValueError('MemDecode and speculative decoding cannot be enabled together.')
        memory_model_config = memdecode_config.memory_model_config
        base_has_state = len(self.model_config.states_shapes) > 0
        memory_has_state = len(memory_model_config.states_shapes) > 0
        if base_has_state != memory_has_state:
            raise ValueError('Base and memory model must both use SSM state caches or both not use them.')
```

Update `_get_mem_state_cache_mem()`:

```python
        mem_model_config = self._get_memory_model_config()
        if mem_model_config is None or len(mem_model_config.states_shapes) == 0:
            return 0
```

Update `_get_cache_block_sizes()`:

```python
        memory_cache_block_size = 0
        memory_model_config = self._get_memory_model_config()
        if memory_model_config is not None:
            memory_cache_block_size = CacheEngine.get_cache_block_size(
                self.cache_config,
                memory_model_config,
                self.dist_config.attn_tp,
            )
```

At the start of `update_configs()`, after `_sync_spec_cache_block_size()`, add:

```python
        self._validate_memdecode_configs()
```

Update logging in `init()`:

```python
        if self._get_memdecode_config() is not None:
            logger.info('Building MemDecode memory KV/state cache engines.')
```

- [ ] **Step 4: Run executor tests**

Run:

```bash
pytest tests/pytorch/engine/test_executor_base.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit executor sizing**

```bash
git add lmdeploy/pytorch/engine/executor/base.py tests/pytorch/engine/test_executor_base.py
git commit -m "feat: size memdecode cache from nested config"
```

## Task 6: Remove Wrapper Entry Point And Update Example

**Files:**

- Modify: `lmdeploy/pytorch/models/module_map.py`

- Delete or retire: `lmdeploy/pytorch/models/memdecode.py`

- Modify: `examples/memdec/main.py`

- [ ] **Step 1: Remove architecture override from smoke example**

In `examples/memdec/main.py`, remove:

```python
ARCH_NAME = 'MemDecodeForCausalLM'
```

Remove the `--arch-name` argument.

Change `build_hf_overrides()` to:

```python
def build_hf_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        'memory_model_path': args.memory_model_path,
        'lambda_value': args.lambda_value,
        'adaptive_router': False,
    }
    if args.mode == 'adaptive':
        overrides['router_path'] = args.router_path
        overrides['adaptive_router'] = True
        overrides['lambda_base_only_threshold'] = args.lambda_base_only_threshold
    return overrides
```

Update the module docstring to say MemDecode is enabled by `memory_model_path`, not by overriding HF `architectures`.

- [ ] **Step 2: Remove active module map entry**

In `lmdeploy/pytorch/models/module_map.py`, delete:

```python
MODULE_MAP.update({'MemDecodeForCausalLM': f'{LMDEPLOY_PYTORCH_MODEL_PATH}.memdecode.MemDecodeForCausalLM'})
```

- [ ] **Step 3: Retire old wrapper file**

Delete `lmdeploy/pytorch/models/memdecode.py` if no code imports it:

```bash
git rm lmdeploy/pytorch/models/memdecode.py
```

If deleting causes import failures from stale references, replace the file with this explicit failure:

```python
# Copyright (c) OpenMMLab. All rights reserved.
"""Deprecated MemDecode model wrapper.

MemDecode v4 is implemented through lmdeploy.pytorch.memdecode.MemDecodeAgent.
"""


class MemDecodeForCausalLM:

    def __init__(self, *args, **kwargs):
        raise RuntimeError('MemDecodeForCausalLM is retired. Use memory_model_path MemDecode config instead.')
```

Prefer deletion when `rg "MemDecodeForCausalLM|models.memdecode" lmdeploy tests examples` shows only documentation or removed example text.

- [ ] **Step 4: Run import and example syntax checks**

Run:

```bash
python -m py_compile examples/memdec/main.py
python -m py_compile lmdeploy/pytorch/memdecode/fusion.py lmdeploy/pytorch/memdecode/agent.py
rg -n "MemDecodeForCausalLM|architectures.*MemDecode" lmdeploy examples tests
```

Expected: `py_compile` succeeds. `rg` returns no active code path that configures `MemDecodeForCausalLM`.

- [ ] **Step 5: Commit wrapper removal**

```bash
git add examples/memdec/main.py lmdeploy/pytorch/models/module_map.py lmdeploy/pytorch/models/memdecode.py
git commit -m "refactor: remove memdecode model wrapper entrypoint"
```

Use `git add -u` if `lmdeploy/pytorch/models/memdecode.py` was deleted.

## Task 7: Add End-To-End Fixed-Fusion Smoke Guard

**Files:**

- Create: `tests/pytorch/memdecode/test_model_agent_flow.py`

- [ ] **Step 1: Add a lightweight sliced-logits flow test**

Create `tests/pytorch/memdecode/test_model_agent_flow.py` with:

```python
import asyncio
from types import SimpleNamespace

import torch

from lmdeploy.pytorch.config import MemDecodeConfig, ModelConfig
from lmdeploy.pytorch.engine.model_agent.agent import BaseModelAgent
from lmdeploy.pytorch.memdecode.fusion import MemDecodeFusion


def _model_config():
    return ModelConfig(
        hidden_size=2,
        num_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=[2],
        head_dim=2,
        vocab_size=3,
    )


class _BaseModel:

    def get_logits(self, hidden_states):
        return hidden_states @ torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]])


class _MemoryAgent:

    def __init__(self):
        self.calls = 0

    def is_enabled(self):
        return True

    async def async_forward(self, inputs):
        self.calls += 1
        return {'hidden_states': torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]), 'seq_length': inputs.seq_length}

    def get_logits(self, hidden_states):
        return hidden_states @ torch.tensor([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0]])


def test_memdecode_forward_slices_hidden_states_before_logits():
    base_config = _model_config()
    memory_config = _model_config()
    memdecode_config = MemDecodeConfig(
        memory_model_path='memory',
        memory_model_config=memory_config,
        lambda_value=0.0,
    )

    agent = BaseModelAgent.__new__(BaseModelAgent)
    agent.memdecode_agent = _MemoryAgent()
    agent.memdecode_fusion = MemDecodeFusion(memdecode_config, 2, 2, base_vocab_size=3)
    agent.patched_model = _BaseModel()
    agent.agent_strategy = SimpleNamespace(slice_outputs=lambda hidden, seq_length: hidden[-1:])

    async def _base_forward(inputs):
        return {'hidden_states': torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]), 'seq_length': inputs.seq_length}

    agent.async_forward = _base_forward
    agent.get_logits = agent.patched_model.get_logits

    inputs = SimpleNamespace(seq_length=torch.tensor([2]), is_chunk=False)

    output = asyncio.run(agent._async_model_forward(inputs, return_logits=False))

    assert agent.memdecode_agent.calls == 1
    assert output['logits'].shape == (1, 1, 3)
```

- [ ] **Step 2: Run flow test**

Run:

```bash
pytest tests/pytorch/memdecode/test_model_agent_flow.py -q
```

Expected: PASS. This proves the fixed-fusion path calls the memory agent and only gets logits after slicing.

- [ ] **Step 3: Run MemDecode-focused test group**

Run:

```bash
pytest tests/pytorch/config/test_memdecode_config.py tests/pytorch/memdecode tests/pytorch/engine/test_executor_base.py::test_get_cache_block_sizes_includes_memdecode_memory -q
```

Expected: PASS.

- [ ] **Step 4: Commit smoke guard**

```bash
git add tests/pytorch/memdecode/test_model_agent_flow.py
git commit -m "test: cover memdecode fixed fusion flow"
```

## Task 8: Final Verification

**Files:**

- No source edits expected.

- [ ] **Step 1: Run format and whitespace checks**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
pytest tests/pytorch/config/test_memdecode_config.py tests/pytorch/memdecode tests/pytorch/engine/test_executor_base.py tests/pytorch/engine/test_model_agent.py::TestMemDecodeModelAgentLifecycle -q
```

Expected: PASS.

- [ ] **Step 3: Compile touched Python modules**

Run:

```bash
python -m py_compile \
  lmdeploy/pytorch/config.py \
  lmdeploy/pytorch/memdecode/__init__.py \
  lmdeploy/pytorch/memdecode/fusion.py \
  lmdeploy/pytorch/memdecode/agent.py \
  lmdeploy/pytorch/engine/model_agent/agent.py \
  lmdeploy/pytorch/engine/executor/base.py \
  examples/memdec/main.py
```

Expected: command exits 0.

- [ ] **Step 4: Confirm wrapper references are gone from active code**

Run:

```bash
rg -n "MemDecodeForCausalLM|memory_model_config|memory_model_path|lambda_value|adaptive_router|lambda_base_only_threshold" lmdeploy/pytorch examples/memdec/main.py
```

Expected: `MemDecodeForCausalLM` has no active module-map or example override references. MemDecode config references point through `memdecode_config` internally, except flat external override parsing in `ModelConfig.from_pretrained()` and example CLI JSON.

- [ ] **Step 5: Final commit if verification required cleanup**

If Step 1 through Step 4 required small cleanup edits, commit them:

```bash
git add \
  lmdeploy/pytorch/config.py \
  lmdeploy/pytorch/memdecode/__init__.py \
  lmdeploy/pytorch/memdecode/fusion.py \
  lmdeploy/pytorch/memdecode/agent.py \
  lmdeploy/pytorch/engine/model_agent/agent.py \
  lmdeploy/pytorch/engine/executor/base.py \
  lmdeploy/pytorch/models/module_map.py \
  examples/memdec/main.py \
  tests/pytorch/config/test_memdecode_config.py \
  tests/pytorch/memdecode \
  tests/pytorch/engine/test_executor_base.py \
  tests/pytorch/engine/test_model_agent.py
git commit -m "chore: finalize memdecode v4 wiring"
```

If no cleanup edits were needed, do not create an empty commit.
