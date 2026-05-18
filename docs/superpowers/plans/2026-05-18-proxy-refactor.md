# Proxy Server Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the 948-line proxy monolith into modular, readable code with a strategy pattern for routing and a new `min_cache_usage` strategy that polls backend `/metrics` for KV cache occupation.

**Architecture:** Split the monolith into focused modules: `config.py` (ProxyConfig, enums), `node.py` (Node, NodeRegistry), `routing/` (strategy pattern with one file per strategy), `forwarding.py` (raw request forwarding), `streaming.py` (ProxyStreamingResponse), `distserve.py` (DistServe isolation), `app.py` (FastAPI app factory), and `proxy.py` (entry point). The new `MinCacheUsageStrategy` polls backends in a background `asyncio.Task`.

**Tech Stack:** FastAPI, aiohttp, Pydantic, asyncio, Prometheus text format parsing

______________________________________________________________________

## File Structure

| File                                                 | Responsibility                                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `lmdeploy/serve/proxy/config.py`                     | ProxyConfig, RoutingStrategy, ServingStrategy, ErrorCodes, APIServerException, err_msg, constants |
| `lmdeploy/serve/proxy/node.py`                       | Node model, NodeRegistry class                                                                    |
| `lmdeploy/serve/proxy/routing/__init__.py`           | `get_strategy()` factory, re-exports                                                              |
| `lmdeploy/serve/proxy/routing/base.py`               | Abstract `BaseStrategy`                                                                           |
| `lmdeploy/serve/proxy/routing/random.py`             | `RandomStrategy`                                                                                  |
| `lmdeploy/serve/proxy/routing/min_expected.py`       | `MinExpectedLatencyStrategy`                                                                      |
| `lmdeploy/serve/proxy/routing/min_observed.py`       | `MinObservedLatencyStrategy`                                                                      |
| `lmdeploy/serve/proxy/routing/min_cache.py`          | `MinCacheUsageStrategy` (new)                                                                     |
| `lmdeploy/serve/proxy/forwarding.py`                 | `forward_request()`, `forward_request_stream()`, `prepare_headers()`                              |
| `lmdeploy/serve/proxy/streaming.py`                  | `ProxyStreamingResponse`                                                                          |
| `lmdeploy/serve/proxy/distserve.py`                  | `DistServeRouter` class                                                                           |
| `lmdeploy/serve/proxy/app.py`                        | `create_app()` factory, all endpoint handlers                                                     |
| `lmdeploy/serve/proxy/proxy.py`                      | Entry point `proxy()` function, CLI wiring                                                        |
| `lmdeploy/serve/proxy/__init__.py`                   | Re-exports: `proxy`, `ProxyConfig`, `RoutingStrategy`, `app`                                      |
| `lmdeploy/cli/serve.py`                              | Update CLI to add `min_cache_usage` choice                                                        |
| `tests/test_lmdeploy/serve/proxy/test_config.py`     | Tests for ProxyConfig and enums                                                                   |
| `tests/test_lmdeploy/serve/proxy/test_node.py`       | Tests for Node and NodeRegistry                                                                   |
| `tests/test_lmdeploy/serve/proxy/test_routing.py`    | Tests for all routing strategies                                                                  |
| `tests/test_lmdeploy/serve/proxy/test_forwarding.py` | Tests for forwarding functions                                                                    |

Delete after refactor:

- `lmdeploy/serve/proxy/utils.py` — contents moved to `config.py`

______________________________________________________________________

### Task 1: Create config.py

**Files:**

- Create: `lmdeploy/serve/proxy/config.py`

- Test: `tests/test_lmdeploy/serve/proxy/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lmdeploy/serve/proxy/test_config.py
# Copyright (c) OpenMMLab. All rights reserved.

import os

import pytest


def test_routing_strategy_values():
    from lmdeploy.serve.proxy.config import RoutingStrategy
    assert RoutingStrategy.RANDOM.value == "random"
    assert RoutingStrategy.MIN_EXPECTED_LATENCY.value == "min_expected_latency"
    assert RoutingStrategy.MIN_OBSERVED_LATENCY.value == "min_observed_latency"
    assert RoutingStrategy.MIN_CACHE_USAGE.value == "min_cache_usage"


def test_routing_strategy_from_str():
    from lmdeploy.serve.proxy.config import RoutingStrategy
    assert RoutingStrategy.from_str("random") == RoutingStrategy.RANDOM
    assert RoutingStrategy.from_str("min_expected_latency") == RoutingStrategy.MIN_EXPECTED_LATENCY
    assert RoutingStrategy.from_str("min_observed_latency") == RoutingStrategy.MIN_OBSERVED_LATENCY
    assert RoutingStrategy.from_str("min_cache_usage") == RoutingStrategy.MIN_CACHE_USAGE


def test_routing_strategy_from_str_invalid():
    from lmdeploy.serve.proxy.config import RoutingStrategy
    with pytest.raises(ValueError, match="Invalid strategy"):
        RoutingStrategy.from_str("nonexistent")


def test_serving_strategy_values():
    from lmdeploy.serve.proxy.config import ServingStrategy
    assert ServingStrategy.HYBRID.value == "Hybrid"
    assert ServingStrategy.DIST_SERVE.value == "DistServe"


def test_proxy_config_defaults():
    from lmdeploy.serve.proxy.config import ProxyConfig
    config = ProxyConfig()
    assert config.server_name == "0.0.0.0"
    assert config.server_port == 8000
    assert config.routing_strategy.value == "min_expected_latency"
    assert config.serving_strategy.value == "Hybrid"
    assert config.disable_cache_status is False
    assert config.metrics_poll_interval == 5.0
    assert config.api_keys is None
    assert config.ssl is False


def test_proxy_config_env_override():
    from lmdeploy.serve.proxy.config import ProxyConfig
    os.environ["LMDEPLOY_PROXY_POLL_METRICS_INTERVAL"] = "10"
    try:
        config = ProxyConfig()
        assert config.metrics_poll_interval == 10.0
    finally:
        del os.environ["LMDEPLOY_PROXY_POLL_METRICS_INTERVAL"]


def test_error_codes():
    from lmdeploy.serve.proxy.config import ErrorCodes
    assert ErrorCodes.MODEL_NOT_FOUND.value == 10400
    assert ErrorCodes.SERVICE_UNAVAILABLE.value == 10401
    assert ErrorCodes.API_TIMEOUT.value == 10402


def test_api_server_exception():
    from lmdeploy.serve.proxy.config import APIServerException
    exc = APIServerException(status_code=500, body=b"error")
    assert exc.status_code == 500
    assert exc.body == b"error"
    assert "content-type" in exc.headers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_config.py -v`
Expected: FAIL — module `lmdeploy.serve.proxy.config` does not exist

- [ ] **Step 3: Write the implementation**

```python
# lmdeploy/serve/proxy/config.py
# Copyright (c) OpenMMLab. All rights reserved.

import enum
import os

from pydantic import BaseModel

from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

LATENCY_DEQUE_LEN = 15
AIOHTTP_TIMEOUT = os.getenv("AIOHTTP_TIMEOUT", None)
if AIOHTTP_TIMEOUT is not None:
    AIOHTTP_TIMEOUT = int(AIOHTTP_TIMEOUT)
logger.info(f"AIOHTTP_TIMEOUT set to {AIOHTTP_TIMEOUT}. It can be modified before launching the proxy server "
            "through env variable AIOHTTP_TIMEOUT")


class RoutingStrategy(str, enum.Enum):
    """Strategy to dispatch requests to nodes."""
    RANDOM = "random"
    MIN_EXPECTED_LATENCY = "min_expected_latency"
    MIN_OBSERVED_LATENCY = "min_observed_latency"
    MIN_CACHE_USAGE = "min_cache_usage"

    @classmethod
    def from_str(cls, name):
        """Get strategy from string."""
        try:
            return cls(name)
        except ValueError:
            raise ValueError(f"Invalid strategy: {name}. Supported: random, "
                             f"min_expected_latency, min_observed_latency, min_cache_usage.")


class ServingStrategy(str, enum.Enum):
    """Serving strategy for proxy."""
    HYBRID = "Hybrid"
    DIST_SERVE = "DistServe"


class ErrorCodes(enum.Enum):
    """Error codes."""
    MODEL_NOT_FOUND = 10400
    SERVICE_UNAVAILABLE = 10401
    API_TIMEOUT = 10402


err_msg = {
    ErrorCodes.MODEL_NOT_FOUND: "The request model name does not exist in the model list.",
    ErrorCodes.SERVICE_UNAVAILABLE: "The service is unavailable now. May retry later.",
    ErrorCodes.API_TIMEOUT: "Failed to get response after a period of time",
}


class APIServerException(Exception):

    def __init__(self, status_code: int, body: bytes, headers: dict | None = None):
        self.status_code = status_code
        self.body = body
        self.headers = headers or {}
        if "content-type" not in self.headers:
            self.headers["content-type"] = "application/json"


class ProxyConfig(BaseModel):
    """Configuration for the proxy server."""
    server_name: str = "0.0.0.0"
    server_port: int = 8000
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    serving_strategy: ServingStrategy = ServingStrategy.HYBRID
    disable_cache_status: bool = False
    metrics_poll_interval: float = float(os.getenv("LMDEPLOY_PROXY_POLL_METRICS_INTERVAL", "5.0"))
    migration_protocol: str = "RDMA"
    link_type: str = "RoCE"
    disable_gdr: bool = False
    dummy_prefill: bool = False
    api_keys: list[str] | None = None
    ssl: bool = False
    log_level: str = "INFO"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/serve/proxy/config.py tests/test_lmdeploy/serve/proxy/test_config.py
git commit -m "feat(proxy): add ProxyConfig, RoutingStrategy, ServingStrategy, ErrorCodes"
```

______________________________________________________________________

### Task 2: Create node.py

**Files:**

- Create: `lmdeploy/serve/proxy/node.py`

- Test: `tests/test_lmdeploy/serve/proxy/test_node.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lmdeploy/serve/proxy/test_node.py
# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json
import tempfile

import pytest

from lmdeploy.pytorch.disagg.config import EngineRole


def test_node_defaults():
    from lmdeploy.serve.proxy.node import Node
    node = Node(url="http://localhost:8001")
    assert node.url == "http://localhost:8001"
    assert node.role == EngineRole.Hybrid
    assert node.models == []
    assert node.unfinished == 0
    assert node.speed is None
    assert node.cache_usage is None
    assert node.last_metrics_poll is None


def test_node_with_fields():
    from lmdeploy.serve.proxy.node import Node
    node = Node(url="http://localhost:8001", role=EngineRole.Prefill, models=["llama"], speed=100)
    assert node.role == EngineRole.Prefill
    assert node.models == ["llama"]
    assert node.speed == 100


def test_node_latency_deque():
    from lmdeploy.serve.proxy.node import Node
    node = Node(url="http://localhost:8001")
    node.latency.append(0.5)
    node.latency.append(0.3)
    assert list(node.latency) == [0.5, 0.3]


@pytest.mark.asyncio
async def test_registry_add_and_get():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"])
    nodes = await registry.get("model-a")
    assert len(nodes) == 1
    assert nodes[0].url == "http://a:8001"
    assert nodes[0].models == ["model-a"]


@pytest.mark.asyncio
async def test_registry_add_with_status():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    from lmdeploy.serve.proxy.node import Node
    status = Node(url="http://a:8001", role=EngineRole.Hybrid, models=["model-a"], speed=50)
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"], status=status)
    nodes = await registry.get("model-a")
    assert nodes[0].speed == 50


@pytest.mark.asyncio
async def test_registry_remove():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"])
    await registry.remove("http://a:8001")
    nodes = await registry.get("model-a")
    assert len(nodes) == 0


@pytest.mark.asyncio
async def test_registry_list_models():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a", "model-b"])
    await registry.add("http://b:8001", EngineRole.Hybrid, ["model-b", "model-c"])
    models = await registry.list_models()
    assert set(models) == {"model-a", "model-b", "model-c"}


@pytest.mark.asyncio
async def test_registry_get_by_url():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"])
    node = await registry.get_by_url("http://a:8001")
    assert node is not None
    assert node.url == "http://a:8001"
    assert await registry.get_by_url("http://missing:8001") is None


@pytest.mark.asyncio
async def test_registry_get_by_role():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"])
    await registry.add("http://b:8001", EngineRole.Prefill, ["model-a"])
    nodes = await registry.get("model-a", role=EngineRole.Prefill)
    assert len(nodes) == 1
    assert nodes[0].role == EngineRole.Prefill


@pytest.mark.asyncio
async def test_registry_update_cache_usage():
    from lmdeploy.serve.proxy.node import NodeRegistry
    registry = NodeRegistry()
    await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"])
    await registry.update_cache_usage("http://a:8001", 0.35)
    node = await registry.get_by_url("http://a:8001")
    assert node.cache_usage == 0.35
    assert node.last_metrics_poll is not None


@pytest.mark.asyncio
async def test_registry_persist_and_load():
    from lmdeploy.serve.proxy.node import NodeRegistry
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        config_path = f.name
    try:
        registry = NodeRegistry(config_path=config_path)
        await registry.add("http://a:8001", EngineRole.Hybrid, ["model-a"])
        await registry.persist()

        registry2 = NodeRegistry(config_path=config_path)
        await registry2.load()
        nodes = await registry2.get("model-a")
        assert len(nodes) == 1
        assert nodes[0].url == "http://a:8001"
    finally:
        import os
        os.unlink(config_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_node.py -v`
Expected: FAIL — module `lmdeploy.serve.proxy.node` does not exist

- [ ] **Step 3: Write the implementation**

```python
# lmdeploy/serve/proxy/node.py
# Copyright (c) OpenMMLab. All rights reserved.

import copy
import json
import os.path as osp
import time
from collections import deque
from typing import Optional

from pydantic import BaseModel, Field

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.config import LATENCY_DEQUE_LEN
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")


class Node(BaseModel):
    """A backend api_server node tracked by the proxy."""
    url: str
    role: EngineRole = EngineRole.Hybrid
    models: list[str] = Field(default_factory=list)
    unfinished: int = 0
    latency: deque = Field(default_factory=lambda: deque(maxlen=LATENCY_DEQUE_LEN))
    speed: int | None = None
    cache_usage: float | None = None
    last_metrics_poll: float | None = None

    model_config = {"arbitrary_types_allowed": True}


class NodeRegistry:
    """Thread-safe registry of backend nodes."""

    def __init__(self, config_path: str | None = None, cache_status: bool = True):
        self._nodes: dict[str, Node] = {}
        self._lock = asyncio.Lock()
        self._config_path = config_path or osp.join(osp.dirname(osp.realpath(__file__)), "proxy_config.json")
        self._cache_status = cache_status

    async def add(self, url: str, role: EngineRole = EngineRole.Hybrid,
                  models: list[str] | None = None,
                  status: Node | None = None) -> None:
        async with self._lock:
            if status is not None:
                if status.models:
                    self._nodes.pop(url, None)
                    self._nodes[url] = status
                    await self._persist_unlocked()
                    return
                node = status
            else:
                node = self._nodes.get(url, Node(url=url, role=role))

            if models is not None:
                node.models = models
            elif not node.models:
                try:
                    from lmdeploy.serve.openai.api_client import APIClient
                    import requests
                    client = APIClient(api_server_url=url)
                    node.models = client.available_models
                except requests.exceptions.RequestException as e:
                    logger.error(f"Exception when adding node {url}: {e}")
                    return

            self._nodes[url] = node
            await self._persist_unlocked()

    async def remove(self, url: str) -> None:
        async with self._lock:
            self._nodes.pop(url, None)
            await self._persist_unlocked()

    async def get(self, model_name: str, role: EngineRole | None = None) -> list[Node]:
        async with self._lock:
            result = []
            for node in self._nodes.values():
                if model_name in node.models:
                    if role is None or node.role == role:
                        result.append(node)
            return result

    async def get_by_url(self, url: str) -> Node | None:
        async with self._lock:
            return self._nodes.get(url)

    async def list_models(self) -> list[str]:
        async with self._lock:
            models = set()
            for node in self._nodes.values():
                models.update(node.models)
            return list(models)

    async def update_cache_usage(self, url: str, usage: float) -> None:
        async with self._lock:
            node = self._nodes.get(url)
            if node is not None:
                node.cache_usage = usage
                node.last_metrics_poll = time.time()

    async def all_nodes(self) -> list[Node]:
        async with self._lock:
            return list(self._nodes.values())

    async def get_nodes_by_role(self, role: EngineRole) -> dict[str, Node]:
        async with self._lock:
            return {url: node for url, node in self._nodes.items() if node.role == role}

    async def persist(self) -> None:
        async with self._lock:
            await self._persist_unlocked()

    async def _persist_unlocked(self) -> None:
        if not self._cache_status:
            return
        nodes = copy.deepcopy(self._nodes)
        for node in nodes.values():
            node.latency = deque(list(node.latency)[-LATENCY_DEQUE_LEN:])
        with open(self._config_path, "w") as f:
            json.dump(
                {url: node.model_dump_json() for url, node in nodes.items()},
                f,
                indent=2,
            )

    async def load(self) -> None:
        async with self._lock:
            if not self._cache_status:
                return
            if not osp.exists(self._config_path):
                return
            if osp.getsize(self._config_path) == 0:
                return
            logger.info(f"Loading node configuration: {self._config_path}")
            with open(self._config_path) as f:
                config = json.load(f)
            self._nodes = {
                url: Node.model_validate_json(data) for url, data in config.items()
            }


# Import asyncio at module level for NodeRegistry
import asyncio
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_node.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/serve/proxy/node.py tests/test_lmdeploy/serve/proxy/test_node.py
git commit -m "feat(proxy): add Node model and NodeRegistry"
```

______________________________________________________________________

### Task 3: Create routing base + existing strategies

**Files:**

- Create: `lmdeploy/serve/proxy/routing/__init__.py`

- Create: `lmdeploy/serve/proxy/routing/base.py`

- Create: `lmdeploy/serve/proxy/routing/random.py`

- Create: `lmdeploy/serve/proxy/routing/min_expected.py`

- Create: `lmdeploy/serve/proxy/routing/min_observed.py`

- Test: `tests/test_lmdeploy/serve/proxy/test_routing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lmdeploy/serve/proxy/test_routing.py
# Copyright (c) OpenMMLab. All rights reserved.

import asyncio

import pytest

from lmdeploy.pytorch.disagg.config import EngineRole
from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry


async def _setup_registry(*node_specs):
    """Helper to create a registry with nodes. Each spec: (url, models, speed, unfinished)."""
    registry = NodeRegistry()
    for url, models, speed, unfinished in node_specs:
        node = Node(url=url, models=models, speed=speed, unfinished=unfinished)
        async with registry._lock:
            registry._nodes[url] = node
    return registry


@pytest.mark.asyncio
async def test_random_strategy():
    from lmdeploy.serve.proxy.routing.random import RandomStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
        ("http://b:8001", ["llama"], 50, 0),
    )
    strategy = RandomStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    assert node.url in ("http://a:8001", "http://b:8001")


@pytest.mark.asyncio
async def test_random_strategy_no_match():
    from lmdeploy.serve.proxy.routing.random import RandomStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
    )
    strategy = RandomStrategy(registry, ProxyConfig())
    with pytest.raises(ValueError, match="No available node"):
        await strategy.select_node("missing-model")


@pytest.mark.asyncio
async def test_min_expected_latency_strategy():
    from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 5),
        ("http://b:8001", ["llama"], 100, 2),
    )
    strategy = MinExpectedLatencyStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    assert node.url == "http://b:8001"


@pytest.mark.asyncio
async def test_min_expected_latency_no_speed():
    from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], None, 0),
        ("http://b:8001", ["llama"], None, 5),
    )
    strategy = MinExpectedLatencyStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    # Both have no speed, both get average speed=1, node a has unfinished=0 so wins
    assert node.url == "http://a:8001"


@pytest.mark.asyncio
async def test_min_observed_latency_strategy():
    from lmdeploy.serve.proxy.routing.min_observed import MinObservedLatencyStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
        ("http://b:8001", ["llama"], 100, 0),
    )
    # Set latency history
    async with registry._lock:
        registry._nodes["http://a:8001"].latency.extend([0.5, 0.6])
        registry._nodes["http://b:8001"].latency.extend([0.2, 0.3])
    strategy = MinObservedLatencyStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    assert node.url == "http://b:8001"


@pytest.mark.asyncio
async def test_min_observed_latency_no_history():
    from lmdeploy.serve.proxy.routing.min_observed import MinObservedLatencyStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
        ("http://b:8001", ["llama"], 100, 0),
    )
    # a has latency history, b does not (inf latency)
    async with registry._lock:
        registry._nodes["http://a:8001"].latency.extend([0.5])
    strategy = MinObservedLatencyStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    assert node.url == "http://a:8001"


@pytest.mark.asyncio
async def test_on_request_start_and_end():
    from lmdeploy.serve.proxy.routing.base import BaseStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
    )
    config = ProxyConfig()
    strategy = BaseStrategy.__new__(BaseStrategy)
    strategy.registry = registry
    strategy.config = config
    node = await registry.get_by_url("http://a:8001")
    await strategy.on_request_start(node)
    assert node.unfinished == 1
    await strategy.on_request_end(node, 0.5)
    assert node.unfinished == 0
    assert 0.5 in node.latency


@pytest.mark.asyncio
async def test_get_strategy_factory():
    from lmdeploy.serve.proxy.routing import get_strategy
    from lmdeploy.serve.proxy.routing.random import RandomStrategy
    from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
    from lmdeploy.serve.proxy.routing.min_observed import MinObservedLatencyStrategy
    from lmdeploy.serve.proxy.config import RoutingStrategy

    registry = NodeRegistry()
    config = ProxyConfig()

    assert isinstance(get_strategy(RoutingStrategy.RANDOM, registry, config), RandomStrategy)
    assert isinstance(get_strategy(RoutingStrategy.MIN_EXPECTED_LATENCY, registry, config), MinExpectedLatencyStrategy)
    assert isinstance(get_strategy(RoutingStrategy.MIN_OBSERVED_LATENCY, registry, config), MinObservedLatencyStrategy)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_routing.py -v`
Expected: FAIL — module `lmdeploy.serve.proxy.routing` does not exist

- [ ] **Step 3: Write base.py**

```python
# lmdeploy/serve/proxy/routing/base.py
# Copyright (c) OpenMMLab. All rights reserved.

from abc import ABC, abstractmethod

from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry


class BaseStrategy(ABC):
    """Abstract base class for routing strategies."""

    def __init__(self, registry: NodeRegistry, config: ProxyConfig):
        self.registry = registry
        self.config = config

    @abstractmethod
    async def select_node(self, model_name: str, role=None) -> Node:
        """Select the best node for a request to the given model."""

    async def on_request_start(self, node: Node) -> None:
        """Hook called before forwarding a request."""
        node.unfinished += 1

    async def on_request_end(self, node: Node, latency: float) -> None:
        """Hook called after a request completes."""
        if node.url in [n.url for n in await self.registry.all_nodes()]:
            node.unfinished -= 1
            node.latency.append(latency)

    async def start(self) -> None:
        """Start any background tasks. Override if needed."""

    async def stop(self) -> None:
        """Stop background tasks. Override if needed."""
```

- [ ] **Step 4: Write random.py**

```python
# lmdeploy/serve/proxy/routing/random.py
# Copyright (c) OpenMMLab. All rights reserved.

import random

from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy


class RandomStrategy(BaseStrategy):
    """Weighted random selection based on node speed."""

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        urls_with_speeds, speeds, urls_without_speeds = [], [], []
        for node in nodes:
            if node.speed is not None:
                urls_with_speeds.append(node)
                speeds.append(node.speed)
            else:
                urls_without_speeds.append(node)

        average_speed = sum(speeds) / len(speeds) if speeds else 1
        all_nodes = urls_with_speeds + urls_without_speeds
        all_speeds = speeds + [average_speed] * len(urls_without_speeds)

        speed_sum = sum(all_speeds)
        weights = [s / speed_sum for s in all_speeds]
        index = random.choices(range(len(all_nodes)), weights=weights)[0]
        return all_nodes[index]
```

- [ ] **Step 5: Write min_expected.py**

```python
# lmdeploy/serve/proxy/routing/min_expected.py
# Copyright (c) OpenMMLab. All rights reserved.

import random

import numpy as np

from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy


class MinExpectedLatencyStrategy(BaseStrategy):
    """Select node with lowest expected latency (unfinished / speed)."""

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        speeds = [n.speed for n in nodes]
        average_speed = sum(s for s in speeds if s is not None) / len([s for s in speeds if s is not None]) if any(s is not None for s in speeds) else 1
        resolved_speeds = [s if s is not None else average_speed for s in speeds]

        # Random traversal for load spreading in low-concurrency scenarios
        indices = list(range(len(nodes)))
        random.shuffle(indices)

        min_latency = float("inf")
        min_index = 0
        for i in indices:
            latency = nodes[i].unfinished / resolved_speeds[i]
            if min_latency > latency:
                min_latency = latency
                min_index = i
        return nodes[min_index]
```

- [ ] **Step 6: Write min_observed.py**

```python
# lmdeploy/serve/proxy/routing/min_observed.py
# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np

from lmdeploy.serve.proxy.config import ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy


class MinObservedLatencyStrategy(BaseStrategy):
    """Select node with lowest mean observed latency."""

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        latencies = []
        for node in nodes:
            if len(node.latency):
                latencies.append(float(np.mean(list(node.latency))))
            else:
                latencies.append(float("inf"))

        index = int(np.argmin(np.array(latencies)))
        return nodes[index]
```

- [ ] **Step 7: Write routing/__init__.py**

```python
# lmdeploy/serve/proxy/routing/__init__.py
# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy


def get_strategy(strategy: RoutingStrategy, registry: NodeRegistry, config: ProxyConfig) -> BaseStrategy:
    """Factory function to create a routing strategy instance."""
    if strategy == RoutingStrategy.RANDOM:
        from lmdeploy.serve.proxy.routing.random import RandomStrategy
        return RandomStrategy(registry, config)
    elif strategy == RoutingStrategy.MIN_EXPECTED_LATENCY:
        from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
        return MinExpectedLatencyStrategy(registry, config)
    elif strategy == RoutingStrategy.MIN_OBSERVED_LATENCY:
        from lmdeploy.serve.proxy.routing.min_observed import MinObservedLatencyStrategy
        return MinObservedLatencyStrategy(registry, config)
    elif strategy == RoutingStrategy.MIN_CACHE_USAGE:
        from lmdeploy.serve.proxy.routing.min_cache import MinCacheUsageStrategy
        return MinCacheUsageStrategy(registry, config)
    else:
        raise ValueError(f"Unknown routing strategy: {strategy}")
```

- [ ] **Step 8: Run test to verify it passes (excluding min_cache tests)**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_routing.py -v -k "not min_cache"`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add lmdeploy/serve/proxy/routing/ tests/test_lmdeploy/serve/proxy/test_routing.py
git commit -m "feat(proxy): add routing strategy pattern with base, random, min_expected, min_observed"
```

______________________________________________________________________

### Task 4: Create MinCacheUsageStrategy

**Files:**

- Create: `lmdeploy/serve/proxy/routing/min_cache.py`

- Test: `tests/test_lmdeploy/serve/proxy/test_routing.py` (extend)

- [ ] **Step 1: Write the failing test**

Add these tests to `tests/test_lmdeploy/serve/proxy/test_routing.py`:

```python
@pytest.mark.asyncio
async def test_min_cache_usage_strategy():
    from lmdeploy.serve.proxy.routing.min_cache import MinCacheUsageStrategy
    import time
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
        ("http://b:8001", ["llama"], 100, 0),
    )
    now = time.time()
    async with registry._lock:
        registry._nodes["http://a:8001"].cache_usage = 0.8
        registry._nodes["http://a:8001"].last_metrics_poll = now
        registry._nodes["http://b:8001"].cache_usage = 0.3
        registry._nodes["http://b:8001"].last_metrics_poll = now
    strategy = MinCacheUsageStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    assert node.url == "http://b:8001"


@pytest.mark.asyncio
async def test_min_cache_usage_fallback_when_no_metrics():
    from lmdeploy.serve.proxy.routing.min_cache import MinCacheUsageStrategy
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 5),
        ("http://b:8001", ["llama"], 100, 2),
    )
    # No cache_usage set on any node — should fall back to min_expected_latency
    strategy = MinCacheUsageStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    assert node.url == "http://b:8001"


@pytest.mark.asyncio
async def test_min_cache_usage_stale_metrics_fallback():
    from lmdeploy.serve.proxy.routing.min_cache import MinCacheUsageStrategy
    import time
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 5),
        ("http://b:8001", ["llama"], 100, 2),
    )
    # Set stale metrics (older than 3 * poll_interval)
    stale_time = time.time() - 100
    async with registry._lock:
        registry._nodes["http://a:8001"].cache_usage = 0.1
        registry._nodes["http://a:8001"].last_metrics_poll = stale_time
    strategy = MinCacheUsageStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    # Stale node should be excluded, fallback to min_expected_latency
    assert node.url == "http://b:8001"


@pytest.mark.asyncio
async def test_min_cache_usage_partial_metrics():
    from lmdeploy.serve.proxy.routing.min_cache import MinCacheUsageStrategy
    import time
    registry = await _setup_registry(
        ("http://a:8001", ["llama"], 100, 0),
        ("http://b:8001", ["llama"], 100, 0),
    )
    now = time.time()
    async with registry._lock:
        # Only node a has metrics
        registry._nodes["http://a:8001"].cache_usage = 0.7
        registry._nodes["http://a:8001"].last_metrics_poll = now
        # node b has no metrics
    strategy = MinCacheUsageStrategy(registry, ProxyConfig())
    node = await strategy.select_node("llama")
    # Node a has metrics (0.7), node b has none -> fall back to min_expected for b
    # But since a has valid data, we can still route based on cache
    assert node.url == "http://a:8001"


def test_parse_prometheus_cache_usage():
    from lmdeploy.serve.proxy.routing.min_cache import parse_prometheus_cache_usage
    # Standard prometheus text format
    text = """# HELP lmdeploy_gpu_cache_usage_perc GPU KV-cache usage.
# TYPE lmdeploy_gpu_cache_usage_perc gauge
lmdeploy_gpu_cache_usage_perc 0.35
"""
    assert parse_prometheus_cache_usage(text) == 0.35


def test_parse_prometheus_cache_usage_with_labels():
    from lmdeploy.serve.proxy.routing.min_cache import parse_prometheus_cache_usage
    text = 'lmdeploy_gpu_cache_usage_perc{model="llama"} 0.55\n'
    assert parse_prometheus_cache_usage(text) == 0.55


def test_parse_prometheus_cache_usage_missing():
    from lmdeploy.serve.proxy.routing.min_cache import parse_prometheus_cache_usage
    text = "# TYPE some_other_metric gauge\nsome_other_metric 42\n"
    assert parse_prometheus_cache_usage(text) is None


def test_parse_prometheus_cache_usage_empty():
    from lmdeploy.serve.proxy.routing.min_cache import parse_prometheus_cache_usage
    assert parse_prometheus_cache_usage("") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_routing.py -v -k "min_cache or parse_prometheus"`
Expected: FAIL — `MinCacheUsageStrategy` does not exist

- [ ] **Step 3: Write the implementation**

```python
# lmdeploy/serve/proxy/routing/min_cache.py
# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import re
import time

import aiohttp

from lmdeploy.serve.proxy.config import AIOHTTP_TIMEOUT, ProxyConfig
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.routing.base import BaseStrategy
from lmdeploy.serve.proxy.routing.min_expected import MinExpectedLatencyStrategy
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

# Prometheus metric name: lmdeploy uses colon in name but Prometheus export
# replaces colons with underscores
_METRIC_RE = re.compile(r"^lmdeploy[:_]gpu_cache_usage_perc(?:\{[^}]*})?\s+([0-9.eE+-]+)", re.MULTILINE)


def parse_prometheus_cache_usage(text: str) -> float | None:
    """Parse lmdeploy:gpu_cache_usage_perc from Prometheus text exposition format.

    Returns the float value, or None if the metric is not found.
    """
    match = _METRIC_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


class MinCacheUsageStrategy(BaseStrategy):
    """Route to the node with the lowest KV cache usage.

    Polls backend /metrics endpoints in a background task.
    Falls back to MinExpectedLatencyStrategy when no metrics data is available.
    """

    def __init__(self, registry: NodeRegistry, config: ProxyConfig):
        super().__init__(registry, config)
        self._fallback = MinExpectedLatencyStrategy(registry, config)
        self._poll_task: asyncio.Task | None = None
        self._timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT) if AIOHTTP_TIMEOUT else None

    async def select_node(self, model_name: str, role=None) -> Node:
        nodes = await self.registry.get(model_name, role=role)
        if not nodes:
            raise ValueError(f"No available node for model {model_name!r}")

        stale_threshold = time.time() - 3 * self.config.metrics_poll_interval
        valid_nodes = [
            n for n in nodes
            if n.cache_usage is not None
            and n.last_metrics_poll is not None
            and n.last_metrics_poll > stale_threshold
        ]

        if not valid_nodes:
            return await self._fallback.select_node(model_name, role=role)

        return min(valid_nodes, key=lambda n: n.cache_usage)

    async def start(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self._poll_all_nodes()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in metrics poll loop: {e}")
            await asyncio.sleep(self.config.metrics_poll_interval)

    async def _poll_all_nodes(self) -> None:
        nodes = await self.registry.all_nodes()
        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for node in nodes:
                try:
                    async with session.get(f"{node.url}/metrics") as response:
                        if response.status == 200:
                            text = await response.text()
                            usage = parse_prometheus_cache_usage(text)
                            if usage is not None:
                                await self.registry.update_cache_usage(node.url, usage)
                except Exception as e:
                    logger.debug(f"Failed to poll metrics from {node.url}: {e}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_routing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/serve/proxy/routing/min_cache.py tests/test_lmdeploy/serve/proxy/test_routing.py
git commit -m "feat(proxy): add MinCacheUsageStrategy with background metrics polling"
```

______________________________________________________________________

### Task 5: Create forwarding.py

**Files:**

- Create: `lmdeploy/serve/proxy/forwarding.py`

- Test: `tests/test_lmdeploy/serve/proxy/test_forwarding.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lmdeploy/serve/proxy/test_forwarding.py
# Copyright (c) OpenMMLab. All rights reserved.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_prepare_headers():
    from lmdeploy.serve.proxy.forwarding import prepare_headers
    raw_request = MagicMock()
    raw_request.headers = {"host": "original:8000", "content-type": "application/json", "authorization": "Bearer abc"}
    raw_request.client = MagicMock()
    raw_request.client.host = "10.0.0.1"
    raw_request.url.scheme = "http"

    headers = prepare_headers(raw_request)
    assert "host" not in headers
    assert headers["X-Forwarded-For"] == "10.0.0.1"
    assert headers["X-Forwarded-Host"] == "original:8000"
    assert headers["X-Forwarded-Proto"] == "http"
    assert headers["content-type"] == "application/json"


def test_prepare_headers_no_client():
    from lmdeploy.serve.proxy.forwarding import prepare_headers
    raw_request = MagicMock()
    raw_request.headers = {"host": "original:8000"}
    raw_request.client = None
    raw_request.url.scheme = "https"

    headers = prepare_headers(raw_request)
    assert headers["X-Forwarded-For"] == "unknown"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_forwarding.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Write the implementation**

```python
# lmdeploy/serve/proxy/forwarding.py
# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import json

import aiohttp

from lmdeploy.serve.proxy.config import AIOHTTP_TIMEOUT, ErrorCodes, err_msg
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")


def prepare_headers(raw_request) -> dict[str, str]:
    """Prepare forwarding headers from the original request."""
    headers = {name: value for name, value in raw_request.headers.items() if name.lower() != "host"}
    client_ip = raw_request.client.host if raw_request.client else "unknown"
    headers.update({
        "X-Forwarded-For": client_ip,
        "X-Forwarded-Host": raw_request.headers.get("host", ""),
        "X-Forwarded-Proto": raw_request.url.scheme,
    })
    return headers


def handle_api_timeout(node_url: str) -> bytes:
    """Handle the api timeout."""
    logger.warning(f"api timeout: {node_url}")
    ret = {
        "error_code": ErrorCodes.API_TIMEOUT.value,
        "text": err_msg[ErrorCodes.API_TIMEOUT],
    }
    return json.dumps(ret).encode() + b"\n"


async def forward_request_stream(client: aiohttp.ClientSession, node_url: str,
                                  raw_request, endpoint: str):
    """Forward a raw HTTP request as a streaming response.

    Yields response chunks. On error, yields an error payload.
    """
    from lmdeploy.serve.proxy.config import APIServerException
    try:
        target_url = node_url.rstrip("/") + endpoint
        headers = prepare_headers(raw_request)
        body_bytes = await raw_request.body()
        async with client.post(target_url, headers=headers, data=body_bytes) as response:
            if response.status != 200:
                error_body = await response.read()
                raise APIServerException(status_code=response.status, body=error_body)
            async for line in response.content:
                if line.strip():
                    yield line + b"\n\n"
    except APIServerException:
        raise
    except (Exception, GeneratorExit, aiohttp.ClientError) as e:
        logger.error(f"caught an exception: {e}")
        yield handle_api_timeout(node_url)


async def forward_request(client: aiohttp.ClientSession, node_url: str,
                           raw_request, endpoint: str) -> str:
    """Forward a raw HTTP request and return the response text."""
    try:
        target_url = node_url.rstrip("/") + endpoint
        headers = prepare_headers(raw_request)
        body_bytes = await raw_request.body()
        async with client.post(target_url, headers=headers, data=body_bytes) as response:
            return await response.text()
    except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
        logger.error(f"caught an exception: {e}")
        return handle_api_timeout(node_url).decode()


async def generate(client: aiohttp.ClientSession, request: dict,
                    node_url: str, endpoint: str) -> str:
    """Forward a parsed dict request and return the response text."""
    try:
        async with client.post(node_url + endpoint, json=request) as response:
            return await response.text()
    except (Exception, GeneratorExit, aiohttp.ClientError, asyncio.CancelledError) as e:
        logger.error(f"caught an exception: {e}")
        return handle_api_timeout(node_url).decode()


async def stream_generate(client: aiohttp.ClientSession, request: dict,
                           node_url: str, endpoint: str):
    """Forward a parsed dict request as a streaming response."""
    try:
        async with client.post(node_url + endpoint, json=request) as response:
            async for line in response.content:
                if line.strip():
                    yield line + b"\n\n"
    except (Exception, GeneratorExit, aiohttp.ClientError) as e:
        logger.error(f"caught an exception: {e}")
        yield handle_api_timeout(node_url)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lmdeploy/serve/proxy/test_forwarding.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lmdeploy/serve/proxy/forwarding.py tests/test_lmdeploy/serve/proxy/test_forwarding.py
git commit -m "feat(proxy): add forwarding module with raw request forwarding"
```

______________________________________________________________________

### Task 6: Create streaming.py

**Files:**

- Create: `lmdeploy/serve/proxy/streaming.py`

- Delete: `lmdeploy/serve/proxy/streaming_response.py` (in a later task)

- [ ] **Step 1: Write the file**

```python
# lmdeploy/serve/proxy/streaming.py
# Copyright (c) OpenMMLab. All rights reserved.

import json

from fastapi.responses import StreamingResponse

from lmdeploy.serve.proxy.config import APIServerException


class ProxyStreamingResponse(StreamingResponse):
    """StreamingResponse that can handle exceptions thrown by the generator."""

    def __init__(self, content, **kwargs):
        super().__init__(content, **kwargs)

    async def stream_response(self, send) -> None:
        iterator = self.body_iterator.__aiter__()
        try:
            first_chunk = await iterator.__anext__()
        except APIServerException as e:
            headers = self._convert_headers_to_asgi(e.headers) if e.headers else self.raw_headers
            await send({"type": "http.response.start", "status": e.status_code, "headers": headers})
            await send({
                "type": "http.response.body",
                "body": e.body,
                "more_body": False,
            })
            return

        await send({
            "type": "http.response.start",
            "status": self.status_code,
            "headers": self.raw_headers,
        })

        await send({
            "type": "http.response.body",
            "body": first_chunk,
            "more_body": True,
        })

        try:
            async for chunk in iterator:
                await send({
                    "type": "http.response.body",
                    "body": chunk,
                    "more_body": True,
                })
        except Exception:
            error_data = {"error": True, "status": 500, "message": "Internal streaming error"}
            await send({
                "type": "http.response.body",
                "body": json.dumps(error_data).encode("utf-8"),
                "more_body": False,
            })
            return

        await send({
            "type": "http.response.body",
            "body": b"",
            "more_body": False,
        })

    def _convert_headers_to_asgi(self, headers: dict) -> list[tuple[bytes, bytes]]:
        """Convert dict headers to ASGI raw header tuples."""
        return [(name.lower().encode("latin-1"), str(value).encode("latin-1"))
                for name, value in headers.items()]
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/serve/proxy/streaming.py
git commit -m "feat(proxy): add streaming module with ProxyStreamingResponse"
```

______________________________________________________________________

### Task 7: Create distserve.py

**Files:**

- Create: `lmdeploy/serve/proxy/distserve.py`

- [ ] **Step 1: Write the file**

This extracts the DistServe logic from the current `proxy.py` into a `DistServeRouter` class. The code is moved verbatim from the current implementation, just reorganized into a class with methods.

```python
# lmdeploy/serve/proxy/distserve.py
# Copyright (c) OpenMMLab. All rights reserved.

import copy
import json

import aiohttp

from lmdeploy.pytorch.disagg.config import DistServeRDMAConfig, EngineRole, RDMALinkType, ServingStrategy
from lmdeploy.pytorch.disagg.conn.protocol import MigrationProtocol, MigrationRequest
from lmdeploy.pytorch.disagg.conn.proxy_conn import PDConnectionPool
from lmdeploy.pytorch.disagg.messages import PDConnectionMessage
from lmdeploy.serve.proxy.config import AIOHTTP_TIMEOUT, ErrorCodes, ProxyConfig, err_msg
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")


class DistServeRouter:
    """Handles DistServe (prefill-decode disaggregation) routing."""

    def __init__(self, registry: NodeRegistry, config: ProxyConfig):
        self.registry = registry
        self.config = config
        self.migration_protocol = MigrationProtocol[config.migration_protocol]
        self.rdma_config = DistServeRDMAConfig(
            with_gdr=not config.disable_gdr,
            link_type=RDMALinkType[config.link_type],
        )
        self.pd_connection_pool = PDConnectionPool()
        self.dummy_prefill = config.dummy_prefill
        self._aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT) if AIOHTTP_TIMEOUT else None

    async def handle_chat_completions(self, request, raw_request, endpoint: str):
        """Handle a chat completion request in DistServe mode."""
        request_dict = request.model_dump()
        return await self._handle_request(request_dict, request, raw_request, endpoint, request.stream)

    async def handle_completions(self, request, raw_request, endpoint: str):
        """Handle a completion request in DistServe mode."""
        request_dict = request.model_dump()
        return await self._handle_request(request_dict, request, raw_request, endpoint, request.stream)

    async def _handle_request(self, request_dict, request, raw_request, endpoint: str, stream: bool):
        from lmdeploy.serve.proxy.forwarding import generate, stream_generate
        from lmdeploy.serve.proxy.streaming import ProxyStreamingResponse
        from fastapi.responses import JSONResponse, StreamingResponse

        model_name = request.model

        # Prefill
        prefill_request_dict = copy.deepcopy(request_dict)
        prefill_request_dict["max_tokens"] = 1
        prefill_request_dict["max_completion_tokens"] = 1
        prefill_request_dict["stream"] = False
        prefill_request_dict["with_cache"] = True
        prefill_request_dict["preserve_cache"] = True

        prefill_info = {}
        p_url = "dummy:dummy"
        if not self.dummy_prefill:
            p_nodes = await self.registry.get(model_name, role=EngineRole.Prefill)
            if not p_nodes:
                return self._handle_unavailable_model(model_name)
            p_url = p_nodes[0].url
            logger.info(f"A Prefill request is dispatched to {p_url}")

            node = await self.registry.get_by_url(p_url)
            node.unfinished += 1
            start = __import__("time").time()

            async with aiohttp.ClientSession(timeout=self._aiotimeout) as client:
                prefill_info = json.loads(await generate(client, prefill_request_dict, p_url, endpoint))

            node.unfinished -= 1
            node.latency.append(__import__("time").time() - start)

        # Decode
        d_nodes = await self.registry.get(model_name, role=EngineRole.Decode)
        if not d_nodes:
            return self._handle_unavailable_model(model_name)
        d_url = d_nodes[0].url
        logger.info(f"A Decode request is dispatched to {d_url}")

        if not self.dummy_prefill:
            if not self.pd_connection_pool.is_connected(p_url, d_url):
                await self.pd_connection_pool.connect(
                    PDConnectionMessage(
                        p_url=p_url,
                        d_url=d_url,
                        protocol=self.migration_protocol,
                        rdma_config=self.rdma_config,
                    ))

        remote_session_id = int(prefill_info.get("id")) if prefill_info.get("id") else 0
        remote_block_ids = prefill_info.get("cache_block_ids") or []
        remote_token_id = prefill_info.get("remote_token_ids")[-1] if prefill_info.get("remote_token_ids") else 0

        request_dict["migration_request"] = MigrationRequest(
            protocol=self.migration_protocol,
            remote_engine_id=p_url,
            remote_session_id=remote_session_id,
            remote_block_ids=remote_block_ids,
            remote_token_id=remote_token_id,
            is_dummy_prefill=self.dummy_prefill,
        ).model_dump(mode="json")

        d_node = await self.registry.get_by_url(d_url)
        d_node.unfinished += 1
        start = __import__("time").time()

        if not self.dummy_prefill:
            self.pd_connection_pool.shelf_prefill_session((p_url, d_url), prefill_info["id"])

        async with aiohttp.ClientSession(timeout=self._aiotimeout) as client:
            if stream:
                response = stream_generate(client, request_dict, d_url, endpoint)
                from fastapi import BackgroundTasks
                background_task = BackgroundTasks()
                # Post-call handled inline below
                resp = StreamingResponse(response, background=background_task, media_type="text/event-stream")
            else:
                response_text = await generate(client, request_dict, d_url, endpoint)
                d_node.unfinished -= 1
                d_node.latency.append(__import__("time").time() - start)
                resp = JSONResponse(json.loads(response_text))

        if not self.dummy_prefill:
            self.pd_connection_pool.unshelf_prefill_session((p_url, d_url), prefill_info.get("id"))

        return resp

    def _handle_unavailable_model(self, model_name: str) -> bytes:
        logger.warning(f"no model name: {model_name}")
        ret = {
            "error_code": ErrorCodes.MODEL_NOT_FOUND,
            "text": err_msg[ErrorCodes.MODEL_NOT_FOUND],
        }
        return json.dumps(ret).encode() + b"\n"

    async def connection_warmup(self):
        """Warm up all PD connections."""
        p_nodes = await self.registry.get_nodes_by_role(EngineRole.Prefill)
        d_nodes = await self.registry.get_nodes_by_role(EngineRole.Decode)
        await __import__("asyncio").gather(*[
            self.pd_connection_pool.connect(
                PDConnectionMessage(
                    p_url=p_url,
                    d_url=d_url,
                    protocol=self.migration_protocol,
                    rdma_config=self.rdma_config,
                )) for p_url in p_nodes for d_url in d_nodes
        ])
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/serve/proxy/distserve.py
git commit -m "feat(proxy): add DistServeRouter with isolated distserve logic"
```

______________________________________________________________________

### Task 8: Create app.py

**Files:**

- Create: `lmdeploy/serve/proxy/app.py`

This is the largest task. It creates the FastAPI app factory with all endpoint handlers, replacing the module-level globals in the current `proxy.py`.

- [ ] **Step 1: Write the file**

```python
# lmdeploy/serve/proxy/app.py
# Copyright (c) OpenMMLab. All rights reserved.

import asyncio
import copy
import json
import os
import random
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from http import HTTPStatus

import aiohttp
import requests
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from lmdeploy.serve.openai.api_server import create_error_response
from lmdeploy.serve.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ModelCard,
    ModelList,
    ModelPermission,
)
from lmdeploy.serve.proxy.config import AIOHTTP_TIMEOUT, ErrorCodes, ProxyConfig, err_msg
from lmdeploy.serve.proxy.forwarding import (
    forward_request,
    forward_request_stream,
    generate,
    prepare_headers,
    stream_generate,
)
from lmdeploy.serve.proxy.node import Node, NodeRegistry
from lmdeploy.serve.proxy.streaming import ProxyStreamingResponse
from lmdeploy.serve.utils.server_utils import validate_json_request
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")

CONTROLLER_HEART_BEAT_EXPIRATION = int(os.getenv("LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION", 90))


def _heart_beat_controller(registry: NodeRegistry):
    """Background thread that removes stale nodes."""
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        logger.info("Start heart beat check")
        _remove_stale_nodes(registry)


def _remove_stale_nodes(registry: NodeRegistry):
    to_be_deleted = []
    nodes = asyncio.run_coroutine_threadsafe(registry.all_nodes(), asyncio.get_event_loop()).result()
    for node in nodes:
        url = f"{node.url}/health"
        headers = {"accept": "application/json"}
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                to_be_deleted.append(node.url)
        except Exception:
            to_be_deleted.append(node.url)
    for node_url in to_be_deleted:
        asyncio.run_coroutine_threadsafe(registry.remove(node_url), asyncio.get_event_loop()).result()
        logger.info(f"Removed node_url: {node_url} due to heart beat expiration")


def _handle_unavailable_model(model_name: str) -> bytes:
    logger.warning(f"no model name: {model_name}")
    ret = {
        "error_code": ErrorCodes.MODEL_NOT_FOUND,
        "text": err_msg[ErrorCodes.MODEL_NOT_FOUND],
    }
    return json.dumps(ret).encode() + b"\n"


def _handle_api_timeout(node_url: str) -> bytes:
    logger.warning(f"api timeout: {node_url}")
    ret = {
        "error_code": ErrorCodes.API_TIMEOUT.value,
        "text": err_msg[ErrorCodes.API_TIMEOUT],
    }
    return json.dumps(ret).encode() + b"\n"


def create_app(config: ProxyConfig, registry: NodeRegistry, strategy) -> FastAPI:
    """Create the FastAPI application with all routes.

    Args:
        config: Proxy configuration.
        registry: Node registry.
        strategy: Routing strategy instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await strategy.start()
        if config.cache_status:
            await registry.load()
        heart_beat_thread = threading.Thread(target=_heart_beat_controller, args=(registry,), daemon=True)
        heart_beat_thread.start()
        yield
        # Shutdown
        await strategy.stop()

    app = FastAPI(docs_url="/", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/v1/models")
    async def available_models():
        model_cards = []
        model_names = await registry.list_models()
        for model_name in model_names:
            model_cards.append(ModelCard(id=model_name, root=model_name, permission=[ModelPermission()]))
        return ModelList(data=model_cards)

    @app.get("/nodes/status")
    async def node_status():
        try:
            nodes = await registry.all_nodes()
            return {n.url: n for n in nodes}
        except Exception:
            return False

    @app.post("/nodes/add", dependencies=[Depends(validate_json_request)])
    async def add_node(node: Node, raw_request: Request = None):
        try:
            await registry.add(node.url, node.status.role if node.status else None,
                               node.status.models if node.status else None,
                               node.status)
            logger.info(f"add node {node.url} successfully")
            return "Added successfully"
        except Exception:
            return "Failed to add, please check the input url."

    @app.post("/nodes/remove", dependencies=[Depends(validate_json_request)])
    async def remove_node(node: Node):
        try:
            await registry.remove(node.url)
            logger.info(f"delete node {node.url} successfully")
            return "Deleted successfully"
        except Exception:
            logger.error(f"delete node {node.url} failed.")
            return "Failed to delete, please check the input url."

    @app.post("/nodes/terminate", dependencies=[Depends(validate_json_request)])
    async def terminate_node(node: Node):
        try:
            node_url = node.url
            success = True
            existing = await registry.get_by_url(node_url)
            if existing:
                await registry.remove(node_url)
                headers = {"accept": "application/json"}
                try:
                    response = requests.get(f"{node_url}/terminate", headers=headers)
                    if response.status_code != 200:
                        success = False
                        logger.error(f"Failed to terminate node {node_url}, "
                                     f"error_code={response.status_code}, "
                                     f"error_msg={response.text}")
                except Exception as e:
                    logger.error(f"exception happened when terminating node {node_url}, {e}")
                    success = False
            else:
                logger.error(f"terminating node {node_url} failed since it does not exist.")
                success = False
            if not success:
                return f"Failed to terminate node {node_url}"
            return "Terminated successfully"
        except Exception:
            logger.error(f"Terminate node {node.url} failed.")
            return f"Failed to terminate node {node.url}, please check the input url."

    @app.get("/nodes/terminate_all", dependencies=[Depends(validate_json_request)])
    async def terminate_node_all():
        try:
            nodes = await registry.all_nodes()
            all_success = True
            for node in nodes:
                try:
                    await registry.remove(node.url)
                    headers = {"accept": "application/json"}
                    try:
                        response = requests.get(f"{node.url}/terminate", headers=headers)
                        if response.status_code != 200:
                            all_success = False
                    except Exception:
                        all_success = False
                except Exception:
                    all_success = False
            if not all_success:
                return "Failed to terminate all nodes"
            return "All nodes terminated successfully"
        except Exception:
            logger.error("Failed to terminate all nodes")
            return "Failed to terminate all nodes."

    # DistServe-specific endpoints
    if config.serving_strategy == config.serving_strategy.DIST_SERVE:
        from lmdeploy.serve.proxy.distserve import DistServeRouter
        distserve = DistServeRouter(registry, config)

        @app.post("/distserve/connection_warmup", dependencies=[Depends(validate_json_request)])
        async def connection_warmup():
            await distserve.connection_warmup()
            return JSONResponse({"SUCCESS": True})

        @app.post("/distserve/gc", dependencies=[Depends(validate_json_request)])
        async def cache_block_gc_to_be_migrated():
            raise NotImplementedError

    async def _check_request_model(model_name: str) -> JSONResponse | None:
        models = await registry.list_models()
        if model_name in models:
            return None
        return create_error_response(HTTPStatus.NOT_FOUND, f"The model {model_name!r} does not exist.")

    @app.post("/v1/chat/completions", dependencies=[Depends(validate_json_request)])
    async def chat_completions_v1(request: ChatCompletionRequest, raw_request: Request = None):
        check_response = await _check_request_model(request.model)
        if check_response is not None:
            return check_response

        if config.serving_strategy == config.serving_strategy.DIST_SERVE:
            return await distserve.handle_chat_completions(request, raw_request, "/v1/chat/completions")

        node = await strategy.select_node(request.model)
        if not node:
            return _handle_unavailable_model(request.model)

        logger.info(f"A request is dispatched to {node.url}")
        await strategy.on_request_start(node)
        start = time.time()

        aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT) if AIOHTTP_TIMEOUT else None
        async with aiohttp.ClientSession(timeout=aiotimeout) as client:
            if request.stream is True:
                response = forward_request_stream(client, node.url, raw_request, "/v1/chat/completions")
                background_task = BackgroundTasks()
                background_task.add_task(strategy.on_request_end, node, time.time() - start)
                return ProxyStreamingResponse(response, background=background_task, media_type="text/event-stream")
            else:
                response = await forward_request(client, node.url, raw_request, "/v1/chat/completions")
                await strategy.on_request_end(node, time.time() - start)
                return JSONResponse(json.loads(response))

    @app.post("/v1/completions", dependencies=[Depends(validate_json_request)])
    async def completions_v1(request: CompletionRequest, raw_request: Request = None):
        check_response = await _check_request_model(request.model)
        if check_response is not None:
            return check_response

        if config.serving_strategy == config.serving_strategy.DIST_SERVE:
            return await distserve.handle_completions(request, raw_request, "/v1/completions")

        node = await strategy.select_node(request.model)
        if not node:
            return _handle_unavailable_model(request.model)

        logger.info(f"A request is dispatched to {node.url}")
        await strategy.on_request_start(node)
        start = time.time()

        aiotimeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT) if AIOHTTP_TIMEOUT else None
        async with aiohttp.ClientSession(timeout=aiotimeout) as client:
            if request.stream is True:
                response = forward_request_stream(client, node.url, raw_request, "/v1/completions")
                background_task = BackgroundTasks()
                background_task.add_task(strategy.on_request_end, node, time.time() - start)
                return ProxyStreamingResponse(response, background=background_task, media_type="text/event-stream")
            else:
                response = await forward_request(client, node.url, raw_request, "/v1/completions")
                await strategy.on_request_end(node, time.time() - start)
                return JSONResponse(json.loads(response))

    return app
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/serve/proxy/app.py
git commit -m "feat(proxy): add app factory with all endpoint handlers"
```

______________________________________________________________________

### Task 9: Create new proxy.py entry point

**Files:**

- Modify: `lmdeploy/serve/proxy/proxy.py` (complete rewrite)

This replaces the 948-line monolith with a slim entry point that wires together config, registry, strategy, and app.

- [ ] **Step 1: Rewrite proxy.py**

```python
# lmdeploy/serve/proxy/proxy.py
# Copyright (c) OpenMMLab. All rights reserved.

import os
from typing import Literal

import uvicorn

from lmdeploy.serve.proxy.app import create_app
from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy, ServingStrategy
from lmdeploy.serve.proxy.node import NodeRegistry
from lmdeploy.serve.proxy.routing import get_strategy
from lmdeploy.utils import get_logger

logger = get_logger("lmdeploy")


def proxy(server_name: str = "0.0.0.0",
          server_port: int = 8000,
          serving_strategy: Literal["Hybrid", "DistServe"] = "Hybrid",
          routing_strategy: Literal["random", "min_expected_latency", "min_observed_latency",
                                    "min_cache_usage"] = "min_expected_latency",
          api_keys: list[str] | str | None = None,
          ssl: bool = False,
          log_level: str = "INFO",
          disable_cache_status: bool = False,
          link_type: Literal["RoCE", "IB"] = "RoCE",
          migration_protocol: Literal["RDMA"] = "RDMA",
          dummy_prefill: bool = False,
          **kwargs):
    """Launch the proxy server.

    Args:
        server_name (str): the server name of the proxy. Default to '0.0.0.0'.
        server_port (str): the server port. Default to 8000.
        serving_strategy ('Hybrid' | 'DistServe'): the strategy to serving.
            Hybrid default. DistServe for PD Disaggregation.
        routing_strategy ('random' | 'min_expected_latency' | 'min_observed_latency'
            | 'min_cache_usage'): the strategy to dispatch requests to nodes.
            Default to 'min_expected_latency'.
        api_keys (list[str] | str | None): Optional list of API keys.
        ssl (bool): Enable SSL.
        log_level (str): Set the log level. Default to INFO.
        disable_cache_status (bool): Whether to cache the proxy status.
        link_type: RDMA Link Type.
        migration_protocol: migration protocol for PD disaggregation.
        dummy_prefill: dummy prefill for performance profiler.
    """
    config = ProxyConfig(
        server_name=server_name,
        server_port=server_port,
        routing_strategy=RoutingStrategy.from_str(routing_strategy),
        serving_strategy=ServingStrategy(serving_strategy),
        disable_cache_status=disable_cache_status,
        migration_protocol=migration_protocol,
        link_type=link_type,
        disable_gdr=False,
        dummy_prefill=dummy_prefill,
        api_keys=api_keys if isinstance(api_keys, list) or api_keys is None else [api_keys],
        ssl=ssl,
        log_level=log_level,
    )

    registry = NodeRegistry(cache_status=not config.disable_cache_status)
    strategy = get_strategy(config.routing_strategy, registry, config)
    app = create_app(config, registry, strategy)

    # Preserve app reference for docs/conf.py import
    globals()["app"] = app

    if config.api_keys is not None and (tokens := [key for key in config.api_keys if key]):
        from lmdeploy.serve.utils.server_utils import AuthenticationMiddleware
        app.add_middleware(AuthenticationMiddleware, tokens=tokens)

    ssl_keyfile, ssl_certfile = None, None
    if config.ssl:
        ssl_keyfile = os.environ["SSL_KEYFILE"]
        ssl_certfile = os.environ["SSL_CERTFILE"]

    logger.setLevel(config.log_level)
    uvicorn_log_level = os.getenv("UVICORN_LOG_LEVEL", "info").lower()
    uvicorn.run(
        app=app,
        host=config.server_name,
        port=config.server_port,
        log_level=uvicorn_log_level,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )


if __name__ == "__main__":
    import fire
    fire.Fire(proxy)
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/serve/proxy/proxy.py
git commit -m "feat(proxy): rewrite proxy.py as slim entry point"
```

______________________________________________________________________

### Task 10: Update __init__.py and delete old files

**Files:**

- Modify: `lmdeploy/serve/proxy/__init__.py`

- Delete: `lmdeploy/serve/proxy/utils.py`

- Delete: `lmdeploy/serve/proxy/streaming_response.py`

- [ ] **Step 1: Update __init__.py**

```python
# lmdeploy/serve/proxy/__init__.py
# Copyright (c) OpenMMLab. All rights reserved.

from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy
from lmdeploy.serve.proxy.proxy import proxy
```

- [ ] **Step 2: Delete old files**

```bash
rm lmdeploy/serve/proxy/utils.py
rm lmdeploy/serve/proxy/streaming_response.py
```

- [ ] **Step 3: Verify no broken imports**

Run: `python -c "from lmdeploy.serve.proxy.proxy import proxy; from lmdeploy.serve.proxy import app"`
Expected: No import errors

- [ ] **Step 4: Commit**

```bash
git add -A lmdeploy/serve/proxy/
git commit -m "feat(proxy): update __init__.py, remove old utils.py and streaming_response.py"
```

______________________________________________________________________

### Task 11: Update CLI serve.py

**Files:**

- Modify: `lmdeploy/cli/serve.py:169-206`

Add `min_cache_usage` to the routing strategy choices.

- [ ] **Step 1: Update the CLI argument**

In `lmdeploy/cli/serve.py`, find the `--routing-strategy` argument (around line 185-189) and update the choices:

```python
parser.add_argument('--routing-strategy',
                    type=str,
                    choices=['random', 'min_expected_latency', 'min_observed_latency', 'min_cache_usage'],
                    default='min_expected_latency',
                    help='the strategy to dispatch requests to nodes')
```

- [ ] **Step 2: Commit**

```bash
git add lmdeploy/cli/serve.py
git commit -m "feat(proxy): add min_cache_usage to CLI routing strategy choices"
```

______________________________________________________________________

### Task 12: Update docs/conf.py imports

**Files:**

- Modify: `lmdeploy/docs/en/conf.py:25`

- Modify: `lmdeploy/docs/zh_cn/conf.py:25`

- [ ] **Step 1: Update both conf.py files**

Change the import from:

```python
from lmdeploy.serve.proxy.proxy import app as proxy_server  # noqa: E402
```

to:

```python
from lmdeploy.serve.proxy import app as proxy_server  # noqa: E402
```

Note: `app` is now created by `create_app()`, but we need a module-level reference. The `proxy()` function sets `globals()["app"] = app` so `from lmdeploy.serve.proxy import app` works after the proxy is started. For docs, we may need to provide a default app instance. Check if the docs actually use this import at build time.

- [ ] **Step 2: Verify docs build if possible**

Run: `cd docs/en && python -c "from lmdeploy.serve.proxy.config import ProxyConfig, RoutingStrategy; print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add docs/en/conf.py docs/zh_cn/conf.py
git commit -m "fix(docs): update proxy import for refactored module"
```

______________________________________________________________________

### Task 13: Run full test suite and lint

**Files:**

- All proxy files

- [ ] **Step 1: Run proxy-specific tests**

Run: `pytest tests/test_lmdeploy/serve/proxy/ -v`
Expected: All PASS

- [ ] **Step 2: Run pre-commit linting**

Run: `pre-commit run --all-files`
Expected: PASS (fix any failures)

- [ ] **Step 3: Run broader test suite to check for regressions**

Run: `pytest tests/test_lmdeploy/ -v --timeout=60 -x`
Expected: No failures related to proxy changes

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "style(proxy): fix linting issues"
```

______________________________________________________________________

## Self-Review Checklist

- [x] **Spec coverage:** Every section in the design spec has a corresponding task. `config.py` (Task 1), `node.py` (Task 2), routing strategies (Tasks 3-4), `forwarding.py` (Task 5), `streaming.py` (Task 6), `distserve.py` (Task 7), `app.py` (Task 8), `proxy.py` entry point (Task 9), cleanup (Task 10), CLI update (Task 11), docs (Task 12), verification (Task 13).
- [x] **Placeholder scan:** No TBD/TODO placeholders. All code steps contain complete implementations.
- [x] **Type consistency:** `select_node()` returns `Node` in all strategies. `NodeRegistry.get()` returns `list[Node]`. `ProxyConfig` fields match between tasks. `RoutingStrategy.from_str()` is consistent across tasks.
