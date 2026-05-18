# Proxy Server Refactor Design

## Goal

Re-implement the lmdeploy proxy server (`lmdeploy/serve/proxy/`) with clean module boundaries, a pluggable routing strategy pattern, and a new `min_cache_usage` strategy that routes requests to the backend with the lowest KV cache occupation.

## Requirements

- Replace the 948-line monolith with focused, independently readable modules
- Add a `min_cache_usage` routing strategy that polls backend `/metrics` endpoints for `lmdeploy:gpu_cache_usage_perc`
- Preserve all existing functionality: Hybrid and DistServe serving modes, all endpoints, node management
- Use a Pydantic `ProxyConfig` class instead of scattered kwargs
- Background polling for metrics (default 5s, configurable via `LMDEPLOY_PROXY_POLL_METRICS_INTERVAL` env var)
- Fallback to `min_expected_latency` strategy when a node has no metrics data

## Module Structure

```
lmdeploy/serve/proxy/
├── __init__.py          # Re-exports: proxy(), ProxyConfig, RoutingStrategy
├── config.py            # ProxyConfig, RoutingStrategy enum, ServingStrategy enum, ErrorCodes enum
├── node.py              # Node model, NodeRegistry class
├── routing/
│   ├── __init__.py      # Re-exports: get_strategy()
│   ├── base.py          # Abstract BaseStrategy
│   ├── random.py        # RandomStrategy
│   ├── min_expected.py  # MinExpectedLatencyStrategy
│   ├── min_observed.py  # MinObservedLatencyStrategy
│   └── min_cache.py     # MinCacheUsageStrategy (new)
├── forwarding.py        # forward_request(), forward_request_stream()
├── streaming.py         # ProxyStreamingResponse
├── distserve.py         # DistServe routing, KV cache migration
├── app.py               # FastAPI app, all endpoint handlers, lifespan
└── proxy.py             # Entry point: proxy() CLI function
```

## ProxyConfig

```python
class RoutingStrategy(str, Enum):
    RANDOM = "random"
    MIN_EXPECTED_LATENCY = "min_expected_latency"
    MIN_OBSERVED_LATENCY = "min_observed_latency"
    MIN_CACHE_USAGE = "min_cache_usage"

class ServingStrategy(str, Enum):
    HYBRID = "Hybrid"
    DIST_SERVE = "DistServe"

class ProxyConfig(BaseModel):
    server_name: str = "0.0.0.0"
    server_port: int = 8000
    routing_strategy: RoutingStrategy = RoutingStrategy.MIN_EXPECTED_LATENCY
    serving_strategy: ServingStrategy = ServingStrategy.HYBRID
    disable_cache_status: bool = False
    metrics_poll_interval: float = 5.0  # overridden by LMDEPLOY_PROXY_POLL_METRICS_INTERVAL env var
    migration_protocol: str = "RDMA"
    link_type: str = "RoCE"
    disable_gdr: bool = False
    dummy_prefill: bool = False
    api_keys: list[str] | None = None
    ssl: bool = False
    log_level: str = "INFO"
```

## Node & NodeRegistry

```python
class Node(BaseModel):
    url: str
    role: EngineRole = EngineRole.Hybrid
    models: list[str] = []
    unfinished: int = 0
    latency: deque[float] = Field(default_factory=lambda: deque(maxlen=15))
    speed: int | None = None
    cache_usage: float | None = None       # from /metrics polling
    last_metrics_poll: float | None = None  # timestamp of last successful poll
```

**NodeRegistry** methods:

- `add(url, role, models, status)` — register a node
- `remove(url)` — remove a node
- `get(model_name: str) -> list[Node]` — nodes serving a model
- `get_by_url(url: str) -> Node` — lookup by URL
- `list_models() -> list[str]` — unique model names
- `update_cache_usage(url, usage)` — update cached KV cache metric
- `persist()` / `load()` — save/restore state from `proxy_config.json`
- Uses `asyncio.Lock` for thread-safe mutations

## Routing Strategy Pattern

```python
class BaseStrategy(ABC):
    def __init__(self, registry: NodeRegistry, config: ProxyConfig): ...

    @abstractmethod
    async def select_node(self, model_name: str) -> Node: ...

    async def on_request_start(self, node: Node) -> None:
        node.unfinished += 1

    async def on_request_end(self, node: Node, latency: float) -> None:
        node.unfinished -= 1
        node.latency.append(latency)

    async def start(self) -> None: ...    # start background tasks
    async def stop(self) -> None: ...     # stop background tasks
```

### Existing Strategies

| Strategy                     | select_node logic           |
| ---------------------------- | --------------------------- |
| `RandomStrategy`             | Weighted random by speed    |
| `MinExpectedLatencyStrategy` | Lowest `unfinished / speed` |
| `MinObservedLatencyStrategy` | Lowest mean latency         |

### New: MinCacheUsageStrategy

Holds an internal `MinExpectedLatencyStrategy` instance for fallback routing.

**Background polling:**

- On `start()`, launches an `asyncio.Task` that runs a polling loop
- Every `metrics_poll_interval` seconds, iterates all registered nodes
- For each node: `GET {node.url}/metrics`, parses `lmdeploy:gpu_cache_usage_perc` from Prometheus text format
- Calls `registry.update_cache_usage(url, usage)` with the parsed value
- On `stop()`, cancels the background task

**Routing decision:**

- Get nodes for the requested model
- Filter to nodes with `cache_usage is not None` and `last_metrics_poll` within `3 * metrics_poll_interval`
- Pick the node with lowest `cache_usage`
- If no node has valid metrics data, fall back to `MinExpectedLatencyStrategy.select_node()`

**Metrics parsing:**

- Parse the Prometheus text exposition format line by line
- Extract the value for `lmdeploy:gpu_cache_usage_perc` (handles label variants)
- On parse failure or connection error: leave `cache_usage` as `None` for that node

**Staleness:**

- If `last_metrics_poll` is older than `3 * metrics_poll_interval`, treat `cache_usage` as stale (set to `None`)
- This ensures nodes with dead metric endpoints are eventually excluded from cache-based routing

## Forwarding

Two async functions in `forwarding.py`:

- `forward_request(client, node_url, request) -> Response` — non-streaming raw forwarding
- `forward_request_stream(client, node_url, request) -> AsyncIterator[bytes]` — streaming raw forwarding

Both add `X-Forwarded-For` / `X-Forwarded-Host` headers. Catch `aiohttp.ClientError` → raise `APIServerException`.

## App

```python
def create_app(config: ProxyConfig, registry: NodeRegistry, strategy: BaseStrategy) -> FastAPI:
```

- Lifespan handler: calls `strategy.start()` on startup, `strategy.stop()` on shutdown
- All current endpoints preserved
- Request flow: parse model → `strategy.select_node()` → `strategy.on_request_start()` → forward → `strategy.on_request_end()`
- No global state

## DistServe

Isolated in `distserve.py`:

- `DistServeRouter` class handles prefill/decode node selection
- `forward_prefill()` / `forward_decode()` two-step request flow
- Connection management: warmup, RDMA/NVLink setup
- Delegated to by the proxy app when `serving_strategy == DistServe`

## Error Handling

- `APIServerException` preserved from current code (in `config.py` alongside other shared types)
- `ProxyStreamingResponse` preserved — catches `APIServerException` from streaming generators
- Background polling failures are logged but don't crash the proxy — nodes with failed polls simply have `cache_usage = None`

## Environment Variables

- `LMDEPLOY_PROXY_POLL_METRICS_INTERVAL` — override `metrics_poll_interval` in ProxyConfig (in seconds)
- `LMDEPLOY_CONTROLLER_HEART_BEAT_EXPIRATION` — heartbeat check interval (preserved from current code)
- `AIOHTTP_TIMEOUT` — client timeout for aiohttp requests (preserved from current code)
