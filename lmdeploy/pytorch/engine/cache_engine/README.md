# PyTorch Cache Engine

This package separates three decisions that were previously mixed inside
`CacheEngine`: what cache an operator needs, how a backend stores it, and which
tensors own the runtime memory.

## Reviewer Mental Model

Only three primary objects are needed to follow the construction pipeline:

```text
built operator requirements
        |
        v
BlockCacheRequest[]          "what is needed"
        |
        v
BlockCachePlan               "the finalized worker-local recipe"
        |
        +-- allocate on meta --> bytes per logical block
        +-- allocate on CPU  --> CacheAllocation
        `-- allocate on GPU  --> CacheAllocation
                                  "the real tensors"
```

`CacheEngine` retains one plan and the CPU/GPU allocations realized from it.
It owns their lifetime, compatibility views, swap ordering, streams, and
events. It does not decide operator payloads or backend packing.

The remaining types support one of those three stages:

| Supporting type           | Why it exists                                                     |
| ------------------------- | ----------------------------------------------------------------- |
| `ScopedBlockCacheRequest` | Lets the model bind an operator request to a stable layer ID      |
| `CacheResource`           | Normalized, validated request data retained inside the plan       |
| `BlockCacheLayout`        | Backend-selected allocation strategy retained inside the plan     |
| `CachePool`               | One owning tensor and the axis representing movable cache entries |

These supporting types are not additional runtime managers. Requests and
scoped requests are temporary build inputs; resources and layout are immutable
plan details; pools belong to one allocation.

## Construction Pipeline

### 1. Collect requests

A selected cache-using operator declares one physical kernel-block payload:

```python
BlockCacheRequest(
    name='index_cache',
    shape=(kernel_block_size, num_heads, head_dim),
    dtype=dtype,
    per_layer_contiguous=True,
)
```

The operator knows its payload and kernel-visible requirements, but not the
model's complete layer topology. The enclosing model binds each request:

```python
ScopedBlockCacheRequest(request, layer_id=7)
```

The optional built-model `get_block_cache_requests(geometry)` method collects
those scoped requests. Collection happens independently on every target,
speculative, and memory-model worker after logical/kernel block geometry is
finalized.

Models without a provider retain the compatibility path through standard K/V
configuration, `block_cache_specs`, and anonymous `cache_shapes`. A present
provider is authoritative for custom block resources, even when it returns no
requests.

### 2. Build one worker-local plan

The collector validates requests and combines equal requests from different
layers into `CacheResource` objects with compact layer-row maps. The active
`CacheBackend` then selects physical layouts for the ordered resources.

`BlockCachePlan` retains:

- ordered resources and model-facing access metadata;
- the selected physical layout;
- logical-to-kernel block geometry.

It owns no tensors, streams, events, or movement policy. Plans never cross the
executor RPC boundary; each worker returns only target/speculative/memory byte
counts to the executor.

Current provider limits are explicit: global requests, heterogeneous segments
under one name, and providers used through the old dlinfer allocator monkey
patch are rejected at build time.

### 3. Realize allocations

The same retained plan is realized on `meta`, CPU, and accelerator devices:

```python
allocation = plan.allocate(num_logical_blocks, device)
```

`CacheAllocation.pools` own storage and drive byte accounting and movement.
`CacheAllocation.caches` are typed views in the same order as
`plan.resources`. Count bytes from pools, never from possibly overlapping
views.

Every pool records its cache-entry axis. This allows swap and later block-copy
operations to handle one or many pools without assuming a tensor rank or
memory order.

## Layout Selection and Composition

An atomic layout implements one physical storage policy:

| Layout                       | Storage policy                                                    |
| ---------------------------- | ----------------------------------------------------------------- |
| `PackedBlockCacheLayout`     | Pack compatible full-layer resources into one byte pool           |
| `LayerRowBlockCacheLayout`   | Give layer-scoped resources compact padded byte pools             |
| `ContiguousBlockCacheLayout` | Give every resource an independent contiguous typed tensor        |
| `PackedStateCacheLayout`     | Pack resources behind one state-slot axis                         |
| dlinfer block/state layouts  | Use dlinfer's backend-specific contiguous resource representation |

`CompositeBlockCacheLayout` combines ordered atomic layouts. It allocates each
child, concatenates their owning pools, and preserves child/resource view
order. It contains no resource-selection policy; `DefaultCacheBackend` owns
that decision.

For standard K/V plus a contiguous index cache, the selected plan is:

```text
BlockCachePlan
└── CompositeBlockCacheLayout
    ├── PackedBlockCacheLayout(K, V)       -> packed pool
    └── ContiguousBlockCacheLayout(index)  -> typed contiguous pool
```

The default backend groups consecutive resources by requirement:

- unscoped resources that accept strides use the packed layout;
- layer-scoped resources that accept padded strides use the layer-row layout;
- resources with `per_layer_contiguous=True` use the contiguous layout.

Different resources may therefore use different layouts while sharing one
plan and scheduler block count. Conflicting requirements for the same resource
and layer fail during request aggregation.

## Model-Facing Views

Two access forms coexist during migration:

- `gpu_cache` / `cpu_cache` provide the legacy per-layer tuple used as
  `past_key_values`;
- `block_caches` provides named access, and
  `block_caches.layer(name, layer_id)` resolves compact layer rows.

`BlockCachePlan.legacy_cache_indices` determines which resources participate
in legacy tuples. In a mixed allocation, adding a scoped named resource does
not change standard K/V tuple ordering.

These views do not own memory. The corresponding `CacheAllocation` must remain
alive.

## Logical and Kernel Blocks

`CacheConfig.block_size` is the scheduler and prefix-cache unit.
`kernel_block_size` is the physical page unit expected by kernels. The logical
size must be an exact positive multiple of the kernel size:

```text
num_kernel_blocks = num_logical_blocks
                    * plan.kernel_blocks_per_logical_block
```

Meta sizing and real allocation use this same conversion and layout path.

## Runtime Ownership

After CPU and accelerator allocation, `CacheEngine._build_swap_pairs()` checks
that corresponding pools have the same count, entry axis, dtype, and
per-entry payload shape. Swap operations then use those pre-resolved pairs on
the cache stream.

`StateCacheEngine` uses the same schema/layout/allocation primitives but has a
different slot lifecycle and currently does not retain a plan. State
initialization and copies operate on allocation-owned resources rather than an
assumed shared pool.

Same-device block copy, CPU/accelerator swap, and PD/LMCache/Mooncake transfer
remain separate operations. Layout selection does not own those lifecycles.

## Compatibility Boundaries

The package temporarily preserves:

- unpacking `CacheAllocation` as `(mem_pool, caches)`;
- `CacheEngine.allocate_caches()` for direct callers and older dlinfer patches;
- anonymous cache/state shapes and model-config named specifications;
- `gpu_cache`, `cpu_cache`, `full_gpu_cache`, and `full_cpu_cache`.

Native providers require the retained-plan allocator path. PD migration still
rejects unsupported multi-pool layouts before registering memory.

## Code Reading Order

1. [`schema.py`](./schema.py): requests, normalized resources, and layer rows.
2. [`plan.py`](./plan.py): the retained worker-local allocation recipe.
3. [`layout.py`](./layout.py): atomic/composite layouts, pools, and allocations.
4. [`backends/default/cache.py`](../../backends/default/cache.py): default
   resource-to-layout selection.
5. [`engine.py`](./engine.py): allocation lifetime, model views, and movement.

Backend-specific layouts remain with their backend, for example
[`backends/dlinfer/cache.py`](../../backends/dlinfer/cache.py). Avoid adding a
generic manager, registry, or utility module; add a component only when it owns
a complete decision or lifecycle.

## Focused Tests

```bash
python -m pytest -q \
  tests/pytorch/engine/test_cache_schema.py \
  tests/pytorch/engine/test_cache_layout.py \
  tests/pytorch/engine/test_cache_engine.py \
  tests/pytorch/engine/test_executor_base.py \
  tests/pytorch/engine/test_model_agent.py
```
