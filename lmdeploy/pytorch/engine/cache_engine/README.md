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
It owns their lifetime, view construction, swap ordering, streams, and events.
`NamedCacheView` owns only model-facing name/row/layer lookup. Neither decides
operator payloads or backend packing.

The remaining types support one of those three stages:

| Supporting type    | Why it exists                                                     |
| ------------------ | ----------------------------------------------------------------- |
| `CacheTensorSpec`  | Non-owning tensor metadata retained inside the plan               |
| `BlockCacheLayout` | Backend-selected allocation strategy retained inside the plan     |
| `CachePool`        | One owning tensor and the axis representing movable cache entries |
| `NamedCacheView`   | Read-only model-facing lookup across physical cache tensors       |

These supporting types are not additional runtime managers. Requests are
temporary build inputs; tensor specs and layout are immutable plan details;
pools belong to one allocation.

## Construction Pipeline

### 1. Collect requests

A selected cache-using operator declares one physical kernel-block payload:

```python
BlockCacheRequest(
    name='index_cache',
    shape=(kernel_block_size, num_heads, head_dim),
    dtype=dtype,
    per_row_contiguous=True,
)
```

The operator knows its payload and kernel-visible requirements. A generic
worker-local collector walks the built model's operator modules, collects
their `get_block_cache_requests(context)` results, and calls
`bind_block_cache(binding)` on each consumer. `BlockCacheRequestContext`
carries worker-finalized inputs, currently logical/kernel block geometry;
`BlockCacheBinding` carries the consumer's logical cache name and row. These
records can grow without changing every operator method signature, but never
carry physical tensors or layout details. Collection uses built module
instances and registration order; it does not infer layer IDs from module names
or require a model-specific collector. It happens independently on every
target, speculative, and memory-model worker after block geometry is finalized.

The DSA indexer is the first production cache-requesting operator on this
path. Its selected backend implementation requests one packed
`dsa_indexer_k` byte tensor with a per-row contiguity requirement. Every built
indexer receives its compact row and retrieves that named row at runtime.
Target and MTP model classes contain no cache-discovery code, and model
configuration no longer describes separate anonymous K and scale cache
tensors.

Models without a cache-requesting operator retain the compatibility path
through standard K/V configuration, `block_cache_specs`, and anonymous
`cache_shapes`. A discovered requester is authoritative for custom block
caches, even when it returns no requests.

### 2. Build one worker-local plan

Schema construction validates requests and combines equal contracts from built
consumers into `CacheTensorSpec` objects. Each spec records the stable consumer
rows stored by its future tensor. For example, requests with contracts
`A, B, A` become `A(consumer_rows=(0, 2))` and `B(consumer_rows=(1,))`.
Configuration-owned specs may instead retain explicit layer-row maps.
`schema.py` also derives standard K/V and quantized payload descriptions from
the finalized model/cache policy. `plan.py` passes the ordered tensor specs to
the active `CacheBackend`, retains its selected physical layout, and constructs
the plan.

`BlockCachePlan` retains:

- ordered tensor specs and model-facing access metadata;
- the selected physical layout;
- logical-to-kernel block geometry.

It owns no tensors, streams, events, or movement policy. Plans never cross the
executor RPC boundary; each worker returns only target/speculative/memory byte
counts to the executor.

The exact construction path is:

```text
CacheEngine.build_cache_plan()
  ├── finalize BlockCacheGeometry and sparse-MLA cache policy
  ├── request_collector(BlockCacheRequestContext(geometry))
  └── plan.build_block_cache_plan(...)
        ├── schema.build_model_block_cache_tensor_specs(...)
        ├── get_backend().get_cache_backend().build_block_layout(...)
        └── BlockCachePlan(...)
```

The first method is a compatibility and request-collection boundary. The
function in `plan.py` receives finalized inputs and owns physical layout
selection plus construction of the immutable plan.

Request collection through the old dlinfer allocator monkey patch remains
rejected at build time because that compatibility path cannot consume a
worker-local plan. `CacheEngine.build_cache_plan()` remains the compatibility
facade that finalizes geometry, invokes the worker-owned request collector, and
enforces this guard before delegating plan construction.

### 3. Realize allocations

The same retained plan is realized on `meta`, CPU, and accelerator devices:

```python
allocation = plan.allocate(num_logical_blocks, device)
```

`CacheAllocation.pools` own storage and drive byte accounting and movement.
`CacheAllocation.tensor_views` are typed tensors in the same order as
`plan.tensor_specs`. Count bytes from pools, never from possibly overlapping
views.

Every pool records its cache-entry axis. This allows swap and later block-copy
operations to handle one or many pools without assuming a tensor rank or
memory order.

### Local logical-block copy

`CacheEngine.copy_logical_blocks()` accepts a device tensor with shape
`[2, num_pairs]` containing physical block-table offsets at scheduler-block
granularity. The retained plan supplies the scheduler-block to kernel-page
ratio, while `CacheAllocation` supplies every owning pool and its entry axis.

The active `CacheBackend` builds the physical copy primitive once from that
stable allocation. CUDA copies contiguous pools with Triton; other backends
inherit the bounded tensor fallback unless they provide a more suitable local
primitive. CacheEngine validates only plan metadata on the hot path. The caller
owns source/destination relationship validation and keeps block lifetimes safe
until the stream-ordered copy completes.

### Per-forward checkpoint pipeline

Checkpoint copies are one-forward side effects, not persistent model inputs:

```text
checkpoint lifecycle + paging ids
              |
              v
InputsMaker reserves/pins and resolves physical block offsets
              |
              v
CacheCheckpointInputs
    restore plans -> model forward -> save plans
              |                         |
              +---- CacheEngine --------+  KV logical-block copies
              `---- StateCacheEngine ---+  state-slot copies
```

`CacheCheckpointInputs` travels beside `ModelInputs` in the executor payload.
Only its KV plans move to the cache device; state plans remain compact host
index pairs. The model agent consumes restores after context construction and
before the model, then consumes saves after the model. The engine loop
publishes reserved state checkpoints and releases their pins at the existing
forward/output boundaries.

Keeping this payload separate prevents one-shot operations from being cloned,
reindexed, merged, or advanced with persistent decode inputs. At this stage the
existing SSM state checkpoint flow produces state plans. The KV plan fields and
logical-to-physical validation boundary are ready for the later non-aligned
checkpoint chapter; that chapter will decide which frozen KV block pairs to
emit.

## Layout Selection and Composition

An atomic layout implements one physical storage policy:

| Layout                       | Storage policy                                                  |
| ---------------------------- | --------------------------------------------------------------- |
| `PackedBlockCacheLayout`     | Pack compatible full-layer tensors into one byte pool           |
| `RowBlockCacheLayout`        | Give compact-row tensors independent padded byte pools          |
| `ContiguousBlockCacheLayout` | Give every tensor spec an independent contiguous typed tensor   |
| `PackedStateCacheLayout`     | Pack tensors behind one state-slot axis                         |
| dlinfer block/state layouts  | Use dlinfer's backend-specific contiguous tensor representation |

`CompositeBlockCacheLayout` combines ordered atomic layouts. It allocates each
child, concatenates their owning pools, and preserves child/tensor view order.
It contains no tensor-selection policy; `DefaultCacheBackend` owns that
decision.

For standard K/V plus a contiguous index cache, the selected plan is:

```text
BlockCachePlan
└── CompositeBlockCacheLayout
    ├── PackedBlockCacheLayout(K, V)       -> packed pool
    └── ContiguousBlockCacheLayout(index)  -> typed contiguous pool
```

The default backend groups consecutive tensor specs by requirement:

- full-model tensors that accept strides use the packed layout;
- compact-row tensors that accept padded strides use the row layout;
- specs with `per_row_contiguous=True` use the contiguous layout.

Different tensor specs with the same semantic name may therefore use different
layouts while sharing one plan and scheduler block count.

## Model-Facing Views

Two access forms coexist during migration:

- `gpu_cache` / `cpu_cache` provide the per-layer tuple used as
  `past_key_values`;
- `block_caches.row(name, consumer_row)` resolves the row assigned to a built
  operator, even when that name spans different physical shapes;
- `block_caches.layer(name, layer_id)` remains the compatibility lookup for
  configuration-owned layer maps.

`block_caches[name]` still returns the complete tensor when a name has one
physical tensor. When a name is heterogeneous, direct lookup raises and asks
the caller to select a consumer or layer row explicitly.

`BlockCachePlan.model_cache_indices` determines which tensors participate in
per-layer tuples. In a mixed allocation, adding a compact-row named tensor
does not change standard K/V tuple ordering.

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
initialization and copies operate on allocation-owned pools rather than an
assumed shared pool.

Same-device block copy, CPU/accelerator swap, and PD/LMCache/Mooncake transfer
remain separate operations. Layout selection does not own those lifecycles.

PD migration registers every accelerator allocation pool with a stable
memory-region key. The P/D endpoint handshake exchanges each pool's shape,
dtype, element size, and entry axis. Migration then decomposes a requested
block into contiguous byte segments for that pool. Pools may use different
entry axes, and corresponding P/D pools may place the entry axis differently;
the planner pairs their segments in logical payload order before dispatching
one backend batch per remote engine.

## Compatibility Boundaries

The package temporarily preserves:

- `CacheEngine.build_cache_plan()`, `allocate_caches()`, and the K/V/quant/custom
  description facades used by dlinfer's contiguous allocator patch;
- `CacheEngine.num_layers` and `kv_cache_dtype`, which dlinfer's Ascend310P
  allocator patch still reads even though native allocation uses the plan;
- anonymous cache/state shapes and model-config named specifications;
- `gpu_cache` and `cpu_cache` as per-layer model-facing tuples;
- the external dlinfer allocator result `(mem_pool, caches)`.

For native block caches, `gpu_allocation` and `cpu_allocation` are the internal
sources of truth. Native allocation, swap, copy, sizing, and PD paths consume
`CacheAllocation` directly and do not coerce it into a tuple. A private external
GPU pool is retained only when a downstream patched allocator returns
`(mem_pool, caches)`, because that result has no ownership or entry-axis
metadata.

Native operator request collection requires the retained-plan allocator path.
PD migration accepts one or more contiguous native allocation pools with equal
logical/kernel block sizes. Corresponding P/D pools must retain the same order,
dtype, and logical payload shape after removing the entry axis. It does not yet
map a packed pool on one endpoint to several pools on the other. External patched
allocators remain limited to one `[layer, block, ...]` tensor because they do
not provide per-pool entry-axis metadata.

## Code Reading Order

01. [`collector.py`](./collector.py): built-operator request collection and row
    binding.
02. [`schema.py`](./schema.py): payload descriptions, requests, tensor specs,
    and row bindings.
03. [`plan.py`](./plan.py): backend layout selection and the retained
    worker-local allocation recipe.
04. [`layout.py`](./layout.py): atomic/composite layouts, pools, and allocations.
05. [`migration.py`](./migration.py): PD pool metadata and byte-transfer planning.
06. [`backends/default/cache.py`](../../backends/default/cache.py): default
    tensor-spec-to-layout selection.
07. [`view.py`](./view.py): named tensor, consumer-row, and layer-row lookup.
08. [`../cache_inputs.py`](../cache_inputs.py): one-forward checkpoint copy
    payloads.
09. [`state.py`](./state.py): state allocation and state-slot transitions.
10. [`engine.py`](./engine.py): compatibility construction facades, block-cache
    allocation lifetime, view construction, and movement.

Backend-specific layouts remain with their backend, for example
[`backends/dlinfer/cache.py`](../../backends/dlinfer/cache.py). Avoid adding a
generic manager, registry, or utility module; add a component only when it owns
a complete decision or lifecycle.

## Focused Tests

```bash
python -m pytest -q \
  tests/pytorch/engine/test_cache_engine \
  tests/pytorch/engine/test_executor_base.py \
  tests/pytorch/engine/test_model_agent.py
```
