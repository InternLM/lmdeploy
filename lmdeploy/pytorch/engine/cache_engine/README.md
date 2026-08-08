# PyTorch Cache Engine

This package owns cache resource descriptions, backend-selected physical
layouts, owning allocations, and the runtime lifecycle of block and state
caches. It keeps the model-facing cache views separate from the tensors that
own their storage.

Cache construction is a staged process. Model and cache declarations describe
the required payloads, the active backend selects a physical layout, and the
cache engine realizes that layout on the device and host before serving
requests.

## Recommended Reading Order

1. Read this document for the ownership and lifecycle contracts.
2. Read [`schema.py`](./schema.py) for resource descriptions and layer
   membership.
3. Read [`plan.py`](./plan.py) for finalized block-cache geometry and layout.
4. Read [`layout.py`](./layout.py) for owning allocations and the default
   physical layouts.
5. Read `CacheEngine.build_cache_plan()` and the allocation methods in
   [`engine.py`](./engine.py).
6. Read [`backends/cache.py`](../../backends/cache.py), then the active
   backend's cache provider, to understand layout selection.
7. Read `CacheEngine._build_swap_pairs()` and `StateCacheEngine` only when
   changing runtime movement.

Do not infer physical contiguity from a model-facing cache view. The selected
backend layout and its `CacheAllocation.pools` are the source of truth for
owning storage and cache-entry axes.

## Construction Model

```text
ModelConfig / CacheConfig declarations
        |
        v
CacheResource[]
  - name
  - payload shape and dtype
  - alignment
  - optional global-layer membership
        |
        v
CacheBackend.build_block_layout()
        |
        v
BlockCachePlan
  - ordered resources and access metadata
  - selected physical layout
  - logical-to-kernel block geometry
        |
        +-- meta allocation --> bytes per logical block
        |
        +-- device allocation --> CacheAllocation
        |
        `-- host allocation ---> CacheAllocation
                                  - owning CachePool objects
                                  - typed resource views
        |
        v
CacheEngine
  - model-facing compatibility views
  - pre-resolved swap pairs
  - stream and event ordering
```

`BlockCachePlan` is a finalized build-time recipe, not a runtime allocation.
It owns no tensors, streams, or events. One runtime `CacheEngine` retains one
plan and realizes it for both device and host storage.

State caches currently follow the same schema, backend-layout, and allocation
contracts but are allocated directly by `StateCacheEngine`; they do not yet
have a retained plan object.

## Ownership Map

| Component               | Owns                                                                                | Does not own                                  |
| ----------------------- | ----------------------------------------------------------------------------------- | --------------------------------------------- |
| `CacheDesc`             | One payload's shape, dtype, alignment, and derived sizes                            | Layer membership or storage                   |
| `LayerRowMap`           | Ordered global-layer membership and compact-row lookup                              | Model-wide layer count                        |
| `CacheResource`         | One stable cache name, payload description, and optional layer rows                 | Physical packing or tensors                   |
| `CacheBackend`          | Backend-specific physical layout selection                                          | CacheEngine lifecycle or allocated tensors    |
| `BlockCachePlan`        | Finalized block resources, access metadata, selected layout, and block geometry     | Tensors, streams, or runtime movement         |
| `CachePool`             | One owning tensor and its cache-entry axis                                          | Typed interpretation of every resource view   |
| `CacheAllocation`       | Owning pools and typed views derived from them                                      | Host/device movement policy                   |
| `CacheEngine`           | Device/host allocation lifetime, compatibility views, swap pairs, stream, and event | Backend packing policy                        |
| `StateCacheEngine`      | State allocation lifetime, named views, initialization, and local slot copies       | Prefix-checkpoint identity or slot scheduling |
| Scheduler/state manager | Logical block or state-slot assignment                                              | Physical cache layout                         |

`CacheEngine` is the composition root for block-cache construction and
runtime movement. Schema, plan, and layout modules do not hold an engine
back-reference.

## Resource and Layer Model

A `CacheResource` describes one independently named cache payload. Examples
include standard K and V caches, quantization metadata, a compressed cache, or
an index cache used by a non-attention operator.

Layer membership is explicit when a resource exists only for selected model
layers:

```text
declared layer ids:       [9, 1]
compact resource rows:    [0, 1]
row_by_layer:             {9: 0, 1: 1}
```

The declared order determines physical row order. Layer ids must be
non-negative, unique, and non-empty. They are identifiers, not evidence that a
model has one uniform `num_layers` concept.

`NamedCacheView.layer(name, layer_id)` translates a global layer id to the
resource's compact row. Resources without an explicit `LayerRowMap` retain the
legacy direct layer-index behavior.

The current declaration source is `ModelConfig`: standard KV fields,
`block_cache_specs`, `state_cache_specs`, and legacy anonymous shapes are
normalized into `CacheResource` objects. Collection from built cache-using
operators is a planned replacement for model-config-owned declarations; it is
not implemented by this package yet.

## Plan, Layout, and Allocation

These types represent different decisions and lifetimes:

| Type              | Question answered                                                            | Lifetime                      |
| ----------------- | ---------------------------------------------------------------------------- | ----------------------------- |
| `CacheResource`   | What cache payload exists, and for which layers?                             | Schema construction onward    |
| `BlockCachePlan`  | What finalized recipe will size and realize this worker's block cache?       | Engine build/runtime          |
| Physical layout   | How should the active backend arrange these resources?                       | Retained by the plan          |
| `CacheAllocation` | Which tensors actually own one realized cache, and what are its typed views? | One device or host allocation |

A layout may create one pool or many pools. Pool count is a physical decision,
not part of the model-facing cache contract. Every pool records the axis whose
indices are independently movable cache entries, allowing runtime movement to
work without assuming a fixed tensor rank or memory order.

`CacheAllocation.caches` are typed resource views consumed by model code.
`CacheAllocation.pools` own the storage and are used for byte accounting and
block/state movement. Count bytes from pools so shared storage is not counted
once per view.

## Physical Layouts

The default and dlinfer backends intentionally make different layout choices:

| Layout                     | Owning storage                                 | Cache-entry axis | Intended use                                            |
| -------------------------- | ---------------------------------------------- | ---------------- | ------------------------------------------------------- |
| `PackedBlockCacheLayout`   | One `[layer, kernel_block, packed_bytes]` pool | `1`              | Uniform-layer block resources on the default backend    |
| `LayerRowBlockCacheLayout` | One compact-row pool per resource              | `1`              | Heterogeneous layer membership on the default backend   |
| `PackedStateCacheLayout`   | One `[state_slot, packed_bytes]` pool          | `0`              | Default state resources                                 |
| `DlinferBlockCacheLayout`  | One contiguous typed tensor per block resource | `1`              | dlinfer operators requiring resource-contiguous storage |
| `DlinferStateCacheLayout`  | One contiguous typed tensor per state resource | `0` or `1`       | dlinfer state operators                                 |

Cross-layer block-major packing is not a portable default. Add a new physical
layout only when the consuming backend kernels support its strides and the
layout exposes truthful owning pools and entry axes.

The backend provider is stateless. It selects layouts; it does not allocate on
its own schedule, retain engine state, or own CPU/device transfers.

## Logical and Kernel Blocks

`CacheConfig.block_size` and `CacheConfig.kernel_block_size` describe different
units:

- A logical block is the scheduler and prefix-cache unit.
- A kernel block is the physical page unit used by backend kernels.

`BlockCachePlan.kernel_blocks_per_logical_block` records their exact integer
ratio. Allocation converts logical block counts to kernel block counts once:

```text
num_kernel_blocks = num_logical_blocks
                    * kernel_blocks_per_logical_block
```

The logical block size must be greater than or equal to the kernel block size
and exactly divisible by it. Sizing uses a one-logical-block allocation on the
PyTorch `meta` device, so sizing and real allocation exercise the same selected
layout.

## Runtime Views and Movement

The runtime exposes two model-facing forms during migration:

- Uniform legacy caches use `gpu_cache` / `cpu_cache`, grouped by layer.
- Named or heterogeneous caches use `block_caches` and
  `named_state_caches`; `NamedCacheView.layer()` resolves compact layer rows.

The owning allocation remains alive even when a compatibility list or mapping
is returned. Do not treat those facades as memory owners.

After host and device allocation, `CacheEngine._build_swap_pairs()` validates
and records corresponding owning pools. CPU and device entries must have the
same pool count, entry axis, dtype, and per-entry payload shape. `swap_in()` and
`swap_out()` then index those pre-resolved axes on the cache stream.

`StateCacheEngine` similarly resolves each state allocation's slot axis once.
Initialization and copies operate on those resolved entries, not on an assumed
single `mem_pool`. State-copy indices are host integers; source and destination
ranges must be in bounds, destinations unique, and source/destination sets
disjoint.

Same-device block copies, host/device swaps, and external PD or remote-cache
transfers are distinct operations. Do not force them into one primitive merely
because they move cache data.

## Compatibility Boundaries

The package still preserves several migration paths:

- `CacheAllocation` can be unpacked as the legacy `(mem_pool, caches)` pair.
- `CacheEngine.allocate_caches()` remains a class-level compatibility facade.
- Existing dlinfer releases may monkey-patch that allocator and return a legacy
  tuple. Runtime allocation and sizing detect that override and keep using it.
- Anonymous `cache_shapes` and `states_shapes` remain accepted alongside named
  specifications.
- `gpu_cache`, `cpu_cache`, and `full_gpu_cache` / `full_cpu_cache` remain
  available to existing consumers.

These are compatibility surfaces, not the preferred ownership model. New
native code should return `CacheAllocation`, keep resource names stable, and
put physical layout selection in the backend cache provider.

PD migration currently rejects multi-pool named block-cache allocations. That
restriction belongs to the external transfer path and must not silently change
local allocation semantics.

## Correctness Invariants

- Resource names and order are stable from schema construction through typed
  allocation views.
- Layer membership is explicit; unrelated resources need not share a global
  layer count or the same layer ids.
- Native device and host runtime allocations reuse one retained block-cache
  plan; the temporary patched-allocator compatibility path is the explicit
  exception.
- Sizing and native runtime allocation use the same resource and layout path.
- `BlockCachePlan` owns no tensors, and `CacheAllocation` owns all storage
  needed by its views.
- Byte accounting counts owning pools, never overlapping typed views.
- Every owning pool exposes the correct cache-entry axis.
- CPU and device swap entries have matching axes, dtypes, and payload shapes.
- Backend code selects physical mechanisms; `CacheEngine` owns allocation
  lifetime, validation, streams, events, and public transitions.
- Cache planning and layout construction stay outside the per-token hot path.
- Do not encode backend contiguity or packing requirements as model
  configuration flags.

## Where to Make Changes

| Change                                                             | Primary owner                                                  |
| ------------------------------------------------------------------ | -------------------------------------------------------------- |
| Payload size, alignment, resource names, or layer membership       | [`schema.py`](./schema.py)                                     |
| Logical/kernel block conversion or finalized block access metadata | [`plan.py`](./plan.py)                                         |
| Default owning-pool arrangement or allocation metadata             | [`layout.py`](./layout.py)                                     |
| Backend layout selection contract                                  | [`backends/cache.py`](../../backends/cache.py)                 |
| Default layout policy                                              | [`backends/default/cache.py`](../../backends/default/cache.py) |
| dlinfer contiguous layouts                                         | [`backends/dlinfer/cache.py`](../../backends/dlinfer/cache.py) |
| Device/host allocation lifetime, compatibility views, or swaps     | [`engine.py`](./engine.py) `CacheEngine`                       |
| State initialization or local slot copies                          | [`engine.py`](./engine.py) `StateCacheEngine`                  |
| Logical block or state-slot assignment                             | Paging scheduler and state manager, outside this package       |
| PD or remote-cache transfer                                        | [`disagg`](../../disagg), outside the local layout contract    |

Avoid adding a generic manager, registry, or utility module. Split a module only
when the new component owns a complete decision, lifecycle, or independently
testable invariant.

## Tests

Tests follow the package ownership boundaries:

- [`test_cache_schema.py`](../../../../tests/pytorch/engine/test_cache_schema.py):
  payload sizing, layer maps, named resources, and legacy normalization.
- [`test_cache_layout.py`](../../../../tests/pytorch/engine/test_cache_layout.py):
  owning pools, physical layouts, backend selection, dlinfer contiguity, and
  block plans.
- [`test_cache_engine.py`](../../../../tests/pytorch/engine/test_cache_engine.py):
  runtime ownership, compatibility views, swaps, sizing, and state operations.

Run the focused suite with:

```bash
python -m pytest -q \
  tests/pytorch/engine/test_cache_schema.py \
  tests/pytorch/engine/test_cache_layout.py \
  tests/pytorch/engine/test_cache_engine.py
```
