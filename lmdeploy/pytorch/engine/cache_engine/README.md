# PyTorch Cache Engine

The cache engine turns model cache requirements into backend-selected storage
and then owns that storage at runtime. Most contributors only need the normal
path below; layout, transfer, and checkpoint details are separate extension
paths.

## Start Here

Follow four stages:

```text
built operators                    finalized K/V configuration
       |                                      |
       v                                      |
BlockCacheRequest[]                          |
       `------------------+------------------'
                          v
BlockCachePlan               finalized worker-local recipe
                          |
                          v
CacheAllocation              real CPU or accelerator storage
                          |
                          v
CacheEngine                  runtime owner and movement lifecycle
```

The boundaries are intentional:

- a request describes one built operator's requirement without choosing
  storage;
- a plan combines all requirements and retains one backend-selected layout;
- an allocation owns real tensors realized from that plan;
- `CacheEngine` retains the plan and CPU/accelerator allocations and owns
  views, swap ordering, streams, events, local copy, and migration state.

Sizing and real allocation both use the same retained plan. This prevents the
driver, model configuration, and runtime allocator from independently guessing
the physical cache layout.

## One Complete Example

A DSA indexer is built with the attention implementation selected for the
current worker. Its backend implementation knows the shape and contiguity
required by its kernel, so the wrapper delegates request construction:

```python
def get_block_cache_requests(self, context):
    return self.index_impl.get_block_cache_requests(
        context.geometry, self.head_dim)
```

The selected implementation returns a request such as:

```python
BlockCacheRequest(
    name='dsa_indexer_k',
    shape=(kernel_block_size, packed_head_dim),
    dtype=torch.uint8,
    per_row_contiguous=True,
)
```

During worker-local plan construction, the collector walks the built model,
assigns this indexer a stable row, and calls:

```python
indexer.bind_block_cache(
    BlockCacheBinding(cache_name='dsa_indexer_k', consumer_row=row))
```

The backend sees the collected tensor specifications and selects physical
storage. `CacheEngine` later realizes the plan on CPU and accelerator devices.
At forward time, the indexer uses its logical binding:

```python
cache = block_caches.row(binding.cache_name, binding.consumer_row)
```

The operator never receives a pool, stride, tensor address, or layout object.
The model never assigns cache layer IDs. This request/binding/view path is the
complete interface needed by most new cache-using operators.

## Choose a Reading Route

| If you are changing...          | Read...                        | You normally do not need...  |
| ------------------------------- | ------------------------------ | ---------------------------- |
| A cache-using operator          | Operator Requests and Bindings | layouts, pools, PD migration |
| Backend storage or copy kernels | Backend Layout and Allocation  | checkpoint lifecycle         |
| Runtime swap, copy, or transfer | Runtime Ownership and Movement | operator internals           |
| State caches                    | State Cache Identity and Slots | block-cache consumer binding |
| The construction pipeline       | Plan Construction              | PD byte segmentation         |

The remaining types are supporting details, not additional runtime managers.

## Operator Requests and Bindings

### Request ownership

A selected cache-using operator declares one physical kernel-block payload
through `get_block_cache_requests(context)`. `BlockCacheRequestContext`
contains worker-finalized inputs, currently `BlockCacheGeometry`. The operator
already owns model- and backend-specific facts such as head dimensions and
packing, so the context does not duplicate them.

The collector discovers request methods on actual built module instances. It
uses deterministic module registration order, assigns compact consumer rows by
cache name, and calls `bind_block_cache(binding)` on every requester. It does
not use a global registry, infer layer IDs from module names, or require a
model-specific collector.

Collection happens independently for target, speculative, and memory models.
Plans remain worker-local; only their byte counts cross the executor RPC
boundary.

### Request versus tensor specification

`BlockCacheRequest` belongs to one consumer before physical grouping.
`CacheTensorSpec` is immutable metadata retained by the finalized plan after
equal requests are grouped. For requests with contracts `A, B, A`, schema
construction produces specifications equivalent to:

```text
A(consumer_rows=(0, 2))
B(consumer_rows=(1,))
```

This distinction lets consumers keep stable logical rows while the backend
places heterogeneous contracts in different physical tensors.

Standard K/V and quantization auxiliaries still come from finalized model and
cache configuration. Every additional pageable block cache comes from a built
operator request. Block plans contain only plain model tensors and operator
consumer rows; configured layer rows belong to state caches.

### Model-facing access

Two access forms coexist:

- `gpu_cache` and `cpu_cache` provide the per-layer tuple used as
  `past_key_values`;
- `block_caches.row(name, consumer_row)` resolves an operator's bound cache.

`block_caches[name]` returns the complete tensor when the name has one physical
tensor. Direct lookup raises for a heterogeneous name because the caller must
select a consumer row. `BlockCachePlan.model_cache_indices` keeps scoped named
tensors out of the standard per-layer tuple.

`NamedCacheView` owns only logical-to-physical lookup. It does not own memory;
the corresponding `CacheAllocation` must remain alive.

If you are only adding an operator-owned cache, you can stop here.

## Plan Construction

`build_block_cache_plan()` is the worker-local composition boundary:

```text
finalize BlockCacheGeometry
        |
        v
collect BlockCacheRequest[] and bind consumers
        |
        v
build ordered CacheTensorSpec[]
        |
        v
CacheBackend selects BlockCacheLayout
        |
        v
BlockCachePlan
```

`BlockCachePlan` retains:

- ordered tensor specifications and model-facing access metadata;
- the selected physical layout;
- the kernel-page count represented by one logical scheduler block.

It owns no tensors, streams, events, or movement policy. A finalized plan is
required for sizing and `CacheEngine` construction; runtime allocation does
not rebuild a configuration-only fallback.

Sparse-MLA cache policy is finalized before executor construction, operator
building, and plan construction. `CacheEngine` does not mutate cache policy
while sizing or allocating.

## Backend Layout and Allocation

`CacheBackend.build_block_layout()` decides how ordered tensor specifications
are stored. An atomic layout implements one storage policy:

| Layout                       | Storage policy                                                 |
| ---------------------------- | -------------------------------------------------------------- |
| `PackedBlockCacheLayout`     | Pack compatible full-model tensors into one byte pool          |
| `RowBlockCacheLayout`        | Give row-scoped tensors independent padded byte pools          |
| `ContiguousBlockCacheLayout` | Give each specification an independent contiguous typed tensor |
| `PackedStateCacheLayout`     | Pack state tensors behind one slot axis                        |
| `ContiguousStateCacheLayout` | Give each state specification an independent typed tensor      |

`CompositeBlockCacheLayout` combines ordered atomic layouts without making a
selection decision. For example:

```text
BlockCachePlan
`-- CompositeBlockCacheLayout
    |-- PackedBlockCacheLayout(K, V)      -> packed pool
    `-- ContiguousBlockCacheLayout(index) -> contiguous typed pool
```

The default backend selects packed storage for plain full-model tensors, row
storage for compact consumer tensors that accept padded strides, and
contiguous storage when `per_row_contiguous=True`. Dlinfer instead selects the
shared contiguous layouts required by its kernels.

The plan realizes a logical block count through its selected layout:

```python
allocation = plan.allocate(num_logical_blocks, device)
```

`CacheAllocation.pools` own storage and drive byte accounting and movement.
`CacheAllocation.tensor_views` are typed views in tensor-specification order.
Count bytes from pools, never from possibly overlapping views.

Each `CachePool.entry_axis` identifies independently movable physical kernel
pages or state slots. Every other tensor axis belongs to one entry's payload.
This metadata lets runtime operations support one or many pools without
assuming tensor rank or memory order.

## State Cache Identity and Slots

State caches reuse tensor specifications, layouts, allocations, and named
views, but their runtime unit is a state slot rather than a block.
`StateCacheEngine` owns slot initialization and copying; it does not share the
block-cache plan or movement lifecycle.

A configured state tensor may cover only selected global model layers.
`LayerRowMap` validates their ordered identities and relates them to compact
tensor rows. The model accesses one row through:

```python
named_state_caches.layer(cache_name, layer_id)
```

Consumer rows and layer rows are different identities:

| Identity        | Meaning                                                     |
| --------------- | ----------------------------------------------------------- |
| Consumer row    | Assigned to one built cache-using operator                  |
| Layer row       | Compact state-cache row selected by a global model layer ID |
| Pool entry axis | Physical axis indexing movable blocks or slots              |

Layer rows are state-cache metadata. They are not used to give standard K/V or
operator-requested block caches synthetic model-layer identities.

### Indexed compact state

CUDA Qwen3.5 and GLM-5.3 Flash use unified compact state as their only
state implementation. `LMDEPLOY_ENABLE_COMPACT_STATE` has been removed: unset,
`0` and `1` no longer select different paths. Stateless models allocate no state.
Other models and non-CUDA backends retain their existing storage contracts.

The current supported targets are CUDA Qwen3.5 and GLM-5.3 Flash with AR or
matching single-layer MTP, using UniExecutor or RayExecutor. By default,
`num_gpu_blocks=0` automatically sizes KV capacity using `cache_max_entry_count`
after reserving each worker's actual compact/auxiliary state and runtime bytes.
The shared block count fits both target and MTP draft KV on every rank.
An explicit positive `num_gpu_blocks` (CLI `--num-gpu-blocks`) is optional and
is never silently reduced. PD, MemDecode and deprecated MP execution are not
supported. Mooncake Store uses the existing `model_name` / `cache_prefix`
namespace; CPU KV paging and other connectors are unsupported. The caller must
use separate namespaces for different weights/adapters. The compact fingerprint
isolates legacy/compact formats and incompatible retained state/KV layouts,
dtypes, TP, block size and target/draft execution contracts; it does **not**
identify weight contents. Reloading different weights into the same namespace
can hit stale remote entries. No automatic weight-version isolation is provided.
Unsupported configurations fail at startup; there is no legacy fallback.

Compact state requires `num_cpu_blocks=0`, with or without Store; CPU KV
paging is not implemented for this state layout. Positive CPU block counts are
rejected rather than silently accepted as an unused setting.

RayExecutor also supports DP: each DP frontend owns its own scheduler/state
manager and worker-local pools, not one shared owner across DP ranks. The
GLM-5.3 Flash validation includes TP1/DP4/EP4 with AR, MTP5 and piecewise CUDA
graphs; this is not a guarantee for every topology or strict token equality
across different batching/cache histories. Validation includes compact kernel
UT on PyTorch 2.10/A100 and serving/accuracy checks on 2.11/H200. These are
tested environments, not an exhaustive version matrix or a new PyTorch 2.10
minimum requirement.

Sleep does not have a complete watchdog: a stuck CUDA event, collective or
connector I/O can prevent it from reaching the drain point. Existing shutdown
RPC/process timeouts do not prove I/O completion, and the DP launcher's
15-second termination fallback can shorten the worker shutdown grace period.
Do not release pinned buffers or reuse cache storage merely because a wait
has timed out. Hang recovery and consistent shutdown deadlines remain separate
follow-up work; this change does not abandon I/O threads or add sleep-triggered
backend termination.

The executor derives request/checkpoint/Store capacities from the existing cache
settings and the MTP width, with owner-validated input reuse enabled. Internal
explicit capacity objects remain available for geometry tests.

The model builder declares canonical `StateCacheSpec.shape`: one exact-boundary
recurrent state or ordered K-1 convolution history, not a speculative ring.
Its internal `state_cache_is_indexed` contract also selects the matching backend
at construction; it is not a public feature switch. Auxiliary KPool tails retain
their own shapes. No legacy-to-compact shape conversion is performed.

`SSMStatePlan.from_model_config()` compiles these declarations and retains
the backend layouts for both sizing and allocation. Passing that plan as
`StateCacheEngine(..., compact_plan=plan)` replaces the legacy allocation:

- recoverable fields have `capacity.num_units` entries;
- runtime-only auxiliary fields retain their shape and have
  `capacity.request_slots` entries;
- model-facing declaration order and names stay unchanged;
- connector pools contain only the compact recovery fields, not auxiliary tails.

The caller must select indexed kernels and validated per-forward unit IDs with
this allocation. Auxiliary consumers still use request-slot IDs, not unit IDs.
Legacy state copy/zero APIs reject compact mode; checkpoints retain references
and kernels write new private outputs. Non-indexed model consumers retain their existing allocation.
Stateless model plans allocate no state pools and do not select a backend.

Standard AR/MTP uses a private decode
bank of `W = 1 + num_speculative_tokens` units. A newly admitted prefill leases
one private destination; protected capacity credits guarantee promotion to the
full bank when decode is scheduled. Shared checkpoints and in-flight Store
reads remain outside that writable bank. An already promoted bank is retained
across later prefills while its submitted accesses still require it.

This is phase-specific **unit-ID leasing**, not per-request GPU allocation:
the owning pool is allocated up front with `1 + R * W + B + S` units, including
the reserved zero unit, request capacity `R`, checkpoint budget `B`, and Store
budget `S`. AR uses one unit; MTP3 uses four. No request or KV-block capacity
is implicitly reduced. Only an owner-held private input may be reused as the
last output, after all old-state reads. Shared checkpoints, in-flight Store
units and the zero unit are read-only. Empty prefill rows do not write; a
nonempty decode cannot roll back to an input already overwritten.

`SSMStateManager` owns host-side allocation and lifetime. The worker's
`SSMStateCursor` only tracks current unit IDs by stable request slot;
`bind`, `advance`, and `rebind` are the execution-planning interface, not
allocation or release operations. `SSMStateMetadata` carries submitted ID
tables, while `SSMPrefillMetadata` carries one-forward token/chunk layout and
checkpoint exports. Compact advancement requires the worker cursor; there is
no metadata-only serving fallback that can leave its current IDs stale.

RL weight updates reuse the existing sleep/update/wakeup lifecycle: drain GPU
and Store work, drop local requests/checkpoints and cache registrations, then
rebuild cache engines, compact cursor IDs and CUDA graphs after successful
weight loading. Old state is not restored across weight updates. Compact has
no separate reload transaction or per-model bucket ledger. General reload
readiness/IPC fixes are deferred patches, not guarantees of this PR. In-flight
non-DP sleep also retains the baseline limitation: callers must finish requests
before sleeping. The sender's finished marker is not a server-side proof of
complete parameter coverage.

Rebuilding local caches does not delete remote Mooncake entries. Callers must
isolate weight versions using the Store namespace or invalidate remote data
before reuse. A remote deletion API is separate follow-up work, not part of this
implementation. Capacity-denied Store saves reuse the existing skip counter,
with credit/unit reasons counted only after CPU packet preparation succeeds.

## Runtime Ownership and Movement

After CPU and accelerator allocation, `CacheEngine` resolves corresponding
pool pairs once and verifies their entry axes, dtypes, counts, and per-entry
payload shapes. Swap operations then reuse those pairs on the cache stream.

Three movement operations remain separate because they have different owners
and constraints:

- CPU/accelerator swap moves entries between matching local allocations;
- same-device block copy copies scheduler-sized logical blocks;
- PD/LMCache/Mooncake transfer moves registered allocation pools between
  endpoints.

Layout selection supplies physical mechanisms but does not own these runtime
lifecycles.

### Logical and kernel blocks

`CacheConfig.block_size` is the scheduler and prefix-cache unit.
`kernel_block_size` is the physical page size expected by kernels. The logical
size must be an exact positive multiple of the kernel size:

```text
num_kernel_blocks = num_logical_blocks
                    * plan.kernel_blocks_per_logical_block
```

Meta sizing, CPU allocation, accelerator allocation, and local copy all use
this same relationship.

### Local logical-block copy

`CacheEngine.copy_logical_blocks()` accepts a device tensor with shape
`[2, num_pairs]` containing scheduler-block offsets. The retained plan provides
the kernel pages per logical block, while the accelerator allocation provides
every owning pool and its entry axis.

The active backend builds the physical copy primitive once. CUDA copies
contiguous pools with Triton; other backends inherit the bounded tensor
fallback unless they provide a more suitable implementation. The hot path
validates only copy-plan metadata. The caller owns block lifetimes and
source/destination relationship validation.

### Per-forward checkpoints

Checkpoint copies are one-forward side effects, not persistent model inputs:

```text
InputsMaker reserves state and resolves paging IDs
        |
        v
CacheCheckpointInputs
restore plans -> model forward -> save plans
        |                             |
        +---- CacheEngine ------------+  KV logical-block copies
        `---- StateCacheEngine -------+  state-slot copies
```

`CacheCheckpointInputs` travels beside `ModelInputs`. Keeping it separate
prevents one-shot operations from being cloned, reindexed, merged, or advanced
with persistent decode inputs. Aligned SSM checkpoints emit state plans. A
non-aligned prefill checkpoint additionally copies one partial logical KV block
through a checkpoint-owned frozen block.

### PD migration

PD migration registers every accelerator pool with a stable memory-region key.
The endpoint handshake exchanges each pool's shape, dtype, element size, and
entry axis. The planner decomposes requested blocks into contiguous byte
segments and pairs corresponding source and destination payloads before
dispatching one backend batch per remote engine.

P/D endpoints may place the entry axis differently, but their pools must keep
the same order, dtype, and logical payload shape after removing that axis.
Migration does not yet map one packed pool to several pools on the other
endpoint, and it currently requires equal logical and kernel block sizes.

## Compatibility Boundaries

The package preserves standard K/V configuration, anonymous state shapes,
model-config named state specifications, and the per-layer `gpu_cache` and
`cpu_cache` interfaces. Additional pageable block caches are declared only by
built operators.

`BlockCachePlan` and `CacheAllocation` are the only block-cache allocation
authorities. Runtime sizing, allocation, swap, copy, and PD paths do not call
the removed `CacheEngine.allocate_caches` extension point or accept raw
`(mem_pool, caches)` results. `StateCacheEngine` also realizes its selected
layout directly. Older dlinfer releases may still attach allocator methods at
import time, but neither engine consults them.

## Code Reading Routes

### Operator integration

1. [`nn/nsa.py`](../../nn/nsa.py): request delegation, binding, and forward-time lookup.
2. [`collector.py`](./collector.py): generic collection and consumer-row assignment.
3. [`view.py`](./view.py): logical name/row lookup.

### Plan and backend storage

1. [`plan.py`](./plan.py): construction boundary and retained recipe.
2. [`schema.py`](./schema.py): requests, payloads, tensor specifications, and row metadata.
3. [`backends/default/cache.py`](../../backends/default/cache.py): default layout selection.
4. [`layout.py`](./layout.py): layouts, owning pools, and typed views.

### Runtime and state

1. [`engine.py`](./engine.py): block allocation lifetime and movement.
2. [`state.py`](./state.py): state allocation and slot transitions.
3. [`migration.py`](./migration.py): PD metadata and byte-transfer planning.
4. [`../cache_inputs.py`](../cache_inputs.py): one-forward checkpoint payloads.

Avoid adding a general manager, registry, or utility module. Add a component
only when it owns a complete decision, invariant, or runtime lifecycle.

## Focused Tests

```bash
python -m pytest -q \
  tests/pytorch/engine/test_cache_engine \
  tests/pytorch/engine/test_executor_base.py \
  tests/pytorch/engine/test_model_agent.py
```
