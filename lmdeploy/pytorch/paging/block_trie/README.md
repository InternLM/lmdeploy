# PyTorch Block Trie and Prefix Caching

This package owns reusable prefix identity for the PyTorch engine. It maps full
token blocks to trie nodes, keeps the KV-cache references owned by those nodes,
and, for SSM models, associates exact recurrent-state checkpoints with selected
nodes.

Prefix caching is an end-to-end protocol rather than an isolated trie lookup.
The scheduler performs tentative matching and resource admission, the input
maker describes checkpoint copies, the model agent queues those copies on the
forward stream, and the engine loop publishes or unpins checkpoints at safe
asynchronous boundaries.

## Recommended Reading Order

1. Read this document for the ownership and lifecycle contracts.
2. Read [`BlockTrie.match()` and `BlockTrie.allocate()`](./trie.py) for the
   public trie workflow.
3. Read `_PrefillAdmissionAttempt` in [`scheduler.py`](../scheduler.py) for
   tentative-match commit and rollback.
4. For KV ownership and eviction, read [`kv_lifecycle.py`](./kv_lifecycle.py).
5. For SSM support, read [`checkpoint.py`](./checkpoint.py) before
   [`checkpoint_lifecycle.py`](./checkpoint_lifecycle.py).
6. For device-copy ordering, read
   [`inputs_maker.py`](../../engine/inputs_maker.py),
   [`model_agent/agent.py`](../../engine/model_agent/agent.py), and
   [`engine_loop.py`](../../engine/engine_loop.py) in that order.

Do not infer a match from the return value of `BlockTrie.match()`. Matching
writes tentative state directly to `SchedulerSequence`.

## Ownership Map

| Component                  | Owns                                                                                                         | Does not own                                  |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------- |
| `BlockTrie`                | Adapter roots, block identity, match/application policy, routed-expert replay, prefix-cache statistics       | Scheduler admission, concrete cache tensors   |
| `Node`                     | One trie edge, one trie-owned KV block reference, parent/children topology, optional cache-checkpoint record | Checkpoint allocation policy                  |
| `KVBlockLifecycle`         | KV reference transactions, leaf-candidate bookkeeping, KV leaf eviction                                      | Token or multimodal identity                  |
| `StateCheckpointIndex`     | Sparse checkpoint candidates and exact host verification                                                     | State slots or trie topology                  |
| `StateCheckpointLifecycle` | State slot and frozen-tail reservation, publication, pins, release, checkpoint eviction, stale cleanup       | Normal trie KV references or scheduler policy |
| `StateManager`             | Runtime and checkpoint state slots                                                                           | Prefix identity                               |
| `Scheduler`                | Tentative-match admission, rollback, and sequence resource allocation                                        | Exact matching and device copies              |
| `InputsMakerAsync`         | Compact host KV/state source-destination plans for one forward                                               | Concrete cache storage                        |
| `ModelAgent`               | Stream-ordered restore and save copies around model forward                                                  | Checkpoint publication and eviction           |
| `EngineLoop`               | Publication and pin release at asynchronous forward boundaries                                               | Prefix identity                               |

`BlockTrie` is the composition root for the lower-level trie collaborators. The
dependencies point downward: lower modules do not hold a `Scheduler` or general
`BlockTrie` back-reference.

## Core Data Model

Each adapter has a distinct root. A non-root `Node` represents one full
scheduler block and contains:

- the block hash, exact token ids, and overlapping multimodal hashes that define
  its identity;
- the logical KV block owned by the trie;
- parent and child links;
- optional routed-expert history; and
- an optional `NodeStateCheckpoint` for SSM reuse.

Python hashes only narrow the search. Token ids and multimodal identities are
checked exactly before a block or state checkpoint is reused. Different
adapters never share roots.

The trie owns one allocator reference to every attached KV block. Each sequence
sharing the block owns another reference. Auxiliary leaf and checkpoint indexes
accelerate lookup and eviction; they are not sources of truth.

### Per-sequence state

`PrefixCacheState` in
[`prefix_cache_state.py`](../../prefix_cache_state.py) groups the mutable
sequence-side protocol. `SchedulerSequence` in [`messages.py`](../../messages.py)
owns one instance:

| Field group                                | Purpose                                                                     |
| ------------------------------------------ | --------------------------------------------------------------------------- |
| `multimodal_spans`, `block_extra_identity` | Persistent multimodal identity used by trie keys                            |
| `trie_cursor`                              | Cursor at the deepest trie node reached by this sequence                    |
| `restore`                                  | Published SSM checkpoint selected by matching; pinning happens later        |
| `pending_save`                             | Unpublished checkpoint slot reserved for a forward result                   |
| `producer_save_pin`                        | Published save destination protected until its producer forward is safe     |
| `decode_checkpoint_node`                   | Latest replaceable decode checkpoint owned by the sequence                  |
| `match_start_step`                         | Step before a tentative match, used by chunking and cached-token accounting |
| `recompute_overlap`                        | Cached suffix deliberately recomputed into fresh sequence-owned KV          |
| `suppress_match_stats`                     | Excludes replay work from user-visible hit-rate statistics                  |

## Tentative Match Transaction

Prefix matching happens before final resource admission and mutates the
sequence immediately:

```text
snapshot statistics
        |
        v
BlockTrie.match(seq)
  - append shared KV references
  - advance the sequence step
  - select an SSM restore checkpoint, if needed
  - replay routed experts
  - update match statistics
        |
        v
pin SSM restore -> evict/admit resources
        |
        +-- success --> allocate runtime resources
        |               -> BlockTrie.allocate(seq)
        |
        `-- failure --> restore statistics
                       -> unpin/clear checkpoint restore
                       -> free temporary sequence references
                       -> reset match/recompute cursors
```

This non-local rollback contract is intentional. `BlockTrie` owns matching,
while the scheduler is the first layer that knows whether the request can
actually run. Any new rejection after `match()` must use the same rollback
path.

## AR and VLM Matching

For non-SSM models, `BlockTrie.match()` walks from `trie_cursor` or the
adapter root:

1. Build the key from one full block of token ids plus overlapping multimodal
   identity.
2. Find a child by hash and verify its exact identity.
3. Stop before the request's final forward work. A block is reusable only when
   its end is strictly before `num_valid_ids`.
4. Clamp the hit so the uncached suffix never starts inside a multimodal span.
5. Apply any AR-spec/MTP recompute overlap described below.
6. Acquire sequence references only for the final accepted blocks, advance the
   sequence step, and replay cached routed experts.

`BlockTrie.allocate()` continues from `trie_cursor` and attaches eligible
full blocks. On an exact collision it normally substitutes the existing
trie-owned block and releases the sequence's duplicate. Blocks in the pending
recompute overlap keep their fresh writable allocation even while traversal
continues along the canonical trie path.

## Recompute Overlap

Some strategies, currently AR-spec/MTP, need target hidden-state bridge data
from the end of an otherwise reusable prefix. The KV cache can reuse attention
state, but it does not contain those target hidden states. The strategy
therefore asks prefix matching to leave one or more cached blocks for the next
forward to recompute.

For example, assume the trie matches three blocks but the strategy requires one
recomputed block:

```text
Cached trie path:  [A shared][B shared][C shared]
Accepted cache hit:[A shared][B shared]
Next forward:                          [C fresh]
```

`C fresh` has the same tokens and prefix identity as `C shared`, but they have
different ownership:

- `C shared` remains the canonical trie block and must never be overwritten.
- `C fresh` belongs only to this sequence, so the next forward may write the
  recomputed KV into it.

This is why allocation cannot follow its normal collision behavior. When
`BlockTrie.allocate()` reaches the existing `C shared` child, it traverses that
node to preserve the canonical prefix path but keeps the sequence's fresh block
instead of deduplicating it back to `C shared`.

`PrefixRecomputeOverlap` makes the three lifetimes explicit:

| State               | Lifetime                                                    | Purpose                                                           |
| ------------------- | ----------------------------------------------------------- | ----------------------------------------------------------------- |
| `recompute_blocks`  | Sequence strategy                                           | Minimum number of cached suffix blocks the forward must recompute |
| `fresh_block_range` | One match-to-allocation transaction                         | Blocks that must keep fresh writable KV during allocation         |
| `trie_block_map`    | While the corresponding fresh blocks remain on the sequence | Maps each fresh block position to its shared trie block id        |

The fresh block range is cleared after allocation. The canonical mapping remains
longer because an SSM checkpoint published by this sequence must record the
shared trie path, not sequence-private physical KV blocks. Rollback or sequence
release clears both the fresh range and the canonical mapping while keeping
the strategy's `recompute_blocks` policy.

For SSM matching, the same concept begins at the selected state-checkpoint
step. The checkpoint supplies the recurrent state and canonical full KV blocks
before its anchor. A non-aligned checkpoint also supplies one frozen partial
block, which is copied into a private request-owned destination before forward;
any deeper cached blocks remain private and writable as well.

## SSM Checkpoint Matching

KV reuse alone is invalid for an SSM model because the recurrent state at the
matched step is also required. SSM matching therefore starts from published
state checkpoints rather than treating a KV walk as the source of truth.

`StateCheckpointIndex` uses a coarse key:

```text
(adapter_name, checkpoint_step, tail_hash)
```

Candidate steps are searched from deepest to shallowest. The exact verifier,
`verify_candidate()`, then checks that:

- the node still owns a published checkpoint;
- the full-block anchor and canonical path have the expected length;
- partial checkpoints own exactly one frozen logical block;
- the sparse key and adapter still match the node;
- the step is valid for this request and outside multimodal interiors; and
- the immutable full-prefix token and multimodal identities match exactly.

On a hit, `BlockTrie` acquires the checkpoint's canonical full KV blocks,
advances the sequence to the exact step, records `prefix_cache.restore`, and
keeps the partial destination and known suffix private. Partial checkpoints
fail closed when routed-expert tail history is requested. A KV path without an
exact published checkpoint is a miss.

`StateCheckpointMatchData` intentionally freezes full-prefix host identity and
the canonical logical KV path at publication. This makes lookup independent of
later sequence mutation and avoids a Python ancestor walk on every hit, at the
cost of metadata proportional to the saved prefix length.

For `step = q * block_size + r`, the checkpoint stays on the full-block anchor
at `q * block_size`. When `r > 0`, it owns one frozen logical KV block and
indexes the exact tail token and multimodal identity from that final partial
range. Matching attaches only the `q` canonical trie blocks; the frozen tail is
copied into a request-owned private block before forward.

## SSM Checkpoint Lifecycle

A node checkpoint has three ownership states. Its record always owns a state
slot and may additionally own one frozen logical KV block when the exact step
is not block-aligned:

| State     | Representation                                | Matchable? |
| --------- | --------------------------------------------- | ---------- |
| Absent    | `node.state_checkpoint is None`               | No         |
| Reserved  | Slot allocated, `published == False`          | No         |
| Published | Exact match data indexed, `published == True` | Yes        |

`pin_count > 0` is an overlay on a published checkpoint. A pinned checkpoint
cannot be evicted or released because an asynchronous restore or save copy may
still reference its state slot or frozen block.

Prefill checkpoints may be reserved at any positive model-forward boundary
outside multimodal spans. Decode checkpoints remain block-aligned. One
checkpoint is retained per full-block anchor; an unpinned variant may replace
the older checkpoint while reusing its resources.

### Save path

```text
InputsMakerAsync
  reserve_save(seq)
  producer partial block -> frozen block KV copy plan (when partial)
  runtime slot -> checkpoint slot copy plan
        |
        v
ModelAgent.model_forward()
  run model
  queue producer partial block -> frozen block
  queue runtime state -> checkpoint copy on the forward stream
        |
        v
EngineLoop
  publish_save(seq, pin_save=True)
  prefetch may now discover the checkpoint, but cannot evict its slot
        |
        v
forward output/event boundary
  unpin_save(seq)
```

Publication is transactional. It revalidates node attachment, slot ownership,
save step, and producer identity before adding exact metadata to the sparse
index. An abandoned or invalid reservation releases both its state slot and
optional frozen block.

### Restore path

```text
BlockTrie.match()
  select published checkpoint in seq.prefix_cache.restore
        |
        v
Scheduler / InputsMakerAsync
  pin_restore(seq) before checkpoint eviction can run
  frozen block -> request-private block KV copy plan (when partial)
  checkpoint slot -> runtime slot copy plan
        |
        v
ModelAgent.model_forward()
  queue frozen block -> request-private block
  queue checkpoint -> runtime state copy before model execution
        |
        v
EngineLoop
  unpin_restore(seq) after the forward is queued
```

Selection and pinning are deliberately separate. Matching decides identity;
the scheduler pins only a match it is attempting to admit. Rollback unpins and
clears a selected restore.

## Device-copy Contract

The input maker stores one-forward KV and state plans in
`CacheCheckpointInputs`, beside persistent model inputs. KV logical ids are
resolved through paging into the device plan consumed by `CacheEngine`; compact
state pairs remain on the host for `StateCacheEngine`. `ModelAgent` executes
both around model execution on the same forward stream.

Keep these rules:

- Do not add a CUDA/device synchronization to construct or execute a copy plan.
- Do not use GPU boolean indexing or `nonzero()` in this path.
- Keep restore before model execution and save after model execution.
- Keep copy plans as one-forward actions; do not persist them through generic
  `StepInputs` merge or reindex logic.
- Keep the checkpoint pinned until its queued source or destination access is
  safe from eviction and slot reuse.

## Eviction

KV and state pressure share the checkpoint lifecycle but remain explicitly
accounted:

- `StateCheckpointLifecycle.evict_frozen_checkpoints()` releases published,
  unpinned partial checkpoints before normal trie KV eviction.
- `KVBlockLifecycle.evict()` removes attached leaf nodes whose KV reference
  count proves that no sequence still shares them. Removing a KV leaf also
  releases its unpinned state checkpoint.
- `StateCheckpointLifecycle.evict()` may release a published, unpinned state
  checkpoint while retaining the node and its KV block. It uses
  `last_access_time` as state-only LRU order.
- KV pressure first calls `evict_frozen_checkpoints()` so an unpinned partial
  checkpoint can release one frozen block without removing its trie anchor.

Both paths revalidate candidates because their heaps and indexes are auxiliary
state that may contain stale entries.

## Correctness Invariants

- Never write new KV into a shared trie block.
- A successful match must advance the sequence before model-input creation.
- Every tentative match rejection must restore statistics and all sequence
  refs, pins, cursors, recompute ranges, and cached-token accounting.
- Adapter name and multimodal content are part of prefix identity.
- A hash match is never sufficient without exact identity verification.
- SSM reuse requires both the canonical KV path and an exact published state
  checkpoint.
- A partial checkpoint's frozen block and its consumer destination are distinct
  from trie-owned blocks and remain private while forward writes them.
- Trie nodes attach once, and eviction detaches leaves only; attached ancestor
  paths never change.
- Pinned checkpoints, their frozen blocks, and shared trie KV nodes are not
  evictable.
- Blocks in `recompute_overlap` must remain sequence-owned and writable through
  `BlockTrie.allocate()`.
- `block_size` is the scheduler/trie identity unit. `kernel_block_size` is a
  backend page-layout unit; conversion belongs in input preparation, not trie
  matching.
- Prefix matching is CPU-hot. Benchmark structural changes rather than assuming
  extra Python objects, walks, or validation are free.

## Where to Make Changes

| Change                                                 | Primary owner                                                     |
| ------------------------------------------------------ | ----------------------------------------------------------------- |
| Token, adapter, or multimodal identity                 | `trie.py`                                                         |
| Monotonic trie attachment or leaf detachment           | `node.py`                                                         |
| Sparse checkpoint keys or exact verification           | `checkpoint.py`                                                   |
| Checkpoint reservation, publication, pins, or eviction | `checkpoint_lifecycle.py`                                         |
| KV references, leaf bookkeeping, or KV eviction        | `kv_lifecycle.py`                                                 |
| Admission order or tentative-match rollback            | `../scheduler.py`                                                 |
| Per-sequence prefix-cache protocol state               | `../../prefix_cache_state.py`                                     |
| Host restore/save copy plans                           | `../../engine/inputs_maker.py` and `../../engine/cache_inputs.py` |
| Stream ordering around model execution                 | `../../engine/model_agent/agent.py`                               |
| Publication and unpin timing                           | `../../engine/engine_loop.py`                                     |

Avoid moving behavior merely to shorten a file. Split only when one component
can own a mutable resource and its invariants without a reverse dependency on
the scheduler or trie facade.

## Tests

Tests are grouped by the same ownership boundaries:

- [`test_trie.py`](../../../../tests/pytorch/paging/test_block_trie/test_trie.py):
  AR/VLM matching, allocation, identity, recompute, and statistics.
- [`test_node.py`](../../../../tests/pytorch/paging/test_block_trie/test_node.py):
  monotonic attachment and leaf detachment.
- [`test_checkpoint.py`](../../../../tests/pytorch/paging/test_block_trie/test_checkpoint.py):
  sparse SSM matching and exact verification.
- [`test_checkpoint_lifecycle.py`](../../../../tests/pytorch/paging/test_block_trie/test_checkpoint_lifecycle.py):
  reserve/publish/pin/release and state eviction.
- [`test_kv_lifecycle.py`](../../../../tests/pytorch/paging/test_block_trie/test_kv_lifecycle.py):
  KV transactions and leaf eviction.
- [`test_scheduler.py`](../../../../tests/pytorch/paging/test_scheduler.py):
  admission, rollback, and state-resource pressure.
- [`test_inputs_maker.py`](../../../../tests/pytorch/engine/test_inputs_maker.py):
  copy-plan construction and engine-loop boundaries.

Run the focused suite with:

```bash
python -m pytest -q tests/pytorch/paging \
  tests/pytorch/engine/test_inputs_maker.py
```

For changes to matching structure or exact checkpoint metadata, also run the
controlled long-prefix CPU benchmark used by the prefix-cache maintainers.
