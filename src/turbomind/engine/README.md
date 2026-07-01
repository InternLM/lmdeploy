# TurboMind Engine Async Execution Model

## addressing

Address a top-level section by its heading, such as `concepts`, `principles`, `ownership`, `invariants`, `contracts`, or `checklist`.

Address a leaf by `<section>.<leaf>`, using the top-level section plus the `###` heading text. Examples: `concepts.phase`, `ownership.cache`, `invariants.cleanup`, `contracts.cache-prepare`, `checklist.cache-memory`.

Leaf headings are local to their section and intentionally short. Do not introduce a front index, global prefix, root id, or sentence-length id.

## scope

### status

This document is the normative developer contract for the TurboMind C++ engine execution model. It covers the host-side engine loop, model executor handoff, scheduler transaction, request lifecycle, cache metadata, prefix ownership, and module-level `BatchOp` contracts under `src/turbomind/engine` and the TurboMind model modules that participate in `BatchOp`.

This document does not describe the PyTorch engine. It is not a refactor proposal and does not prescribe a new scheduler. It records the concepts, ownership rules, invariants, and contracts that current and future TurboMind changes must preserve unless this document is updated in the same change.

When code and this document disagree, treat the disagreement as a design bug. Either fix the code to satisfy the contract or update this document with the new contract and the reason for the change.

## concepts

### request

`Request` is the API-facing unit of work. It owns request identity, a history/KV offset (`step`), generation configuration, input and output tensor references, cancellation state, callbacks, and the externally visible request state.

### sequence

`Sequence` is the engine-local mutable execution state for one accepted request on one local rank. It is created from a `Request` during admission and is the object passed through scheduler and model-module contracts. It stores token progress, scheduling decisions, logical block handles, cache-category request state, generation rows, lifecycle flags, and transient per-pass fields.

### multimodal-spans

`Sequence::multimodal_spans` is the engine-visible `(token span, fingerprint)` projection of multimodal inputs; `multimodal_inputs` (pixels) stays opaque.

`cache_prompt_boundary_skip` is the engine knob for the trailing volatile-suffix length; `Sequence::prompt_boundary_pos` is its per-sequence resolved boundary `B = prompt_len - cache_prompt_boundary_skip`.
`cache_prompt` / `cache_generation` are the two `CacheMode` publication knobs; `cache_checkpoint_interval` is the recurrent-checkpoint spacing (`CacheRegistry::checkpoint_min_interval`, > 0).

### batch-data

`BatchData` is a reusable phase-local carrier between the engine thread and the model executor thread. It contains the phase id, current and previous batch sizes, the active-batch permutation, token-count metadata, and CUDA events used to order host setup and device execution.

### phase

Phase is one slot in the async pipeline. With one phase, the engine behaves synchronously: a submitted batch is updated before the next batch is prepared. With multiple phases, host scheduling and setup may run ahead of model execution by reusing different `BatchData` slots.

### scheduler-transaction

Scheduler transaction is one scheduling pass over eligible `Sequence` objects. For each request the scheduler plans (`Resume` for inactive, `Continue` for active): it sizes logical blocks, reserves cache ids, computes `resume_len`, and emits restore copy plans. `Scheduler::Schedule()` then commits: it decides which requests become active, assigns `history_len` and `input_len`, commits cache allocation and eviction through the memory replay, selects and attaches checkpoint publication slots, emits publication copy plans, and records producer marks.

### logical-block

Logical block is scheduler-owned metadata for a fixed token interval. It records offset, capacity, current size, cache-object slots, prefix-index identity, an intrusive strong refcount (request and fork references held through `BlockHandle`s; the cache-allocation reference taken via `Retain`/`Drop` keyed on the slot's `CacheBlock::owner` identity), and node-level producer ownership.

### cache-object

Cache object is an object-typed allocation handle tracked by `CacheBlockPool` and backed by `ObjectAllocator`. The scheduler owns cache object lifetime, allocation metadata, validity, and release; modules own the meaning and contents of their registered byte ranges within the object. A cache object may be composite: one handle whose bytes are several independent sub-allocations (parts), resolved to multiple `(address, bytes)` segments.

### module

Module is any TurboMind model component that participates in `LanguageModel::Run(BatchOp, phase, env)`, such as input processing, attention, GDN, generation, or output processing. Modules may validate and prepare their own state, but they must obey the `BatchOp` contracts in this document.

### signal

Signal is a callback scheduled from the engine into the `Gateway` signal thread. Signals update externally visible request state and invoke user callbacks outside the engine scheduling thread.

### gateway

Gateway accepts external requests into per-queue `RequestQueue` objects and owns the signal thread used for callbacks. Queue operations may happen concurrently with the engine loop, but accepted requests enter engine-owned mutable state only when the engine thread pops them from the gateway.

### engine-thread

Engine thread runs `Engine::Impl::InternalThreadEntry()`. It owns request admission, validation, cancellation observation, scheduling, host-side setup, completed-batch update, lifecycle retirement, and notification submission. All scheduler state is mutated on this thread.

### model-executor-thread

Model executor thread runs `ModelExecutor::Impl::InternalThreadEntry()`. It owns the CUDA execution context for `BatchOp::kPrepare`, `BatchOp::kForward`, and `BatchOp::kUnprep`. It consumes ready `BatchData` objects from the outbound queue, waits for the setup event, runs device work, records the done event, and returns the batch through the inbound queue.

### data-path

The request data path is:

```text
Request
  -> Sequence
  -> BatchData and module-owned per-phase buffers
  -> device/module state
  -> BatchData and module-owned per-phase buffers
  -> Sequence
  -> Request outputs and signals
```

The engine and executor exchange `BatchData` slots through queues. Each slot has a stable phase id. The phase id selects module-owned per-phase buffers, while the batch slot itself carries the current active membership and CUDA ordering events.

## principles

### engine-state

The engine thread is the owner of request scheduling state. It admits requests, mutates `Sequence` lifecycle fields, runs scheduler transactions, calls host-side module-level `BatchOp` handlers, submits batches, processes completed batches, and releases request-owned state.

### scheduler-boundary

The scheduler is the transaction boundary for shared execution resources. Request-level planning (`Accept`, `Resume`, `Continue`) may match or create logical blocks, reserve cache ids, compute `resume_len`, and emit copy intent, but allocation, eviction, active admission, `history_len`, `input_len`, publication slot attachment, and producer marking are committed by `Scheduler::Schedule()`.

### cache-semantics

Modules own registered byte-range semantics. The scheduler may know that a cache id has an object id, an allocation handle, and a timestamp; it must not know whether bytes in that object contain KV blocks, GDN recurrent state, publication checkpoints, or future module state.

### resume-proof

Generic cache validity is a lifetime fact, not a resume proof. A valid allocation can keep a prefix node alive, but only `Scheduler::Resume()` may decide whether cached state lets a request skip tokens, and only from content proven produced: `is_valid` set by publication for indexed nodes, `filled_len` for private blocks.

### device-content

Device content operations happen on the model executor thread. Module-specific content work (clearing or post-processing a module's own byte range, preparing pointers, reading model outputs) belongs to the relevant `BatchOp` handler. Whole-object cache copies planned by the scheduler as `(src, dst)` cache-id pairs are resolved to addresses during engine-thread setup and performed by the executor: restore copies before `BatchOp::kPrepare`, publication copies after `BatchOp::kUnprep`. The scheduler never knows what the copied bytes mean; modules never know why a copy happened. Resolving a composite handle yields one or more segments, so a scheduler-planned whole-object copy fans out to one device copy per part (same `(src_id, dst_id)` cache-id plan; only the engine-thread resolution multiplies).

### delayed-cleanup

Async execution requires delayed cleanup. A request that has finished or been canceled must be excluded from future scheduling immediately, but its request-owned resources cannot be released until every submitted batch that references it has completed and decremented `inflight`.

### callbacks

Externally visible callbacks do not run on the engine scheduling path. The engine records callback work as signals and the gateway signal thread invokes them with the appropriate external context.

### boundary-policy

Partial-block boundary publication is decided entirely at Accept-time in `SetupForks` (prompt) and at finalization in `PublishGeneration` (generation), from two `CacheMode` knobs — `EngineConfig::cache_prompt` (`all`|`auto`) and `EngineConfig::cache_generation` (`all`|`auto`|`none`) — parsed once into `Scheduler::prompt_cache_mode_` / `generation_cache_mode_`. There is no runtime veto object. `cache_prompt=all` publishes the partial prompt `fork_to` node whenever `B` is mid-block and arms the block-aligned checkpoint clamp otherwise; `cache_prompt=auto` publishes the partial node only when its token range `[j*bs, B)` overlaps a multimodal span (`Scheduler::HasMultimodalOverlap`) and never arms the block-aligned clamp. `cache_generation=all` indexes the terminal partial generated block and adopts the terminal recurrent frontier checkpoint; `auto` indexes full generated blocks only; `none` indexes no generated blocks. The decision is a pure function of cross-rank-identical sequence attributes (prompt geometry, `cache_prompt_boundary_skip`, `multimodal_spans`), so it is consistent across ranks. `Sequence::prompt_boundary_node` now means the boundary will be published (no deferred re-check).

## ownership

### gateway

`Gateway` owns request queues and signal delivery, routing each incoming request to a queue round-robin. It does not own engine-local execution state. After a request is accepted, externally visible completion and streaming updates are delivered by signals scheduled back through the gateway.

### request

`Request` is shared API state. It is referenced by the gateway, engine, callbacks, and request-local engine state. The engine may update `Request::cancel_flag`, `Request::ec`, and external state through `UpdateState()`, but the execution details are kept in `Sequence`.

### sequence

`Sequence` is owned by `Engine::Impl::State::rc`. It remains owned by the engine until retirement cleanup resets the owning slot. Modules may store module-specific handles in `Sequence`, but they do not own the `Sequence` object.

### batch-data

`BatchData` slots are owned by the engine/executor queues. A submitted slot temporarily owns the active membership snapshot encoded by `bs0`, `bsz`, and `perm`, plus CUDA events that order setup and execution. It does not own `Sequence` objects.

### scheduler

`Scheduler` is owned by `Engine::Impl`. It owns the `CacheRegistry` (registration is closed before construction), `LogicalBlockPool`, `PrefixTrie`, and `CacheBlockPool`, and it holds a reference to the engine-owned `ObjectAllocator`. `LogicalBlockPool` is a prefix-agnostic node factory and recycle policy; `PrefixTrie` owns prefix indexing (`Find`/`Search`/`Insert`/`Erase`). The scheduler wires them with `LogicalBlockPool::set_recycle_hook`, so when a node's refcount reaches zero the pool fires the hook to erase it from the trie index before the node is destroyed. Scheduler pools persist across scheduling passes. The scheduler parses `EngineConfig::cache_prompt` / `cache_generation` into `CacheMode` values used for partial-block boundary publish decisions (`concepts.boundary-policy`).

### object-allocator

`ObjectAllocator` owns the backing cache memory region and allocation validity. `CacheBlockPool` stores object ids, allocation handles, timestamps, and a per-slot weak `owner` back-reference to the logical block the slot belongs to (see `cache-metadata`). Logical blocks point to cache ids, not raw memory.

### module-cache

Modules register anonymous byte requirements with prefix or checkpoint cache categories during construction and keep only byte offsets or base part ids (per registration channel). Each category registers one composite `ObjectAllocator` object id after all modules have registered. A category exposes two registration channels: an accumulation channel (grows part 0, returns a within-part byte offset) and a composite channel (appends parts 1..N, returns the base part id). Slab classes in `ObjectAllocator` are deduped by aligned size, and two same-aligned-size simple categories would share an object id (out of scope: prefix is the only simple category). Modules own the content semantics of their registered byte ranges. The `CacheRegistry` is a registration table only; cache id reservation, validity checks, resume selection, and release all live in the scheduler.

### generation-row

Generation rows are request-owned logical resources managed by the `Generation` module. A row is allocated lazily when a request first generates and is returned only by `BatchOp::kDel` during request cleanup.

### prefix

Prefix production is guarded by `LogicalBlock::producer`, the id of the request currently writing a block's token range. It is set for committed requests by `Scheduler::Schedule()` and cleared by the same pass's publication step for the produced range. A request must not be admitted to write a range whose blocks carry a foreign producer mark. Logical block lifetime is governed by a single intrusive refcount (`LogicalBlock::refs`). Requests and fork edges hold strong references through RAII `BlockHandle`s. When a slot's `CacheBlock::owner` is set, a valid allocation on that slot also holds a strong reference in `LogicalBlock::refs`; `CacheBlock::owner` is only a weak identity back-reference to the block a slot belongs to, and the strong allocation reference is taken and dropped explicitly via `LogicalBlockPool::Retain`/`Drop` keyed on that `owner` (request-owned slots leave `owner == nullptr` and take no allocation ref). The allocation reference is taken when the memory replay commits an allocation (`ReplayMemory`), when a finished request adopts its frontier as a terminal checkpoint (`PublishGeneration`), and when checkpoint publication attaches the publish slot to its target block (`CommitResults`); it is dropped on eviction (`ReplayMemory`), when a private block's allocations are released (`Release`), and when the scheduler drains live allocations at teardown (`~Scheduler`). `LogicalBlockPool::Drop` is the sole decrement funnel — `~BlockHandle` calls it for handle references and explicit allocation drops call it directly — and the last drop triggers `Recycle`.

### callbacks

Callbacks are owned outside the engine scheduling path. The engine creates signal closures, and the gateway signal thread invokes them.

## invariants

### seq-len

`seq_len` is the number of known tokens in `Sequence::token_ids` after the last completed update. During generation, `Update()` appends the sampled token and advances `seq_len`.

### resume-len

`resume_len` is the prefix length that can be skipped for the next scheduler transaction. `Scheduler::Resume()` computes it from the async executable upper bound, contiguous valid prefix coverage, and, when checkpoint bytes are registered, the exact restorable checkpoint or frontier position.

### readonly-block-num

`readonly_block_num` is the per-pass count of leading `Sequence::block_ids` reused read-only: fully-valid whole blocks (rounded down to a whole block) whose KV the forward reads for context but must not re-write. `Scheduler::Resume()` counts them; `Continue()` sets it to 0 (decode writes only the new token). It gates only the KV cache *stores* — they are skipped for positions `< readonly_block_num * block_size`; reads, the set of processed tokens, recurrent recomputation, and producer marking are unaffected.

### history-len

`history_len` is the committed resume point for the active forward. `Scheduler::Schedule()` sets `history_len = resume_len` only for admitted requests. Module setup and output selection use `history_len` as the start of already-available state for the submitted batch.

### input-len

`input_len` is the number of tokens admitted for the active forward. It is set by `Scheduler::Schedule()` after resource admission and allocation planning. Inactive requests must have `input_len == 0` and `history_len == 0`.

### filled-len

`filled_len` is the contiguous prefix context currently established for the request — the position a subsequent resume or decode builds on — not limited to KV this request's own forward produced. It is reconciled in two places. (1) `Engine::Update()` reconciles it from a completed forward: a generating request excludes the newly sampled token, so `filled_len` is `sequence_length - 1`; a non-generating prefill chunk uses `sequence_length`. (2) `Scheduler::CommitResults()` reconciles a resuming request to `filled_len = resume_len`, recording the prefix it reused read-only (prefix cache) or restored from a checkpoint; the `[resume_len, end)` span the in-flight resume forward rebuilds is carried by `inflight_input_len` until that forward completes. The resume-commit write never races `Update()` because a resuming request is inactive (not part of the in-flight batch).

### inflight-input-len

`inflight_input_len` is submitted prefix growth that has not yet been reflected into `filled_len`. In async mode, after update of a completed batch, an active request that was submitted into the next batch records `inflight_input_len = input_len`. This equals `input_len` even for a prefix-skipping resume because `CommitResults()` reconciles `filled_len` to `resume_len`, so the growth the forward produces (`end - filled_len`) is exactly `input_len`.

### inflight-new-tokens

`inflight_new_tokens` is submitted sequence-length growth that has not yet been reflected into `seq_len`. In async mode, after update of a completed batch, an active generating request records `inflight_new_tokens = 1`; otherwise it records `inflight_new_tokens = 0`.

### executable-context

The executable context length for a scheduling pass is `seq_len + inflight_new_tokens - inflight_input_len`. Prefix matching and resume initialization must not assume that the full `seq_len + inflight_new_tokens` context is already safely reusable; the engine still needs to execute at least one token to produce logits for generation.

### generating

`generating` means the submitted forward reaches the current context boundary and can produce a next token. The engine sets it from `resume_len + inflight_input_len + input_len == seq_len + inflight_new_tokens`.

### autoregres

`autoregres` means the submitted forward is an already-active one-token decode that can take its input token from the model's autoregressive output path instead of copying prompt tokens from host memory.

### is-active

`is_active` has two time-dependent meanings that must not be collapsed. Before scheduler commit, it describes whether the request was active in the previous scheduling state and is used for resource accounting. After scheduler commit, it describes whether the request is active in the current scheduling state.

### retiring

`retiring` means the request has finished or been canceled and must never be scheduled again. It does not mean resources can be released.

### inflight

`inflight` is the number of submitted batches that still reference the request. It is incremented during `Setup()` for each active request in the submitted batch and decremented during `Update()` for the completed batch membership. A retiring request is releasable only when `inflight == 0`.

### done

`done` records request completion/cancellation for output and update logic. It is not the physical cleanup condition; cleanup is governed by `retiring && inflight == 0`.

### cleanup

The cleanup invariant is:

```cpp
if (request.retiring && request.inflight == 0) {
    Run(BatchOp::kDel, -1, env);
    scheduler.Release(request);
    remove_sequence();
}
```

### protection-set

The eviction-protection set a request stamps (`involved_cache_ids`) is exactly what it needs to run the forward — its prefix blocks and single frontier (when checkpoints are registered) — and is its **required** allocation set. Published block checkpoints are resume-time optimizations, not run-time state, and are deliberately excluded so they stay evictable: a high-priority sequence runs whenever memory fits its prefix blocks + one frontier and may reclaim its own prior checkpoints. The single checkpoint or fork source actually restored in a pass is protected for that pass via stamping its `restore_copies` source.

## contracts

### scheduler-start

A scheduler transaction starts with a list of eligible, non-retiring `Sequence` objects. The engine resets transient scheduling fields, and asks the scheduler to plan each request (`Resume` for inactive, `Continue` for active) before commit.

### prefix-prepare

When prefix caching is enabled and the request is trie-eligible, `Scheduler::Accept()` matches the prompt against the prefix trie at admission: full blocks are matched or created and indexed, the first miss may bind a partial-match source (`fork_from`), and a prompt-boundary publish node (`fork_to`) may be created. `fork_from` is always bound (any prior request may have published a prompt or generation partial node). A partial `fork_to` node is created when `B` falls inside a block and `cache_prompt` admits it: `all` always, `auto` only when the node's token range overlaps a multimodal span. The partial node carries the partial block's KV for every prefix-cached model; a recurrent model additionally publishes a recurrent-state checkpoint onto the same node (the checkpoint payload attaches only when checkpoint cache ids exist). Accept must not allocate backing memory or select `resume_len`. The reusable prompt boundary ends at `B = prompt_len - cache_prompt_boundary_skip` (the configured count of trailing volatile generation-prompt tokens, default 1, so the default excludes only the last prompt token; `B` is capped by the `seq_len-1` resume cap). A partial `fork_to` node is published only when `B` falls inside a block (`B % block_size != 0`); when `B` is block-aligned the whole-block prefix already tiles `[0, B)` and only the boundary clamp/checkpoint applies. Over-excluding (a larger skip) is safe: segment tokens are exact-compared, so a too-long suffix only shortens reuse and never causes a false hit. Accept sets `Sequence::prompt_boundary_node` when `SetupForks` decides the boundary will be published (node insert succeeded, or the block-aligned boundary case) (`concepts.boundary-policy`); KV and checkpoint publication follow at scheduler commit (`contracts.checkpoint-publish`). Every indexing site folds each image's fingerprint into the cumulative key at the block where the image starts (from `Sequence::multimodal_spans`) and stores it on that `LogicalBlock`: `Accept`'s block creation, the partial-block `Search` when a partial prompt-boundary node may be published, and `PublishGeneration` when it later indexes the prompt-tail block that block creation left private (an image start can only fall in that block; generated positions never carry one). The folding is therefore uniform across lookup and indexing, so a published prompt-tail node has the same identity a future request's `Accept` rebuilds.

### cache-prepare

Request-level planning (`Resume` for inactive requests, `Continue` for active ones) runs inside the scheduling pass before admission. It may create missing logical blocks, reserve missing category cache ids, compute `resume_len`, and emit restore copy intent as cache-id pairs. It must not allocate or deallocate backing object memory, run module callbacks, copy, clear, restore, publish, mark a request active, set `history_len`, or set `input_len`.

`Continue` maintains the request's `involved_cache_ids` incrementally rather than rebuilding it: a request active last pass committed, so none of its involved cache ids were evicted and its whole required set was allocated; only blocks appended by `EnsureBlocks` since the last plan are new (and, being freshly created, unallocated). `Resume` cannot — shared prefix nodes it references can be evicted by other requests between its passes — so it rebuilds `involved_cache_ids` from a full scan each pass.

### scheduler-commit

`Scheduler::Schedule()` is the commit step. It sorts candidate requests by `Request::unique_id`, stamps each request's `involved_cache_ids` and the sources of its `restore_copies`, tests composed resources, clamps each forward's end to a boundary candidate (a block boundary, or exactly B = prompt_len - cache_prompt_boundary_skip when `prompt_boundary_node` is set (the publish decision is finalized in `SetupForks`; the clamp fires on the pass that can reach `B`)), checks producer conflicts, selects checkpoint publication targets, and plans cache allocation and eviction with a `ScratchAllocator`. Admission and replay run in two phases (see `contracts.scheduler-admission`): `ReplayMemory` is applied once for the required tier and again for the optional tier, and each call applies only its phase's committed replay to the real allocator and then clears the replay buffer. After replay it attaches committed publication slots, emits publication copy plans, updates frontier metadata, and publishes produced ranges.

For each committed request, the scheduler sets:

```cpp
r.history_len = r.resume_len;
r.input_len = admitted;  // clamped to a boundary candidate: a block boundary, or B = prompt_len - cache_prompt_boundary_skip when prompt_boundary_node is set
r.is_active = true;
```

### scheduler-inactive

For each uncommitted request, the scheduler must leave it inactive for the current pass:

```cpp
r.is_active = false;
r.input_len = 0;
r.history_len = 0;
r.publish_target = nullptr;
r.alloc_cache_ids.clear();
r.restore_copies.clear();
r.publish_copies.clear();
```

### scheduler-admission

Admission is two-phase. The **required** tier (prefix blocks + frontier) evicts up to the request's `cutoff[i]` and, on failure, defers the request and stops the pass — priority enforcement, gated by `max_evict_ts`. The **optional** tier (checkpoint publication, fork-to population) runs only after every required forward is placed, on a `ScratchAllocator` (a copy of the committed slab capacity, `MemoryState`; a committed handle is itself the `Allocation` pointer, read for its slot lists during eviction, and the handle store is never copied — `ObjectAllocator` is move-only), and reclaims only **inactive** slots (`timestamp < pass_floor`, where `pass_floor` is the pass-start timestamp) of any category via the allocator's evict/allocate path. An optional allocation that does not fit is dropped; it never evicts active state and never defers a forward.

### allocation

Allocation planning must be atomic at the transaction boundary. If a request cannot allocate all required cache objects, the scheduler must not partially mutate the real allocator for that failed suffix. Evictions and allocations are applied only for the committed prefix of the planning replay.

### eviction

Eviction is timestamp based and object-type agnostic. Evicting an allocation releases the allocation reference it held on its logical block when the slot's `owner` is set; evicting a prefix-category allocation also clears the block's `is_valid`. A block whose reference count reaches zero is recycled by the pool, which fires a recycle hook that removes it from the `PrefixTrie` index, and its fork-edge handles release as the node is destroyed.

### prefix-conflict

The scheduler may skip a request whose produced range carries a foreign producer mark and continue considering later requests. Producer conflict handling is block-level and must not reset or release the skipped request's logical blocks.

### scheduler-output

The scheduler's output is a set of current active requests plus updated scheduler metadata. The engine owns batch partitioning, permutation construction, setup submission, update processing, and retirement after the scheduler transaction.

### batchop

`BatchOp` is the module-level operation protocol used by `LanguageModel::Run()`. Each operation has a narrow contract. A module may ignore operations that do not apply to it. There is no module-level scheduling operation; scheduler cache preparation owns host-side cache reservation and resume selection.

### batchop-add

`BatchOp::kAdd` runs on the engine thread when new `Sequence` objects are admitted. It initializes module-specific request fields and validates request-local inputs. It may set `Sequence::status` to reject a request. It must not require scheduler logical blocks or cache object allocations.

### batchop-setup

`BatchOp::kSetup` runs on the engine thread after scheduler commit and before batch submission. It consumes committed active requests and scheduler metadata. It prepares host and device metadata buffers, copies non-cache input metadata, may resolve committed cache allocation handles to raw addresses, and may update request-owned module handles that describe the submitted work. It must treat the scheduler decision as fixed.

### object-address

Resolving an `ObjectAllocator` allocation handle to an address is metadata preparation, not backing-memory access. The address may be copied as a pointer value. The engine thread must not dereference that address or issue copies, clears, restores, publishes, kernels, or other operations whose source or destination is the cache object backing memory. A handle resolves to one or more `(address, bytes)` segments (one for a simple object, N+1 for a composite); the same engine-thread restriction applies to every segment. A handle is a typed `const Allocation*` that dereferences directly to a stable `Allocation` holding the per-part `bases`; identity and staleness are owned by a single always-on mechanism — a monotonic `Allocation::key` that a consumer snapshots and later compares (there is no compile-time backend split). Each slab slot stores its owning handle (`slot_owner_`), the reverse link a future compaction pass uses to rewrite the one `Allocation` that owns a relocated slot.

### batchop-prepare

`BatchOp::kPrepare` runs on the model executor thread after the setup event is visible on the executor stream and after scheduler-planned restore copies have been enqueued. It prepares device-side state for forward execution. It may use raw cache object addresses prepared by setup and perform module-owned byte-range content operations, such as clearing state for requests whose forward starts at position 0 (`history_len + inflight_input_len == 0`) or post-processing restored content.

### batchop-forward

`BatchOp::kForward` runs on the model executor thread. It executes model computation for the submitted batch, mutates module device state for the active requests, writes sampled output ids when generation is active, and updates device-side finished and sequence-length state. KV cache writes are bounded below by `readonly_block_num * block_size`; positions in read-only leading blocks are read but not re-written.

### batchop-unprep

`BatchOp::kUnprep` runs on the model executor thread after forward execution and before scheduler-planned publication copies are enqueued. It exports device-side results needed by the engine update path into per-phase module buffers and is the module's last chance to finalize frontier contents before publication snapshots them. It must not invoke external request callbacks.

### batchop-fetch

`BatchOp::kFetch` runs on the engine thread after the completed batch's done event is visible on the engine stream. It schedules copies from per-phase module buffers to host-visible buffers and publishes fetched tensors into `env` for `kUpdate`.

### batchop-update

`BatchOp::kUpdate` runs on the engine thread after fetch copies have completed and the engine stream has synchronized. It updates request-local host state from fetched results and module-owned host buffers. It may update generation sampling state and other CPU-side bookkeeping. It must not release request-owned resources.

### batchop-del

`BatchOp::kDel` runs on the engine thread during retirement cleanup before `Scheduler::Release()`. It releases module-owned request resources such as generation rows. It must tolerate partially initialized request state and must not depend on the request being active in a current batch.

### executor-only

Only `kPrepare`, `kForward`, and `kUnprep` are executed by the model executor thread. Cache object backing memory is accessed only by these module-level operations and by the executor-run, scheduler-planned whole-object copies that bracket them.

### cache-metadata

Cache metadata is generic. `CacheBlockPool` records cache ids, object ids, allocation handles, timestamps, and a weak `owner` back-reference to the logical block a slot belongs to (a valid allocation holds one strong ref on its owner). `LogicalBlock` records which cache ids are attached to a token interval. Neither type defines what an object's bytes mean. A slot caches the resolved `Allocation` handle (giving the per-part `bases` and the part count) plus an `alloc_key` snapshot for ABA-safe stale detection; the cached `allocation` being non-null is the validity flag. The pool still does not know a segment is a layer.

### cache-content

Cache contents are module-specific within registered byte ranges. `UnifiedAttentionLayer` owns KV byte-range semantics. `GatedDeltaNetLayer` owns recurrent and convolution state byte-range semantics. Future modules that register category bytes must define their own resumability and content-update rules.

### cache-reuse

A valid cache allocation is sufficient to keep an indexed prefix node alive, but it is not sufficient to make the node reusable for a request. Reuse requires the block to have been published (`is_valid`) and is revalidated by `Scheduler::Resume()` on every pass.

### resume-selection

`resume_len` is selected by `Scheduler::Resume()`. Without checkpoint bytes it is the contiguous prefix-valid token end, capped by the async executable upper bound. With checkpoint bytes it is the latest position among the request frontier, published block checkpoints, and fork sources that is covered by valid prefix content; restore intent is expressed as copy plans into the frontier. Generic cache validity alone never raises `resume_len` past content that was not proven produced (`is_valid` for indexed nodes, `filled_len` for private blocks). `resume_len` (what every stateful module skips) is distinct from `readonly_block_num` (the KV-store boundary): full validity of leading whole blocks marks them read-only for KV stores even when checkpoint coarseness keeps `resume_len` lower, so the re-processed window `[resume_len, readonly_block_num * block_size)` rebuilds recurrent state without re-writing already-valid KV. When a published prompt-boundary node exists (`prompt_boundary_node`), a duplicate (or history-extending) prompt resumes via fork-extension at the producer's prompt-boundary node end `B` (the producer's `prompt_boundary_pos = prompt_len - cache_prompt_boundary_skip` on the source node) by restoring the node's KV, plus its recurrent-state checkpoint when the model is recurrent; full-block prompt and generation checkpoints remain always-on regardless of the knobs.

### category-registration

Modules register anonymous byte requirements with the prefix or checkpoint category during construction and keep only byte offsets or base part ids (per registration channel). Each category registers one composite `ObjectAllocator` object id after all modules have registered. A category exposes two registration channels: an accumulation channel (grows part 0, returns a within-part byte offset) and a composite channel (appends parts 1..N, returns the base part id). Slab classes in `ObjectAllocator` are deduped by aligned size, and two same-aligned-size simple categories would share an object id (out of scope: prefix is the only simple category). Modules own the content semantics of their registered byte ranges. The `CacheRegistry` only maps categories to object ids and byte offsets; cache id reservation, validity, resume selection, and release are scheduler policy. When a module sizes its composite parts to equal another category's aligned object size (e.g. `GatedDeltaNetLayer` block-sizing recurrent parts to the prefix object), they share one slab class and become interchangeable at slot granularity under the already category-agnostic eviction sweep; page-granular reclamation (`slab.h`, `kMaxEmptySlabs == 0`) is the pre-existing baseline that also applies when sizes differ.

### unified-attention

`UnifiedAttentionLayer` registers its KV byte requirement with the prefix category during construction and stores the returned byte offset. During setup it resolves committed prefix cache ids from logical blocks and prepares KV pointer metadata. Reserving logical-block cache ids and validating contiguous prefix coverage is scheduler planning, not module work. It skips KV cache stores for positions in read-only leading blocks (`< readonly_block_num * block_size`) and supplies those positions from the already-valid blocks during reads.

### gated-deltanet

`GatedDeltaNetLayer` registers its recurrent/convolution state byte requirement with the checkpoint category during construction and stores the relevant offsets (per-layer conv element offsets within part 0, computed by the module; the base part id rec_base for recurrent parts). During setup it resolves the committed frontier cache part bases for each request and records which requests start their forward at position 0 (`history_len + inflight_input_len == 0`; in-flight tokens advance the frontier before this batch runs). During `kPrepare` it clears its registered parts (conv part 0 and each recurrent block part, including any rounding padding) for those requests. It does not know whether checkpoints are restored, published, or shared; those are scheduler-planned, executor-run whole-object copies. The recurrent state is a rounded-up 2D `(L_b layers × H_b v_heads)` block grid: one uniform composite part (`block_bytes_`) per block, conv unchanged. `GatedDeltaNetLayer` resolves a per-(layer-group, batch, head-group) recurrent base (composite part `rec_base + (L/L_b)*ng + (h/H_b)`, shared by all `L_b` layers of the block-row) plus a per-layer in-block element offset `linear_state_offset == (L%L_b)*H_b*cell_elems`, and one accumulated conv base (part 0) with the per-layer conv element offset, instead of one recurrent base per layer. The recurrent kernel indexes head-groups: `state_ptrs[b*ng + h/H_b] + linear_state_offset + (h%H_b)*state_size`. With `TM_GDN_BLOCK_CONFIG` unset (`L_b=1, H_b=num_v_heads, ng=1`) this reduces exactly to one base per layer at offset 0. Consumers that reuse a prompt-boundary checkpoint resume at `B` with a restored checkpoint (not position 0), so the "clear at start" path (`history_len + inflight_input_len == 0`) is unaffected.

### checkpoint-publish

Checkpoint publication is planned and committed entirely by the scheduler. Planning reserves a request-owned publication cache id. Commit knows the forward end only after admitted `input_len`, and skips nodes that already hold a valid checkpoint. Publication planning is routed mutually exclusively by the pass's forward end — a prompt-boundary group (a `fork_to` node's KV copy only when `B` is mid-block, plus the boundary checkpoint published either onto that partial `fork_to` node or onto the block-aligned boundary block, planned only when `prompt_boundary_node` is set and the forward landed at `B`, so a not-yet-reached pass allocates neither the KV block nor the checkpoint slot) and a full-block group. The full-block group is coverage-driven: it publishes iff a full block ends exactly at the forward end, subject to the configured minimum interval, with no knowledge of prompt-boundary mode. The prompt-boundary checkpoint bypasses the minimum interval. (This drops the prior behavior of suppressing a full-block checkpoint just below an upcoming prompt boundary; that checkpoint is now kept, since full-block publication depends only on coverage.) It attaches the allocated cache id to the target's checkpoint slot and emits a frontier-to-slot publication copy that the executor runs after `kUnprep`.

### prefix-identity

Prefix identity is token identity, per-image content identity, plus parent identity. Index lookup must use cumulative `PrefixKey`, exact parent identity, exact segment-token comparison, and exact comparison of the block's start-fingerprints (`LogicalBlock::image_fps`). A fingerprint is the image's opaque 256-bit content identity; an empty fingerprint never compares equal to anything, including another empty fingerprint. Blocks interior to an image carry no fingerprint of their own — their identity is carried by the cumulative key and the parent chain, since the image's first block exact-compares the fingerprint. Hash equality alone is never identity.

### prefix-ownership

Producer marking is a per-pass exclusion mechanism. `Scheduler::Schedule()` sets `LogicalBlock::producer` on the committed produced range and the same pass's publication step clears it. A request must not be admitted to write blocks carrying a foreign producer mark. There is no cross-pass ownership state to clean up on cancel.

### prefix-publish

Publication of produced ranges happens at scheduler commit, after the memory replay. Indexed nodes become `is_valid` only when the committed forward end fully covers them; private blocks become `is_valid` with their content extent tracked by `filled_len`. Device-side content arrives in submission order, so a consumer batch always executes after the producer batch that committed before it.

### cancel-release

Canceling or releasing a request drops the request's references; the order among blocks is immaterial. Private (un-indexed) blocks have their allocations deallocated immediately (dropping the allocation ref while the request ref still pins the block), then clearing `Sequence::block_ids` drops the request refs and recycles any now-unreferenced block; indexed nodes keep valid allocations alive (each allocation holds an allocation ref in `LogicalBlock::refs` via `Retain`/`Drop` on the slot's `owner`) and remain discoverable. Incomplete indexed nodes are left `is_valid == false`, so no consumer can resume from their content; they are reclaimed by eviction.

### checkpoint-adoption

Terminal checkpoint frontier adoption happens inside `Scheduler::PublishGeneration()` for normally finished, non-canceled, trie-eligible requests when the frontier allocation is valid, and is gated by `cache_generation == all`. Adoption does not test `frontier_pos`: at finalization the live recurrent buffer is guaranteed to correspond to `filled_len` because the finishing pass stored its state there and the GDN recurrence kernel bypasses its state write-back whenever the device finished mask is set, so any async over-shoot pass leaves the buffer untouched. `frontier_pos` is resume-fast-path bookkeeping committed speculatively as the scheduled forward end (`CommitResults()`), so under async lookahead it over-counts past `filled_len`; testing it would spuriously block a safe adoption. Adoption transfers the frontier cache id into the checkpoint slot of the newly indexed terminal block and transfers the allocation's reference to that block. On adoption, redundant full-block checkpoints within `checkpoint_min_interval` below `filled_len` that sit on blocks being indexed this pass (`pos > prompt_len`, still private — no consumer reference) are dropped, mirroring eviction. If the only in-window checkpoint sits on an already-shared block (`pos <= prompt_len`), adoption is skipped instead, preserving the interval without touching shared state. The terminal partial generated block is itself indexed into the prefix trie whenever `cache_generation == all`, independent of model type (full generated blocks always index): it carries the partial block's KV for every prefix-cached model, and the frontier-checkpoint adoption above applies only when the frontier id is valid (recurrent models). So the generation-boundary partial node exists only when fork matching can reach it.

### cache-eviction

Eviction may remove cache objects without module-specific knowledge. After eviction, a prefix node remains indexed only while its reference count is positive (requests, fork edges, or remaining valid allocations). Checkpoint and prefix resumability are revalidated by `Resume()` on every pass from current allocation validity. Published checkpoints are not held in any request's eviction-protection set (`involved_cache_ids`), so they age and are reclaimed before live working-set blocks under pressure. Eviction frees a cache *slot* (the allocation), not the `LogicalBlock`: a block referenced by a living sequence or a fork edge survives even with all of its allocations evicted, and is recycled only when its last reference drops.

## checklist

Before changing TurboMind async execution, scheduler, cache management, or module-level `BatchOp` behavior, verify the change preserves these rules:

### state-owner

Does exactly one component own each state mutation?

### cache-prepare

Do `Accept`/`Resume`/`Continue` only match or create logical blocks, reserve cache ids, compute `resume_len`, and emit copy intent, without backing allocation, active admission, or device content mutation? Does `Continue` maintain `involved_cache_ids` incrementally (appending only tail blocks from `EnsureBlocks`) while `Resume` rebuilds it from a full scan each pass?

### scheduler-commit

Does `Scheduler::Schedule()` remain the only active-admission, allocation, eviction, `history_len`, `input_len`, and publication-attach commit point?

### cache-semantics

Are cache object byte ranges interpreted only by the module that registered the byte range?

### cache-validity

Is generic cache validity used only for lifetime, not to raise `resume_len`?

### cache-memory

Are cache object backing-memory reads and writes limited to executor-thread `BatchOp` handlers and executor-run, scheduler-planned whole-object copies, with KV writes further limited to `[readonly_block_num * block_size, end)` (read-only leading blocks are reads only)? For composite objects, are whole-object copies issued as one device copy per part?

### delayed-release

Can a finishing or canceled request be excluded from scheduling before its resources are physically released?

### cleanup

Is every request-owned resource released only after `retiring && inflight == 0`?

### async-progress

Does async state account for submitted-but-not-yet-reflected work through `inflight_input_len`, `inflight_new_tokens`, and `inflight`?

### forward-progress

On a scheduling pass that admits nothing (empty active batch) with no in-flight work remaining (`inflight == 0` for every request), does the engine fail the highest-priority eligible request (smallest `unique_id`) with `kOutOfMemory` — rather than resubmitting empty batches indefinitely — so a request too large for the cache always receives a terminal status?

### callbacks

Are external callbacks delivered through gateway signals rather than directly on the engine scheduling path?

### prefix-ownership

If producer marking is touched, is it set only at scheduler commit and cleared by the same pass's publication step?

### module-cache

If a module registers cache bytes, does it register with exactly one category, store only its byte offset, define setup pointer resolution, and keep backing-memory reads and writes in executor-thread `BatchOp` handlers?

### boundary-policy

Are partial-block boundary publishes decided at Accept/finalization from `cache_prompt` / `cache_generation` (`CacheMode`), with no runtime veto object? Is `cache_prompt=auto` gated on multimodal overlap of the partial node's range? Is the decision a pure function of cross-rank-identical attributes? Is full-block publication kept mode-free (coverage-driven)? Is the recurrent-checkpoint spacing knob `cache_checkpoint_interval` (> 0, no block_seq_len fallback)?

### contract-sync

If this document no longer matches the intended behavior, is the contract updated in the same change as the code?
