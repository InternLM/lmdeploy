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

`Sequence` is the engine-local mutable execution state for one accepted
request on one local rank. It is created from a `Request` during admission and
is the object passed through scheduler and model-module contracts. It stores
token progress, scheduling decisions, logical block handles, cache-category
request state, generation rows, lifecycle flags, and transient per-pass
fields. Its optional `submitted` value is the complete scheduler-to-executor
description of one committed or still-outstanding row. The value contains
input/history length, query and cache-capacity geometry, the `generating` and
`autoregres` execution flags, and the producer-set effect fields
(`verification_positions`, `min_grant`, the inflight completion deltas,
`frontier_reanchor`, and `primes_proposals`). These fields are not stored as
parallel top-level `Sequence` scalars.

### multimodal-spans

`Sequence::multimodal_spans` is the engine-visible `(token span, fingerprint)` projection of multimodal inputs; `multimodal_inputs` (pixels) stays opaque.

`cache_prompt_boundary_skip` is the engine knob for the trailing volatile-suffix length; `Sequence::prompt_boundary_pos` is its per-sequence resolved boundary `B = prompt_len - cache_prompt_boundary_skip`.
`cache_prompt` / `cache_generation` are the two `CacheMode` publication knobs; `cache_checkpoint_interval` is the recurrent-checkpoint spacing (`CacheRegistry::checkpoint_min_interval`, > 0).

### batch-data

`BatchData` is a reusable phase-local carrier between the engine thread and the model executor thread. It contains the phase id, current and previous batch sizes, the active-batch permutation, token-count metadata, and CUDA events used to order host setup and device execution.

### phase

Phase is one slot in the asynchronous pipeline. With one phase, the engine
behaves synchronously: a submitted batch is updated before the next batch is
prepared. With multiple phases, host scheduling and setup may run ahead of
model execution by reusing different `BatchData` slots. Each phase carries
its reusable `BatchData` and selects module-owned phase buffers; committed
cache-allocation handles are resolved to raw addresses during engine setup
and stored in those per-phase module buffers. `ModelExecutor` consumes
phases in submission order, so every device operation of a phase is
stream-ordered before every operation of its successor; auxiliary-stream
work joins the main stream within its own phase. Changing `CacheBlock`
metadata later cannot change an address already resolved into an earlier
phase's module buffers, deallocating a slot in the preallocated cache region
does not unmap it, and `Engine::Update()` waits for done before the phase
slot is reused.

### scheduler-transaction

A scheduler transaction is one scheduling pass over eligible `Sequence`
objects. For each request the scheduler plans (`PlanResume` for inactive,
`PlanContinue` for active): it sizes logical blocks, reserves cache block
slots, computes `resume_len`, and emits restore copy plans.
`Scheduler::Schedule()` then commits: it decides which requests become
active, commits cache allocation and eviction through the memory replay,
selects and attaches checkpoint publication slots, emits publication copy
plans, records producer marks, and assigns one complete `SubmittedRow` for
every admitted request. Its input/history lengths, query bounds, cache bounds,
execution flags, and producer-set effect fields (`verification_positions`,
`min_grant`, the inflight completion deltas, `frontier_reanchor`, and
`primes_proposals`; ADR 0003)
are assigned as one value; consumers read effects and never classify rows by
engine mode. An uncommitted request with no outstanding predecessor
has no `submitted` value.

### logical-block

Logical block is scheduler-owned metadata for a fixed token interval. It records offset, capacity, current size, cache-object slots, prefix-index identity, an intrusive strong refcount (request and fork references held through `LogicalBlockPtr`s; the cache-allocation reference held as the slot's `LogicalBlockPtr pin`), and node-level producer ownership.

### cache-geometry

`cache_block_seq_len` is the physical number of rank-local tokens stored in one KV cache object. With context-parallel size `attn_cp_size`, the corresponding scheduler `logical_block_size` is `cache_block_seq_len * attn_cp_size` global token positions. Logical-block offsets, capacities, prefix-trie segmentation, block counts, and read-only boundaries use this global capacity; `UnifiedAttentionLayer` object sizing and block iteration continue to use the physical rank-local length. Each CP rank stores the positions assigned to it from the same global logical interval, so one logical block still maps to one cache object on each rank.

### cache-object

Cache object is an object-typed allocation handle tracked by `CacheBlockPool` and backed by `ObjectAllocator`. The scheduler owns cache object lifetime, allocation metadata, validity, and release; modules own the meaning and contents of their registered byte ranges within the object. A cache object may be composite: one handle whose bytes are several independent sub-allocations (parts), resolved to multiple `(address, bytes)` segments.

### module

Module is any TurboMind model component that participates in the batch-operation fanouts — the Model's generic fanout and the executor's device-bracket steps — such as input processing, attention, GDN, generation, output processing, or a composed speculative model. Modules may validate and prepare their own state, but they must obey the `BatchOp` contracts in this document.

### signal

Signal is a callback scheduled from the engine into the `Gateway` signal thread. Signals update externally visible request state and invoke user callbacks outside the engine scheduling thread.

### gateway

Gateway accepts external requests into per-queue `RequestQueue` objects and owns the signal thread used for callbacks. Queue operations may happen concurrently with the engine loop, but accepted requests enter engine-owned mutable state only when the engine thread pops them from the gateway.

### engine-thread

Engine thread runs `Engine::Impl::InternalThreadEntry()`. It owns request admission, validation, cancellation observation, scheduling, the host batch-op functions (`kAdd`, `kSetup`, `kFetch`, `kUpdate`, `kDel`), completed-batch update, lifecycle retirement, and notification submission. All scheduler state is mutated on this thread.

### model-executor-thread

Model executor thread runs `ModelExecutor::Impl::InternalThreadEntry()`. It owns the CUDA execution context for the device bracket's named steps — `BatchOp::kPrepare`, `BatchOp::kForward`, and `BatchOp::kUnprep`. It consumes ready `BatchData` objects from the outbound queue, waits for the setup event, runs device work, records the done event, and returns the batch through the inbound queue.

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

The engine thread is the owner of request scheduling state. It admits requests, mutates `Sequence` lifecycle fields, runs scheduler transactions, runs the host-side `BatchOp` operations through the Model's generic fanout, submits batches, processes completed batches, and releases request-owned state.

### scheduler-boundary

The scheduler is the transaction boundary for shared execution resources and
submitted execution geometry. Request-level planning (`AdmitPrompt`,
`PlanResume`, `PlanContinue`) may match or create logical blocks, reserve
cache block slots, compute `resume_len`, and emit copy intent, but allocation,
eviction, active admission, publication-slot attachment, producer marking,
and the complete `Sequence::submitted` value are committed by
`Scheduler::Schedule()`. Engine code and module setup consume that value; they
do not reconstruct or independently rewrite its fields.

### cache-semantics

Modules own registered byte-range semantics. The scheduler may know that a cache block slot (`CacheBlock`) records an object id, an allocation handle, and a timestamp; it must not know whether bytes in that object contain KV blocks, GDN recurrent state, publication checkpoints, or future module state.

### resume-proof

Generic cache validity is a lifetime fact, not a resume proof. A valid allocation can keep a prefix node alive, but only `Scheduler::PlanResume()` may decide whether cached state lets a request skip tokens, and only from content proven produced: `is_valid` set by publication for indexed nodes, `filled_len` for private blocks.

### device-content

Device content operations happen on the model executor thread. Module-specific content work (clearing or post-processing a module's own byte range, preparing pointers, reading model outputs) belongs to the relevant `BatchOp` handler. Whole-object cache copies planned by the scheduler as `(src, dst)` cache-block pairs are resolved to addresses during engine-thread setup and performed by the executor as bracketing steps of its device path: restore copies before the prepare step, publication copies after the unprep step. The scheduler never knows what the copied bytes mean; modules never know why a copy happened. Resolving a composite handle yields one or more segments, so a scheduler-planned whole-object copy fans out to one device copy per part (same `(src, dst)` cache-block plan; only the engine-thread resolution multiplies).

### delayed-cleanup

Async execution requires delayed cleanup. A request that has finished or been
canceled is excluded from future scheduling immediately, but its request-owned
resources are released only after every submitted batch that references it
completes and the retiring request reaches `inflight == 0`. Finishing or
canceling the request does not shorten that lifetime.

### callbacks

Externally visible callbacks do not run on the engine scheduling path. The engine records callback work as signals and the gateway signal thread invokes them with the appropriate external context.

### boundary-policy

Partial-block boundary publication is decided entirely at AdmitPrompt-time in `SetupPartialSiblings` (prompt) and at finalization in `Finalize` (generation), from two `CacheMode` knobs — `EngineConfig::cache_prompt` (`all`|`auto`) and `EngineConfig::cache_generation` (`all`|`auto`|`none`) — parsed once into `Scheduler::prompt_cache_mode_` / `generation_cache_mode_`. There is no runtime veto object. `cache_prompt=all` publishes the prompt partial sibling node (`LogicalBlock::partial`) whenever `B` is mid-block and arms the block-aligned checkpoint clamp otherwise; `cache_prompt=auto` publishes the partial node only when its token range `[j*bs, B)` overlaps a multimodal span (`Scheduler::HasMultimodalOverlap`) and never arms the block-aligned clamp. `cache_generation=all` indexes the terminal partial generated block and adopts the terminal recurrent frontier checkpoint; `auto` indexes full generated blocks only; `none` indexes no generated blocks and additionally suppresses generation-region full-block recurrent-state checkpoints (block end `> prompt_len`). The decision is a pure function of cross-rank-identical sequence attributes (prompt geometry, `cache_prompt_boundary_skip`, `multimodal_spans`), so it is consistent across ranks. `Sequence::prompt_boundary_node` now means the boundary will be published (no deferred re-check).

## ownership

### gateway

`Gateway` owns request queues and signal delivery, routing each incoming request to a queue round-robin. It does not own engine-local execution state. After a request is accepted, externally visible completion and streaming updates are delivered by signals scheduled back through the gateway.

### request

`Request` is shared API state. It is referenced by the gateway, engine, callbacks, and request-local engine state. The engine may update `Request::cancel_flag`, `Request::ec`, and external state through `UpdateState()`, but the execution details are kept in `Sequence`.

### sequence

`Sequence` is owned by `Engine::Impl::State::rc`. It remains owned by the engine until retirement cleanup resets the owning slot. Modules may store module-specific handles in `Sequence`, but they do not own the `Sequence` object.

### batch-data

`BatchData` slots are owned by the engine/executor queues. A submitted slot
temporarily owns the active membership snapshot encoded by `bs0`, `bsz`, and
`perm`, token-count metadata, and CUDA events that order setup and execution.
`BatchData` does not own `Sequence` objects.

A batch slot retains a handle to the engine-owned symmetric scratch
allocation. Setup copies only this handle. The executor exposes it through
the forward environment after vision processing. The handle preserves
allocation lifetime; scratch contents are shared across phases and reused
in executor order, not owned as persistent per-phase state.

### scheduler

`Scheduler` is owned by `Engine::Impl`. It owns the `CacheRegistry` (registration is closed before construction), `LogicalBlockPool`, `PrefixTrie`, and `CacheBlockPool`, and it holds a reference to the engine-owned `ObjectAllocator`. `LogicalBlockPool` is a prefix-agnostic node factory and recycle policy; `PrefixTrie` owns prefix indexing (`Find`/`Search`/`Insert`/`Erase`). The scheduler wires them with `LogicalBlockPool::set_recycle_hook`, so when a node's refcount reaches zero the pool fires the hook to erase it from the trie index before the node is destroyed. Scheduler pools persist across scheduling passes. The scheduler parses `EngineConfig::cache_prompt` / `cache_generation` into `CacheMode` values used for partial-block boundary publish decisions (`concepts.boundary-policy`).

### object-allocator

`ObjectAllocator` owns the backing cache memory region and allocation validity. `CacheBlockPool` stores object ids, allocation handles, timestamps, and a per-slot weak `owner` back-reference to the logical block the slot belongs to (see `cache-metadata`). Logical blocks point to `CacheBlock` slots, not raw memory.

### module-cache

Modules register anonymous byte requirements with prefix or checkpoint cache categories during construction and keep only byte offsets or base part ids (per registration channel). Each category registers one composite `ObjectAllocator` object id after all modules have registered. A category exposes two registration channels: an accumulation channel (grows part 0, returns a within-part byte offset) and a composite channel (appends parts 1..N, returns the base part id). Slab classes in `ObjectAllocator` are deduped by aligned size, and two same-aligned-size simple categories would share an object id (out of scope: prefix is the only simple category). Modules own the content semantics of their registered byte ranges. When a speculative model is composed, target attention registers first and draft attention registers second, so they own disjoint byte ranges in the same prefix-category object; this registration order is fixed by construction order in `CreateEngine`. The `CacheRegistry` is a registration table only; cache block slot reservation, validity checks, resume selection, and release all live in the scheduler.

### generation-row

Generation rows are request-owned logical resources managed by the `Generation` module. A row is allocated eagerly, before a request's first prompt submission, exactly when the speculative policy reports a bootstrap extent for that prompt length, because the bootstrapping forward writes proposals into that row. A method reporting no bootstrap keeps lazy allocation at first generating submission, as does a target-only engine with no policy. A request with no row yet contributes a null row pointer that every consumer skips. In both modes the row is returned only by `BatchOp::kDel` during request cleanup.

### prefix

Prefix production is guarded by `LogicalBlock::producer`, the id of the request currently writing a block's token range. It is set for committed requests by `Scheduler::Schedule()` and cleared by the same pass's publication step for the produced range. A request must not be admitted to write a range whose blocks carry a foreign producer mark. Logical block lifetime is governed by a single intrusive refcount (`LogicalBlock::refs`). Requests and fork edges hold strong references through RAII `LogicalBlockPtr`s. When a slot's `CacheBlock::owner` is set, a valid allocation on that slot also holds a strong reference in `LogicalBlock::refs`; `CacheBlock::owner` is only a weak identity back-reference to the block a slot belongs to, and the strong allocation reference is the slot's `LogicalBlockPtr pin`, set when the memory replay commits allocation and cleared by `CacheBlock::Deallocate` (sequence-owned slots leave `owner == nullptr` and keep an empty pin). The allocation reference is taken when the memory replay commits an allocation (`ReplayMemory`) and when a finished request adopts its frontier as a terminal checkpoint (`Finalize`, where the ownership transfer moves the ref with it); it is dropped on eviction (`ReplayMemory`), when a private block's allocations are released (`Release`), and when the scheduler drains live allocations at teardown (`~Scheduler`). Cache block slots follow a strict ownership model: every slot is owned by exactly one entity — a `LogicalBlock` (its `prefix`/`checkpoint` slots, `CacheBlock::owner` set at `Create`) or a `Sequence` (`frontier`, `owner == nullptr`); slot handles are unique `CacheBlockPtr`s whose destruction invalidates the slot (returned to the pool free list) only when its owner is destroyed (`LogicalBlockPool::Recycle` for block-owned slots, `Scheduler::Release` for sequence-owned slots); nothing else invalidates a slot. Ownership may be transferred between entities (an exchange that updates the holding field and `CacheBlock::owner` in the same statement group), which moves the invalidation duty with it. Eviction deallocates backing memory but never frees the slot: an evicted slot stays bound to its owner as an unallocated slot and consumers always test allocation validity. `LogicalBlockPool::Drop` is the sole decrement funnel for `LogicalBlockPtr` refs, allocation refs are dropped by clearing the slot pin, and the last drop triggers `Recycle`.

### callbacks

Callbacks are owned outside the engine scheduling path. The engine creates signal closures, and the gateway signal thread invokes them.

## invariants

### seq-len

`seq_len` is the number of known tokens in `Sequence::token_ids` after the last completed update. During generation, `Update()` appends the sampled token and advances `seq_len`.

### resume-len

`resume_len` is the prefix length that can be skipped for the next scheduler transaction. `Scheduler::PlanResume()` computes it from the async executable upper bound, contiguous valid prefix coverage, and, when checkpoint bytes are registered, the exact restorable checkpoint or frontier position.

### readonly-block-num

`readonly_block_num` is the per-pass count of leading `Sequence::block_ids` reused read-only: fully-valid whole blocks (rounded down to a whole logical block) whose KV the forward reads for context but must not re-write. `Scheduler::PlanResume()` counts them; `PlanContinue()` sets it to 0 (decode writes only the new token). It gates only the KV cache *stores* — they are skipped for positions `< readonly_block_num * logical_block_size`, where `logical_block_size = cache_block_seq_len * attn_cp_size`; reads, the set of processed tokens, recurrent recomputation, and producer marking are unaffected (`concepts.cache-geometry`).

### history-len

`submitted->history_len` is the committed resume point for the submitted
forward. For an ordinary row the scheduler assigns it from `resume_len`.
Module setup and output selection use it as the start of already available
state. It is assigned only for an admitted request; no `submitted` value means
there is no newly committed row.

### input-len

`submitted->input_len` is the physical number of tokens admitted for the
submitted forward. For an ordinary row it is assigned after resource and
boundary clamping. An inactive request with no outstanding predecessor has no
`submitted` value. Retaining an outstanding predecessor is not current
admission, and zeroing separate input/history fields is not a state
transition.

### filled-len

`filled_len` is the contiguous prefix context currently established for the
request, not merely KV produced by that request's own forward. It is
reconciled in two places. `Engine::Update()` uses the completed device
sequence length: a generating row excludes its newly sampled but unconsumed
token and sets `filled_len = returned_sequence_length - 1`, while a
non-generating row sets `filled_len = returned_sequence_length`.
`Scheduler::CommitResults()` reconciles a resuming request to
`filled_len = resume_len`; the in-flight rebuilt span is then represented by
`inflight_input_len` until completion. These writes do not race because a
resuming request is inactive when its resume is committed. Conservative
submitted query/cache bounds and private draft-extension bytes never advance
`filled_len`.

### inflight-input-len

`inflight_input_len` records submitted prefix growth not yet reflected in
`filled_len`. In async mode, after update of a completed batch, an active
request submitted into the next batch records the submitted row's
`inflight_input_delta`: the full physical width for an ordinary row — still
the full width for a prefix-skipping resume because `CommitResults()` first
reconciles `filled_len = resume_len` — and zero for a speculative row, whose
accepted growth is unknown until device verification completes. Physical
target width and cache capacity remain in `SubmittedRow`.

### inflight-new-tokens

`inflight_new_tokens` records host-predicted sequence growth not yet reflected
in `seq_len`. In async mode, an active ordinary generating successor records
one and every other ordinary successor records zero; the recorded value is the
submitted row's `inflight_new_delta`. A speculative row records
zero; `accept_len` is device-produced and reconciled only after fetch. The
scheduler never predicts speculative acceptance through this field.

### executable-context

The executable context length for a scheduling pass is `seq_len + inflight_new_tokens - inflight_input_len`. Prefix matching and resume initialization must not assume that the full `seq_len + inflight_new_tokens` context is already safely reusable; the engine still needs to execute at least one token to produce logits for generation.

### generating

`submitted->generating` means the row may produce committed output. Ordinary
rows retain the target-only boundary rule
`resume_len + inflight_input_len + submitted->input_len == seq_len + inflight_new_tokens`.
A speculative row is generating by construction even though its physical
query width is `K`; completed growth is the fetched `accept_len`, not a host
scalar attached at submission. Consumers use the committed flag and do not
rederive it after scheduling.

### autoregres

`submitted->autoregres` means target input IDs are gathered from the persistent
device token row after carried predecessor state is visible. For target-only
execution this remains an already-active one-token generating decode,
classified from the predecessor's committed generating state and
`submitted->input_len == 1`. A speculative row also sets it because its
`K` verification IDs are device-resident proposals. It does not mean the
physical query width is one.

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
    model_.Run(BatchOp::kDel, -1, env);
    scheduler.Release(request);
    remove_sequence();
}
```

### protection-set

The eviction-protection set a request stamps (`involved_blocks`) is exactly what it needs to run the forward — its prefix blocks and single frontier (when checkpoints are registered) — and is its **required** allocation set. Published block checkpoints are resume-time optimizations, not run-time state, and are deliberately excluded so they stay evictable: a high-priority sequence runs whenever memory fits its prefix blocks + one frontier and may reclaim its own prior checkpoints. The single checkpoint or fork source actually restored in a pass is protected for that pass via stamping its `restore_copies` source.

## contracts

### scheduler-start

A scheduler transaction starts with eligible, non-retiring `Sequence`
objects. The engine resets transient per-pass planning fields and asks the
scheduler to run `PlanResume` for inactive requests and `PlanContinue` for
active requests before commit. Planning may inspect a still-outstanding
`submitted` row; the engine does not clear submitted geometry before planning.
When `inflight == 0`, no outstanding row exists and the committed value is
cleared only if the request is rejected by commit cleanup. Otherwise the
complete submitted row remains available until a successor is committed or the
phase drains.

### prefix-prepare

When prefix caching is enabled and the request is trie-eligible, `Scheduler::AdmitPrompt()` matches the prompt against the prefix trie at admission: full blocks are matched or created and indexed, the first miss may bind the partial sibling edge (`LogicalBlock::partial`), and a prompt-boundary partial sibling node (`LogicalBlock::partial`) may be created. The matcher-side bind is always attempted (any prior request may have published a prompt or generation partial node). A partial sibling node is created when `B` falls inside a block and `cache_prompt` admits it: `all` always, `auto` only when the node's token range overlaps a multimodal span. The partial node carries the partial block's KV for every prefix-cached model; a recurrent model additionally publishes a recurrent-state checkpoint onto the same node (the checkpoint payload attaches only when checkpoint cache slots exist). AdmitPrompt must not allocate backing memory or select `resume_len`. The reusable prompt boundary ends at `B = prompt_len - cache_prompt_boundary_skip` (the configured count of trailing volatile generation-prompt tokens, default 1, so the default excludes only the last prompt token; `B` is capped by the `seq_len-1` resume cap). A partial sibling node is published only when `B` falls inside a block (`B % block_size != 0`); when `B` is block-aligned the whole-block prefix already tiles `[0, B)` and only the boundary clamp/checkpoint applies. Over-excluding (a larger skip) is safe: segment tokens are exact-compared, so a too-long suffix only shortens reuse and never causes a false hit. AdmitPrompt sets `Sequence::prompt_boundary_node` when `SetupPartialSiblings` decides the boundary will be published (node insert succeeded, or the block-aligned boundary case) (`concepts.boundary-policy`); KV and checkpoint publication follow at scheduler commit (`contracts.checkpoint-publish`). Every indexing site folds each image's fingerprint into the cumulative key at the block where the image starts (from `Sequence::multimodal_spans`) and stores it on that `LogicalBlock`: `AdmitPrompt`'s block creation, the partial-block `Search` when a partial prompt-boundary node may be published, and `Finalize` when it later indexes the prompt-tail block that block creation left private (an image start can only fall in that block; generated positions never carry one). The folding is therefore uniform across lookup and indexing, so a published prompt-tail node has the same identity a future request's `AdmitPrompt` rebuilds. The partial sibling edge is structural, not per-request intent: it lives on the indexed block, points to an identity-verified sibling with strictly smaller size (size strictly decreases along edge paths, keeping the graph acyclic even when a generation-indexed partial carries an edge), and is first-wins — bound at most once, at AdmitPrompt, on a block created in the same pass (matcher side at the miss block, creator side at the boundary block).

### cache-prepare

Request-level planning (`PlanResume` for inactive requests, `PlanContinue` for
active ones) runs inside the scheduling pass before admission. It may create
missing logical blocks, reserve missing category cache-block slots, compute
`resume_len`, and emit restore-copy intent as `CacheBlock*` pairs. It does not
commit a `SubmittedRow`, allocate or deallocate backing object memory, run
module callbacks, copy, clear, restore, publish, or mark a request active.

`PlanContinue` maintains `involved_blocks` incrementally: a request active in
the prior pass committed its required allocation set, so only blocks appended
by `EnsureBlocks` since that plan are new. `PlanResume` rebuilds
`involved_blocks` from a full scan because shared prefix nodes can be evicted
between passes. `PlanResume` may select an interior partial sibling checkpoint
when its end lies inside the contiguous valid prefix; that emits only a
checkpoint restore. A sibling extending past the prefix end retains the
fork-extension behavior of a KV copy plus checkpoint restore when the model
checkpoints. `PlanRequests()` prepares the next row, and required admission
later validates, allocates, and commits it.

### scheduler-commit

`Scheduler::Schedule()` is the commit step. It sorts candidate requests by
`Request::unique_id`, stamps each request's `involved_blocks` and every
restore-copy source, tests
composed resources, and clamps ordinary forward ends to a boundary candidate:
a block boundary, or exactly
`B = prompt_len - cache_prompt_boundary_skip` when
`prompt_boundary_node` is set and the pass can reach `B`. When checkpoint
bytes are registered and a prompt-region forward would run past
`last_ckpt_pos + checkpoint_min_interval`, its end is truncated to the last
block boundary in the admitted range that is at or past the due position and
strictly past the forward begin, so the remainder runs in the next pass. The
commit checks producer conflicts, selects checkpoint publication targets, and
plans cache allocation and eviction with a `ScratchAllocator`.

Admission and replay retain the two phases from
`contracts.scheduler-admission`. `ReplayMemory` is applied once for the
required tier and again for the optional tier; each application commits only
that tier's replay to the real allocator and clears the replay buffer. After
replay, the scheduler attaches publication slots, emits publication copies,
updates frontier metadata, and publishes produced ranges.

For every checkpointed submitted row, commit advances live frontier metadata to
the scheduled forward end. A speculative row's K-wide end is a conservative
pending marker, not a reusable exact frontier, and speculative rows are not
checkpoint-publication targets.

`PlanRequests()` prepares one maximum `SubmittedRow` per request in the existing
`ScheduleState::candidates` storage. It reads the outstanding `submitted` row,
when `inflight != 0`, only to derive the next query and cache-write offsets. It
also applies policy extent and bootstrap geometry before required admission.

`RunRequiredAdmission()` is uniform. It tests the prepared row once, allows a
positive smaller result to clamp an ordinary prefill, and commits the shortened
row. Speculative resources preserve the zero-or-full-count contract, so a
speculative row cannot enter the partial path. The pass then performs producer
validation, block extension, allocation, eviction, and commit without
re-classifying the row by engine mode or calling the speculative policy.

`Sequence::submitted = candidate` and `Sequence::is_active = true` occur only
after every required operation succeeds. A speculative candidate is admitted at
exactly its policy query count or not at all. There are no fallback retries: a
resource, producer, allocation, or replay failure leaves the candidate
uncommitted and follows the required-tier failure rule.

### scheduler-inactive

An uncommitted request is inactive for the current pass. The scheduler clears
its publication target and per-pass allocation/restore/publication-copy
intent. Required admission never overwrites an outstanding `submitted` row
before commit, so cleanup does not restore a captured row. With `inflight == 0`,
cleanup resets `submitted`. Producer conflict may continue to later requests
because it occurs before allocation and replay mutation; `CommitResults()`
clears the uncommitted request's transient vectors. Request-owned logical slots
may remain and are rebuilt by `PlanResume()` on the next pass:

```cpp
r.is_active = false;
r.publish_target = nullptr;
r.alloc_blocks.clear();
r.restore_copies.clear();
r.publish_copies.clear();
if (r.inflight == 0) {
    r.submitted.reset();
}
```

### scheduler-admission

Admission remains two-phase. The required tier covers prefix blocks plus the
frontier, evicts only up to the request's `cutoff[i]` under `max_evict_ts`, and
on failure defers that request and stops the pass so a lower-priority request
cannot pass it. The optional tier covers checkpoint publication and fork-to
population only after every required forward is placed. It operates on a
`ScratchAllocator`, which copies the committed slab-capacity `MemoryState`.
The committed handle is the `Allocation` pointer itself, read for its slot
lists during eviction; the handle store is never copied and `ObjectAllocator`
remains move-only. The optional tier may reclaim only inactive slots whose
timestamp precedes `pass_floor`. Optional failure drops that optional
allocation; it does not evict active state or defer a required forward.

`PlanRequests()` classifies the still-outstanding submitted row and prepares one
candidate per request before required admission. The required pass is uniform:
it does not select policy geometry, predict speculative acceptance, or retry a
failed candidate. Ordinary prefill may be shortened when a composed resource
returns a positive partial count. Speculative resources must return zero or the
complete policy query count, so speculative admission is indivisible. A failed
speculative resource or allocation check defers the request and stops the
required tier; it never falls back to an ordinary candidate.

### allocation

Allocation planning must be atomic at the transaction boundary. If a request
cannot allocate all required cache objects, the scheduler removes only that
request's entries from `pass.planned` and trims the failed replay suffix at
phase cleanup. The real allocator is still mutated only for the committed
prefix when `ReplayMemory()` runs.

### eviction

Eviction is timestamp based and object-type agnostic. Evicting an allocation releases the allocation reference (the slot's pin) it held on its logical block when the slot's `owner` is set; evicting a prefix-category allocation also clears the block's `is_valid`. A block whose reference count reaches zero is recycled by the pool, which fires a recycle hook that removes it from the `PrefixTrie` index, and its fork-edge handles release as the node is destroyed.

### prefix-conflict

The scheduler may skip a request whose produced range carries a foreign producer mark and continue considering later requests. Producer conflict handling is block-level and must not reset or release the skipped request's logical blocks.

### scheduler-output

The scheduler outputs current active requests with complete `SubmittedRow`
values plus updated scheduler metadata. The engine owns batch partitioning,
permutation construction, setup submission, update processing, and retirement
after the transaction. A composed speculative engine orders active target
rows as an extension-candidate prefix — speculative rows first, then bootstrap
final prompt rows (possibly multi-token), so the speculative rows form a
leading run — then ordinary generating rows, then ordinary partial-prefill
rows. Extension candidacy is every speculative row plus rows whose forward
primes the first proposals (`primes_proposals`, the scheduler-recorded
bootstrap fold); decoder row partitions rely on the
speculative leading run (ADR 0003). The partition changes
executor row order only; it does not rewrite `SubmittedRow` geometry,
scheduler priority, request ownership, or cache-block order. Target-only
execution retains generating-first order.

### batchop

`BatchOp` is the module-level operation protocol. Each operation's home implies its thread: `kAdd`, `kSetup`, `kFetch`, `kUpdate`, and `kDel` are host operations on the engine thread, while `kPrepare`, `kForward`, and `kUnprep` are the named steps of the executor's device bracket on the model executor thread; none may be ignored by the side that owns it. Each operation has a narrow contract. A module may ignore operations that do not apply to it. There is no module-level scheduling operation; scheduler cache preparation owns host-side cache reservation and resume selection. An unrecognised `spec_method` is a construction-time failure; it is never admitted and then diagnosed during a later operation.

For every operation the modules are driven in one canonical order: optional
vision, batch status, generation, input processing, target model, optional speculative model, and
output processing. The Model's run method is the generic fanout
for that order — used by every host operation and by the executor's `kUnprep`
step — skipping absent optional modules. `kPrepare` applies the same order
executor-side as a hand-written step carrying its injections: in a composed
engine the verification component's draft inputs are published after
generation's prepare, and the executor publishes the target `k_offsets` buffer
after input preparation and before target-model preparation. Components ignore
operations that do not
apply to them. `kForward` remains an explicit dataflow and
does not use the generic fanout. It is the executor's forward step, branched once per engine composition:
an ordinary engine executes the target-pass routine, a composed engine the
speculative-round routine, which owns the mixed batch (ADR 0002).

When a speculative model is composed, `output_logits`, `output_last_hidden_state`, `output_logprobs`, `return_ppl`, and guided decoding are unsupported at engine scope rather than conditionally by batch or verification position. Python clears the first four request options. Guided decoding is not cleared or rejected, but the C++ speculative path does not execute it, so asking for it produces output that is not grammar-constrained.

### batchop-add

`BatchOp::kAdd` runs on the engine thread when new `Sequence` objects are admitted. It initializes module-specific request fields and validates request-local inputs. It may set `Sequence::status` to reject a request. It must not require scheduler logical blocks or cache object allocations.

### batchop-setup

`BatchOp::kSetup` runs on the engine thread after scheduler commit and before
batch submission. It consumes committed active requests, prepares host and
device metadata, copies non-cache input metadata, resolves committed
cache-allocation handles to raw addresses, and may update request-owned module
handles describing the submitted work. Input length, history length,
query/cache capacity, and execution flags come only from the fixed
`SubmittedRow`. It
treats the scheduler decision as fixed: it does not mutate the submitted
value, read device acceptance, or replace conservative capacity with
predicted progress.

For a composed speculative model, verification positions are the maximum over
generating rows of the row-carried `verification_positions` field (one for an
ordinary generating row, the policy query-row count for a speculative row).
`BatchStatus` derives the count once at `kSetup` and publishes it for every other
module; no forward-time engine module reads `spec_num_draft_tokens`.

### object-address

Resolving an `ObjectAllocator` allocation handle to an address is metadata preparation, not backing-memory access. The address may be copied as a pointer value. The engine thread must not dereference that address or issue copies, clears, restores, publishes, kernels, or other operations whose source or destination is the cache object backing memory. A handle resolves to one or more `(address, bytes)` segments (one for a simple object, N+1 for a composite); the same engine-thread restriction applies to every segment. A handle is a typed `const Allocation*` that dereferences directly to a stable `Allocation` holding the per-part `bases`; identity and staleness are owned by a single always-on mechanism — a monotonic `Allocation::key` that a consumer snapshots and later compares (there is no compile-time backend split). Each slab slot stores its owning handle (`slot_owner_`), the reverse link a future compaction pass uses to rewrite the one `Allocation` that owns a relocated slot.

### batchop-prepare

`BatchOp::kPrepare` is the prepare step of the executor's device bracket. It
runs on the model executor thread after the setup event is
visible on the executor stream and after the bracket's restore-copy step has
been enqueued. It prepares device-side state, may use raw cache-object
addresses resolved by setup, and may perform module-owned byte-range work such
as zero-start clearing at
`submitted->history_len + inflight_input_len == 0` or post-processing restored
content. When a speculative model is composed, it also carries predecessor `finished` and
`sequence_length`, and the verification component's draft inputs — the
persistent token-row pointers among them — are published as an explicit
bracket step rather than through generation's fanout. It performs no device-to-host transfer or blocking
host synchronization.

The executor owns target key-offset production in both compositions: it
publishes the `k_offsets` buffer after input preparation and before the target
model prepares, and fills it at `kForward` — over committed sequence lengths in
an ordinary engine, over the input processor's staged per-request key lengths
in a composed one. The input processor's speculative half owns the composed
query-row layout: its forward-time build step gathers the target's input ids
from the request token rows and stages the key lengths, reading the batch's
published operands (`input_ids`, `q_offsets`, `sequence_length`, `finished`,
the verification bracket's `request_token_ids_ptrs`) plus its own
`target_ids_from_row` staging. A buffer borrowed by the target decoder during
`kPrepare` is published before the target prepares and filled at `kForward`.

### batchop-forward

`BatchOp::kForward` is the forward step of the executor's device bracket: the
visible composition branch (ADR 0002) that runs the speculative-round routine
for a composed engine and the target-pass routine otherwise. It
runs on the model executor thread. It executes model
computation for the submitted batch, mutates module device state, writes
sampled output IDs for generating rows, and updates device-side finished and
sequence-length state. KV stores remain bounded below by
`readonly_block_num * logical_block_size`; leading read-only positions are
read but not rewritten. For a speculative submission, selected target hidden rows
are position-major `[verification_positions, generating_rows, hidden]`. The
executor evaluates the target LM head once over the flattened leading dimensions,
processes the resulting distributions as one block, and performs accept/reject
decisions in position order in one verifier kernel launch containing one CTA per
submitted request. Stop-span clamping precedes the composed speculative model's
draft pass, and persistent `sequence_length` advances exactly once from final
`accept_len`.
The speculative model reads target residuals only through the tap it supplies to the
target decoder. Outside method-owned state, its only persistent cross-round mutations
are its registered cache range and the request token rows. The target pass's
transient staging runs as executor-driven steps before the target decoder: the
input processor's gathered ids and staged key lengths, the executor's
key-offsets prefix-sum, the verification component's selected-states buffer,
and the tap's arming.
Conservative private cache tails may be written but are
not committed prefix progress. No device-written acceptance, token, terminal,
or length value is read by the host in this operation, and executor forward
performs no device-to-host transfer or stream synchronization.

When target recurrent state is present, speculative target verification
computes all submitted transitions with canonical final-state stores suppressed,
journals rank-local transition inputs, and commits exactly the terminal-clamped
`accept_len` prefix before the draft pass. The journal, accepted length,
and commit remain device-side and stream-ordered.

For supported unquantized full-attention layers at CP1, a speculative target
verification partition writes K/V through `ProcessKV_v2` and invokes the
standalone CuTe paged-verification kernel once per layer, plus its independent
split reduction when needed. It does not flatten the prefix. Following
ordinary one-token rows retain the existing decode kernel; ordinary prompt
prefill may run on the auxiliary stream and joins before output projection.
Unsupported verification configurations retain whole-batch flattened prefill.
Draft refresh and extension attention remain method-owned prefill/decode work.

### target-activation

Inside the target, behavior selects on workload shape or typed caller
arguments — never on the presence of an environment key and never on engine
composition. Layer-internal selection reads row-effect data planned at
`kSetup`: the recurrent-state layer keys its store-suppressed path off its
verification-row count and consumes the store-suppression mask as data, while
the per-row speculative flag flows to device kernels as data only.
Caller-intent capabilities activate through typed `DecoderInputs` fields: a
set `selected_hidden_buffer` is the request to write selected hidden states
into the caller's buffer, available to any caller in any composition.
Environment keys carry data; producing or consuming one at the wrong moment
is a missing-data failure, not a mode change.

### target-native-capabilities

`LanguageModel` carries capabilities that exist to serve multi-position
speculative workloads; they are the target's own contract, not speculative
leakage, each with one producer and one consumer:

- `CommitAcceptedState(phase, accept_len)` — produced by the executor's
  speculative round after stop-span clamping; consumed by the target's
  recurrent-state layers, which commit exactly the accepted prefix from their
  transition journal.
- `SpeculativeStateJournalBytes(request_count, verification_positions)` —
  produced at engine construction when sizing transient verification storage;
  consumed by the recurrent-state layers' journal and commit sizing.
- `DecoderInputs.taps` — the hidden-state tap the executor passes into the
  decoder (the speculator's one target-pass hook); consumed by the decoder's
  per-layer capture.
- `DecoderInputs.selected_hidden_buffer` — the caller-owned selected-states
  storage, published by the executor from the verification component's buffer
  in a composed round; consumed by the decoder's selected-states collection.
  `selected_token_pos` is its ordinary-mode sibling.

The draft-only passthroughs (`attention_input`, `attention_metadata`, and the
`decoder_local_token_nums` topology override) remain the accepted price of
reusing the decoder for the draft (ADR 0001).

### batchop-unprep

`BatchOp::kUnprep` is the unprep step of the executor's device bracket, driven
through the Model's generic fanout. It
runs on the model executor thread after the forward step and before
the bracket's publish-copy step. It is the module's last chance to
finalize frontier contents before publication snapshots them and must not
invoke external callbacks. The speculative result export — final phase-owned
selected spans and accepted lengths — happens at `kFetch`, where the
verification component stages them into host-visible buffers for the engine
thread; `kUnprep` itself performs no speculative export. It
is the last module operation before the done event.

### batchop-fetch

`BatchOp::kFetch` runs on the engine thread after the completed batch's done
event is visible on the engine stream. It schedules copies from per-phase
module buffers into host-visible buffers and publishes fetched tensors into
`env` for update. Speculative result copies use pinned staging. Fetch does not
mutate `Sequence` or publish cache nodes.

### batchop-update

`BatchOp::kUpdate` runs on the engine thread after fetch copies complete and
the engine stream synchronizes. It updates request-local host state and
module-owned CPU bookkeeping and must not release request-owned resources.
For a speculative row it maps the completed phase row through its permutation,
reconciles exact `filled_len`, appends exactly `accept_len` committed tokens
for a generating non-retiring request, and finalizes immediately from the
resulting exact prefix.

Update identifies the completed row from phase-owned `BatchData` — which
snapshots the submitted row's `frontier_reanchor` effect at setup — never from
a possibly overwritten `Sequence::submitted`. After reconciling a completed
checkpointed speculative row, update assigns `frontier_pos = filled_len` only
when no newer phase for that request remains in flight; otherwise it retains the
newer phase's conservative scheduled marker.

### batchop-del

`BatchOp::kDel` runs on the engine thread during retirement cleanup before `Scheduler::Release()`. It releases module-owned request resources such as generation rows. It must tolerate partially initialized request state and must not depend on the request being active in a current batch.

### executor-only

The model executor is the device pipeline only: its public interface is construction and start, built around the engine-owned slot queues. Only the device bracket's `kPrepare`, `kForward`, and `kUnprep` steps are executed by the model executor thread; the host operations run engine-side on the engine thread (ADR 0004). Cache object backing memory is accessed only by these device steps and by the executor-run, scheduler-planned whole-object copies that bracket them.

### cache-metadata

Cache metadata is generic. `CacheBlockPool` owns `CacheBlock` slot storage (stable addresses); each slot records its object id, allocation handle, timestamp, and weak `owner` back-reference to the logical block it belongs to (a valid allocation holds one strong ref on its owner via the slot's `LogicalBlockPtr pin`). `LogicalBlock` records which cache slots are attached to a token interval. Neither type defines what an object's bytes mean. A slot caches the resolved `Allocation` handle (giving the per-part `bases` and the part count) plus an `alloc_key` snapshot for ABA-safe stale detection; the cached `allocation` being non-null is the validity flag. The pool still does not know a segment is a layer.

### cache-content

Cache contents are module-specific within registered byte ranges. `UnifiedAttentionLayer` owns KV byte-range semantics. Target and draft KV occupy disjoint ranges in one prefix object, so whole-object allocation, copies, and eviction preserve both ranges together. `GatedDeltaNetLayer` owns recurrent and convolution state byte-range semantics. Future modules that register category bytes must define their own resumability and content-update rules.

A GDN speculative transition journal is transient executor storage, not cache
content. Only the exact accepted convolution and recurrent state is written to
the checkpoint-category frontier.

### cache-reuse

A valid cache allocation is sufficient to keep an indexed prefix node alive, but it is not sufficient to make the node reusable for a request. Reuse requires the block to have been published (`is_valid`) and is revalidated by `Scheduler::PlanResume()` on every pass.

### resume-selection

`resume_len` is selected by `Scheduler::PlanResume()`. Without checkpoint bytes it is the contiguous prefix-valid token end, capped by the async executable upper bound. With checkpoint bytes it is the latest position among the request frontier, published block checkpoints, and fork sources that is covered by valid prefix content; restore intent is expressed as copy plans into the frontier. Generic cache validity alone never raises `resume_len` past content that was not proven produced (`is_valid` for indexed nodes, `filled_len` for private blocks). `resume_len` (what every stateful module skips) is distinct from `readonly_block_num` (the KV-store boundary): full validity of leading whole blocks marks them read-only for KV stores even when checkpoint coarseness keeps `resume_len` lower, so the re-processed window `[resume_len, readonly_block_num * logical_block_size)` rebuilds recurrent state without re-writing already-valid KV (`concepts.cache-geometry`). When a matched indexed block carries a `partial` sibling whose end `ye` lies within the contiguous valid prefix (`ye <= prefix_end`), `PlanResume()` may select that sibling's published checkpoint as `resume_len` (`source=fork`, checkpoint restore only, no KV copy); every sibling-sourced resume reports `source=fork`, and `source=checkpoint` is exclusively a block's own checkpoint. Fork-extension at `B` - when the sibling extends past `prefix_end` or KV must be repopulated - is the highest-precedence resume source: a feasible extension always ends past the valid prefix, so it dominates every frontier and checkpoint candidate. When a published prompt-boundary node exists (`prompt_boundary_node`), a duplicate (or history-extending) prompt may resume at the producer's prompt-boundary node end `B` (the producer's `prompt_boundary_pos = prompt_len - cache_prompt_boundary_skip` on the source node) by restoring the node's KV, plus its recurrent-state checkpoint when the model is recurrent; full-block prompt checkpoints (block end `<= prompt_len`) remain always-on regardless of the knobs, but generation-region full-block checkpoints (block end `> prompt_len`) are suppressed when `cache_generation=none` (generated blocks are never indexed under `none`, so such a checkpoint would only serve the same request's own resume).

### category-registration

Modules register anonymous byte requirements with the prefix or checkpoint category during construction and keep only byte offsets or base part ids (per registration channel). Each category registers one composite `ObjectAllocator` object id after all modules have registered. A category exposes two registration channels: an accumulation channel (grows part 0, returns a within-part byte offset) and a composite channel (appends parts 1..N, returns the base part id). Slab classes in `ObjectAllocator` are deduped by aligned size, and two same-aligned-size simple categories would share an object id (out of scope: prefix is the only simple category). Modules own the content semantics of their registered byte ranges. The `CacheRegistry` only maps categories to object ids and byte offsets; cache block slot reservation, validity, resume selection, and release are scheduler policy. When a module sizes its composite parts to equal another category's aligned object size (e.g. `GatedDeltaNetLayer` block-sizing recurrent parts to the prefix object), they share one slab class and become interchangeable at slot granularity under the already category-agnostic eviction sweep; page-granular reclamation (`slab.h`, `kMaxEmptySlabs == 0`) is the pre-existing baseline that also applies when sizes differ.

### unified-attention

`UnifiedAttentionLayer` registers its KV byte requirement with the prefix category during construction and stores the returned byte offset. Target and draft decoders resolve only their own registered byte offsets while sharing the same logical prefix cache block. During setup each layer resolves committed prefix cache blocks from logical blocks and prepares KV pointer metadata. Reserving logical-block cache slots and validating contiguous prefix coverage is scheduler planning, not module work. Physical KV layout and iteration use `cache_block_seq_len`, while pointer counts and read-only store boundaries use `logical_block_size`. It skips KV cache stores for positions in read-only leading blocks (`< readonly_block_num * logical_block_size`) and supplies those positions from the already-valid blocks during reads (`concepts.cache-geometry`).

`ProcessKV_v2` remains the sole multi-query K/V transformation and store
owner. The standalone verification kernel consumes the same `block::Layout`
paged bytes as decode, applies query bias/RoPE/log-N and a position-specific
causal/window mask, and writes either packed output or executor-owned split
partials. It does not participate in the legacy attention registry.
Verification split indices are local to the verification query partition.

### gated-deltanet

`GatedDeltaNetLayer` does not partition its token recurrence across CP ranks.
It folds attention CP into GDN tensor parallelism: rank-local weights,
convolution state, and recurrent state use
`attn_tp_size * attn_cp_size`, with shards selected by `model_tp_rank`.
Scheduler checkpoint operations publish and restore these rank-local shards
in lockstep at the same global position.

`GatedDeltaNetLayer` registers its recurrent/convolution state byte requirement
with the checkpoint category during construction and stores the relevant
offsets (per-layer convolution element offsets within part 0, computed by the
module, and the base part id `rec_base` for recurrent parts). During setup it
resolves the committed frontier cache part bases for each request, reads
`submitted->input_len` as that row's physical input length, and records which
requests start their forward at position zero
(`submitted->history_len + inflight_input_len == 0`; in-flight tokens advance
the frontier before this batch runs). During `kPrepare` it clears its
registered parts (convolution part 0 and each recurrent block part, including
any rounding padding) for those requests. It does not know whether checkpoints
are restored, published, or shared; those are scheduler-planned, executor-run
whole-object copies.

A speculative target block suppresses the ordinary final convolution and
recurrent-state stores without changing GDN outputs. After target verification
and terminal clamping, the module replays exactly the accepted transition prefix
into its canonical rank-local state and discards the phase journal.

On eligible SM90 verification inputs of at most 16 positions, the speculative
prefix uses the smallest fitting capacity-8 or capacity-16 single-chunk GDR
forward. It reads the entry recurrent state and emits output without exposing
any state-write path. Other rows retain the ordinary recurrent/chunked kernels.
Transition capture still precedes the forward, and only accepted-prefix replay
mutates canonical convolution and recurrent state after target verification.

For eligible SM90 inputs of at most 16 positions, recurrent-state commit uses
the same GDR template with layers and speculative requests folded into one
batch. Setup prepares phase-owned host pointers adjusted to each layer's
state slice; prepare copies these pointers and builds their TMA descriptors
on the executor stream. After terminal clamping, final device `accept_len`
is broadcast across layers and the commit kernel updates recurrent state
without producing GDR output. The forward suppression mask and newly updated
`finished` mask do not apply to commit: a newly terminal row still commits
its accepted prefix, while a previously finished row has zero accepted length.
Convolution commit retains its existing ring update, and unsupported recurrent
commit configurations retain scalar replay. Device commit metadata is transient
executor storage, reserved alongside the journal and released on the same
stream after its consumers are enqueued.

The recurrent state is a rounded-up two-dimensional
`(L_b layers × H_b v_heads)` block grid: one uniform composite part
(`block_bytes_`) per block, with convolution state unchanged.
`GatedDeltaNetLayer` resolves a per-(layer-group, batch, head-group) recurrent
base (composite part
`rec_base + (L/L_b)*ng + (h/H_b)`, shared by all `L_b` layers of the
block-row), plus a per-layer in-block element offset
`linear_state_offset == (L%L_b)*H_b*cell_elems`, and one accumulated
convolution base (part 0) with the per-layer convolution element offset,
instead of one recurrent base per layer. The recurrent kernel indexes
head-groups as
`state_ptrs[b*ng + h/H_b] + linear_state_offset + (h%H_b)*state_size`.
With `TM_GDN_BLOCK_CONFIG` unset
(`L_b=1`, `H_b=num_v_heads`, `ng=1`), this reduces exactly to one base per
layer at offset zero. Consumers that reuse a prompt-boundary checkpoint resume
at `B` with a restored checkpoint, not position zero, so the clear-at-start
path is unaffected.

### checkpoint-publish

Checkpoint publication is planned and committed entirely by the scheduler.
Publication targets the node's own block-owned checkpoint slot, created lazily
at first publication planning (owner attached at `Create`) and reallocated in
place thereafter, which is the same model as the prefix slot; no request-owned
publication slot exists. Commit knows the forward end only after admitted
`submitted->input_len`, and planning skips nodes that already hold a validly
allocated checkpoint. At most one request can plan a given node per pass: a
block target is producer-excluded because the committed forward writes the
token before the block end inside it, while a sibling target is reachable only
by the request whose trie insert created the boundary node through first-wins
arming of `prompt_boundary_node`. A checked per-pass slot reservation enforces
that uniqueness.

Publication planning is routed mutually exclusively by the pass's forward end
between a prompt-boundary group and a full-block group. The prompt-boundary
group plans a partial sibling node's KV copy only when `B` is mid-block, plus
the boundary checkpoint onto either that partial sibling or the block-aligned
boundary block. It runs only when `prompt_boundary_node` is set and the
forward lands at `B`, so a not-yet-reached pass allocates neither the KV block
nor the checkpoint slot. The full-block group is coverage-driven: it publishes
if and only if a full block ends exactly at the forward end, subject to the
configured minimum interval and without knowledge of prompt-boundary mode.
The admission clamp in `contracts.scheduler-commit` guarantees a
block-aligned pass end whenever the minimum interval is due in the prompt
region, and `PlanResume` seeds `last_ckpt_pos` from a restored checkpoint so
spacing is measured from it.

Recurrent checkpoint publication is suppressed while `is_warm_up` is set for
GEMM warm-up; frontier working state is still allocated and updated. The one
exception is `cache_generation=none`, which skips generation-region full
blocks (block end greater than `prompt_len`) while keeping prompt-region
full-block checkpoints. The prompt-boundary checkpoint bypasses the minimum
interval. Terminal adoption in `contracts.checkpoint-adoption` may also
undercut the interval; the adopted checkpoint is demoted to evict-first
priority rather than suppressed. This preserves the current coverage-driven
behavior in which a full-block checkpoint just below a future prompt boundary
is retained. The optional admission phase allocates the target checkpoint
slot, setting the slot pin to retain its owner uniformly with every other
owner-attached allocation. Commit records the publication position and emits
the frontier-to-slot publication copy that the executor runs after `kUnprep`.

### prefix-identity

Prefix identity is token identity, per-image content identity, plus parent identity. Index lookup must use cumulative `PrefixKey`, exact parent identity, exact segment-token comparison, and exact comparison of the block's start-fingerprints (`LogicalBlock::image_fps`). A fingerprint is the image's opaque 256-bit content identity; an empty fingerprint never compares equal to anything, including another empty fingerprint. Blocks interior to an image carry no fingerprint of their own — their identity is carried by the cumulative key and the parent chain, since the image's first block exact-compares the fingerprint. Hash equality alone is never identity.

### prefix-ownership

Producer marking is a per-pass exclusion mechanism. `Scheduler::Schedule()` sets `LogicalBlock::producer` on the committed produced range and the same pass's publication step clears it. A request must not be admitted to write blocks carrying a foreign producer mark. There is no cross-pass ownership state to clean up on cancel.

### prefix-publish

Publication of produced ranges happens at scheduler commit after memory
replay. For ordinary execution, indexed nodes become `is_valid` only when the
committed forward end fully covers them, private blocks become valid with
their content extent tracked by `filled_len`, and device content arrives in
submission order so a consumer executes after the producer batch committed
before it. Speculative target, refresh, and extension writes remain
sequence-private and unindexed while speculative phases are active.
`MarkProduced()` clears producer ownership over the conservative submitted
interval but does not advance `filled_len` or assign prefix identity. Normal
finalization indexes only the exact committed prefix; no delayed prompt
insertion or conservative tail publication exists.

### cancel-release

Canceling or releasing a request drops the request's references; the order among blocks is immaterial. Private (un-indexed) blocks have their allocations deallocated immediately (dropping the allocation ref while the request ref still pins the block), then clearing `Sequence::block_ids` drops the request refs and recycles any now-unreferenced block; indexed nodes keep valid allocations alive (each allocation holds an allocation ref in `LogicalBlock::refs` via the slot's pin) and remain discoverable. Incomplete indexed nodes are left `is_valid == false`, so no consumer can resume from their content; they are reclaimed by eviction.

### checkpoint-adoption

Terminal checkpoint frontier adoption happens inside `Scheduler::Finalize()` for normally finished, non-canceled, trie-eligible requests when the frontier allocation is valid, and is gated by `cache_generation == all`. Adoption does not test `frontier_pos`: at finalization the live recurrent buffer is guaranteed to correspond to `filled_len` because the finishing pass stored its state there and the GDN recurrence kernel bypasses its state write-back whenever the device finished mask is set, so any async over-shoot pass leaves the buffer untouched. `frontier_pos` is resume-fast-path bookkeeping committed speculatively as the scheduled forward end (`CommitResults()`), so under async lookahead it over-counts past `filled_len`; testing it would spuriously block a safe adoption. Adoption transfers the frontier cache slot into the checkpoint slot of the newly indexed terminal block and transfers the allocation's reference to that block (the slot's pin is set to retain that block); when the block already holds a created-but-unallocated checkpoint slot, that slot is transferred the other way — into the finished sequence's frontier field — and is invalidated with its new owner at `Release` (slot invalidation is performed by the owning `CacheBlockPtr` at owner destruction, per `ownership.prefix`). Adoption is unconditional (frontier valid, slot not validly allocated, terminal block indexed). When another valid checkpoint lies within `checkpoint_min_interval` below `filled_len`, the adopted checkpoint is demoted to evict-first priority (timestamp 0) instead of being suppressed: the interval is relaxed at finalization only, and the redundant checkpoint stays evict-first while it remains demoted. The terminal partial generated block is itself indexed into the prefix trie whenever `cache_generation == all`, independent of model type (full generated blocks always index): it carries the partial block's KV for every prefix-cached model, and the frontier-checkpoint adoption above applies only when the frontier slot is valid (recurrent models). So the generation-boundary partial node exists only when fork matching can reach it.

### cache-eviction

Eviction may remove cache objects without module-specific knowledge. After
eviction, a prefix node remains indexed only while its reference count is
positive through requests, fork edges, or remaining valid allocations.
`PlanResume()` revalidates checkpoint and prefix resumability on every pass
from current allocation validity. Published checkpoints are not held in a
request's `involved_blocks`, so they age and are reclaimed before live
working-set blocks. A terminal-adopted slot demoted to timestamp zero sorts
before stamped slots until a later restore or required-use stamp promotes it.
Eviction frees a cache allocation, not its `CacheBlock` slot or
`LogicalBlock`; a request or fork reference may keep the block alive after all
allocations are gone, and the block is recycled only when its last reference
drops.

## non-normative examples

### eagle3

EAGLE3 is one implementation of the speculative seams above, not part of their
contract. For `k` draft tokens, its policy returns `Extent = {k + 1, k - 1}` and
`Bootstrap(prompt_len) = {prompt_len + k - 1, prompt_len + 2 * k}`. Its hidden-state
tap captures residuals at the configured target layer ids. Its draft pass runs one
shifted refresh followed by a serial loop of `k - 1` draft extensions.

## checklist

Before changing TurboMind async execution, scheduler, cache management, or module-level `BatchOp` behavior, verify the change preserves these rules:

### state-owner

Does exactly one module own each state mutation?

### cache-prepare

Do `AdmitPrompt`/`PlanResume`/`PlanContinue` only match or create logical blocks, reserve cache block slots, compute `resume_len`, and emit copy intent, without backing allocation, active admission, or device content mutation? Does `PlanContinue` maintain `involved_blocks` incrementally (appending only tail blocks from `EnsureBlocks`) while `PlanResume` rebuilds it from a full scan each pass?

### scheduler-commit

Does `Scheduler::Schedule()` remain the only active-admission, allocation,
eviction, publication-attachment, and `SubmittedRow` commit point? Does every
consumer use the committed value rather than parallel fields or post-scheduler
rederivation?

### cache-semantics

Are cache object byte ranges interpreted only by the module that registered the byte range?

### cache-validity

Is generic cache validity used only for lifetime, not to raise `resume_len`?

### cache-memory

Are cache-object backing-memory reads and writes limited to executor-thread
`BatchOp` handlers and executor-run, scheduler-planned whole-object copies?
Are KV writes limited to
`[readonly_block_num * logical_block_size, end)`, with physical KV sizing and
iteration still based on `cache_block_seq_len`? Are composite whole-object
copies issued once per part? Are speculative writable
destinations private, keyless, unindexed, and bounded by their committed
`SubmittedRow`?

### delayed-release

Can a finishing or canceled request be excluded from scheduling before its resources are physically released?

### cleanup

Is every request-owned resource released only after
`retiring && inflight == 0`?

### async-progress

Does async state account for submitted but not yet reflected ordinary work
through `inflight_input_len`, `inflight_new_tokens`, and `inflight`? Does host
accounting avoid predicting speculative acceptance and retain exact predecessor
`SubmittedRow` geometry while a phase is outstanding?

### forward-progress

On a scheduling pass that admits nothing (empty active batch) with no in-flight work remaining (`inflight == 0` for every request), does the engine fail the highest-priority eligible request (smallest `unique_id`) with `kOutOfMemory` — rather than resubmitting empty batches indefinitely — so a request too large for the cache always receives a terminal status?

### callbacks

Are external callbacks delivered through gateway signals rather than directly on the engine scheduling path?

### prefix-ownership

If producer marking is touched, is it set only at scheduler commit and cleared by the same pass's publication step?

### module-cache

If a module registers cache bytes, does it register with exactly one category, store only its byte offset, define setup pointer resolution, and keep backing-memory reads and writes in executor-thread `BatchOp` handlers?

### boundary-policy

Are partial-block boundary publishes decided at AdmitPrompt/finalization from `cache_prompt` / `cache_generation` (`CacheMode`), with no runtime veto object? Is `cache_prompt=auto` gated on multimodal overlap of the partial node's range? Is the decision a pure function of cross-rank-identical attributes? Is full-block publication kept mode-free (coverage-driven), except that `cache_generation=none` skips generation-region full-block checkpoints (block end `> prompt_len`)? Is the recurrent-checkpoint spacing knob `cache_checkpoint_interval` (> 0, no block_seq_len fallback)?

### contract-sync

If this document no longer matches the intended behavior, is the contract updated in the same change as the code?
