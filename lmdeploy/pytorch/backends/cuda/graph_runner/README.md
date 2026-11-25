# CUDA Graph Runner Design

This package owns CUDA graph selection, capture, replay, and invalidation for
the PyTorch CUDA backend. It has three sibling execution paths:

```text
CUDAGraphRunner
    |
    +-- supported decode --------> full CUDA graph
    +-- prepared PCG prefill ----> piecewise CUDA graph plan
    +-- everything else ---------> eager model forward
```

The full-graph and piecewise paths share the outer dispatcher and model output
contract. They deliberately do not share one mode-heavy executor: their keys,
buffers, capture lifecycle, and failure rules differ.

## File And Ownership Map

| File                             | Owner and responsibility                                                                                                         |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `runner.py`                      | Three-way dispatch, legacy full-graph cache, optional piecewise manager, and whole-runner reset                                  |
| `full_graph.py`                  | One complete CUDA graph's metadata, buffers, warmup, capture, and replay                                                         |
| `piecewise.py`                   | Generic segmented tracing, ordered plan steps, eager argument binding, and manager-owned bridge storage                          |
| `standard.py`                    | The shared standard-decoder prefill policy: token buckets, graph inputs, request frame, output slicing, and descriptor selection |
| `models/utils/cudagraph.py`      | Model-facing capability mixins and the existing full-graph input/output buffer contract                                          |
| `backends/cuda/step_metadata.py` | Atomic capability discovery and eager-boundary installation for the selected CUDA implementations                                |

Models and backend-agnostic `nn` modules must not import this package. A model
opts into the shared prefill runtime with `PiecewiseCudaGraphMixin`; selected
CUDA operator implementations own any eager boundaries required by their
semantics.

## Outer Dispatch

`CUDAGraphRunner.__call__` chooses exactly one path:

1. Supported decode uses the existing full CUDA graph path.
2. A supported prefill derives a piecewise descriptor. A published plan is
   replayed; a startup dummy request may prepare the plan.
3. An unsupported or unprepared request executes eagerly.

Serving never captures a piecewise plan. If startup warmup is disabled or no
matching plan exists, eager execution is selected before piecewise execution
can mutate KV or state caches.

`CUDAGraphRunner.reset()` is the invalidation owner. It drops every full graph,
every piecewise plan, the shared piecewise bridge storage, and the DeepEP graph
buffer. Address-changing lifecycle operations must pass through this complete
reset instead of repairing captured pointers individually.

## Full CUDA Graph Path

`CUDASingleGraphRunner` owns one graph key and one complete model graph:

```text
allocate model-owned static buffers
    -> fill buffers and redirect StepContext
    -> eager warmup
    -> capture the complete model call
    -> retain output buffers
    -> refill and replay for later matching decode steps
```

The outer runner retains these executors by the legacy decode key and preserves
the existing shared graph-pool policy. `CudaGraphMixin` owns buffer allocation,
per-request buffer filling, context rebinding, and logical output slicing.

The first capture returns the warmup output. This prevents SSM/stateful models
from applying the capture-time state update twice.

Decode `torch.compile` is an optional optimization inside `full_graph.py`. It
is not used to partition piecewise execution and is not a correctness
dependency.

## Piecewise CUDA Graph Path

Piecewise CUDA graph (PCG) traces the ordinary forward directly with CUDA
stream capture. It does not build an FX/Dynamo graph. An eager boundary ends
the current capture, runs the original function eagerly, binds its result, and
starts the next capture:

```text
GraphStep -> EagerStep -> GraphStep -> ...
```

There are intentionally only two step kinds:

- `GraphStep` owns and replays one `torch.cuda.CUDAGraph`.
- `EagerStep` owns one boundary declaration, one replay argument template, and
  one result binding.

### Startup Construction

`PiecewiseGraphManager.prepare()` runs only for startup dummy prefills:

1. `StandardDecoderPiecewiseGraphRuntime.warmup()` runs the bucket-shaped
   forward eagerly to finish lazy kernel initialization.
2. `build()` allocates bucket-shaped static inputs and calls
   `trace_piecewise_cuda_graph()`.
3. `_CaptureBuilder` captures the model until the first eager boundary.
4. It ends and replays that prefix graph so the eager function can consume its
   real output.
5. It runs the original eager function, records its arguments and result
   binding, then resumes capture.
6. It publishes the complete plan only after the final piece succeeds.

Captured prefixes are replayed during construction because ending CUDA capture
records kernels but does not execute them. Construction therefore already
materializes the startup request's output; the manager returns that result and
does not replay the new plan immediately.

### Serving Replay

For a prepared descriptor, replay:

1. fills the plan-owned top-level graph input buffers;
2. binds the live request frame;
3. executes every graph and eager step in recorded order on the engine's main
   forward stream; and
4. returns the logical raw-token view of the reusable output buffers.

Model Python `forward` is not rerun during serving replay.

When DP ranks are in different local phases, the global step is still prefill.
The locally decoding rank reuses the same token-bucket plan, while eager
attention, state, and routed-expert boundaries select their live phase from
request metadata. Speculative attention normalizes its
`[batch, query_len, heads, dim]` result back to the flattened-token shape used
by the captured projection.

Capture and replay both run on the engine's main forward stream
(`ModelAgent.stream`), the same non-default stream the full decode graph uses.
The manager resolves that stream lazily from the current stream at `prepare()`
time, so no dedicated PCG stream is allocated. Because the plan stream is the
caller stream, the cross-stream waits collapse to no-ops on the steady path.
CUDA graph-private pools remain plan-local because different token buckets may
replay in arbitrary request order.

The outer `forward_piecewise_cudagraph` profiler range covers input binding,
the complete ordered plan, and output projection. Nested `piecewise::graph:*`
and `piecewise::eager:*` ranges expose the individual steps.

## Graph Inputs And Request-Frame Inputs

The two categories solve different lifetime problems.

### Graph Inputs

Graph inputs are copied into plan-owned, fixed-address tensors before replay.
The current standard runtime supports:

- `input_ids`, token axis 1;
- `position_ids`, token axis 1; and
- optional `mrope_position_ids`, token axis 1.

Their physical token extent is the selected bucket. Every replay clears the
padded tail before copying the logical request prefix.

### `frame_inputs`

`frame_inputs` is **not** a generic collection of everything that does not
enter a CUDA graph. It is the current request's named lookup table for values
that may appear as direct eager-function arguments. The standard runtime
currently supplies:

- `past_key_values`;
- `attn_metadata`; and
- `state_ids`.

During construction, `_bind_eager_argument()` compares a direct eager argument
by identity with the construction frame. A match becomes `_FrameValueRef(name)`
instead of retaining the dummy request object. Immediately before replaying an
eager step, `_resolve_eager_argument()` replaces that reference with
`frame_inputs[name]` from the live request.

This mapping does not make arbitrary captured use safe. If graph work consumes
one of these values, its tensor address must be stable for the plan lifetime or
it must be represented by a copied graph input and covered by the descriptor.
For example, the cache arena has runner-lifetime storage ownership even though
request-local cache metadata is eager.

Binding is intentionally shallow. Only direct eager-call arguments are
resolved. Nested request objects are unsupported until an operator proves the
ownership of every contained value.

### Other Eager Arguments

An eager argument template may also contain:

- a prior eager-only result slot;
- a stable graph bridge;
- an immutable constant represented by the plan descriptor;
- a runner-owned model/backend object; or
- a non-owning CUDA tensor view whose storage is kept alive by a graph pool,
  static input, model/cache allocation, or bridge.

Published steps must not retain startup request metadata or own captured graph
pool storage accidentally.

## Eager Boundary Output Policies

`@eager_boundary` is transparent outside active piecewise construction. Nested
decorated calls belong to the outermost eager boundary.

A boundary has one output policy:

1. The default `FixedOutputAdapter` copies a fixed tensor pytree into stable
   graph-visible storage.
2. `PaddedTensorOutputAdapter(token_axis=...)` copies a raw-token tensor into a
   bucket-shaped bridge and clears the padded tail.
3. `ViewTolerantPaddedAdapter(token_axis=...)` has the same bucket contract but
   accepts a view result, such as the process-wide DeepEP combine-buffer view,
   and copies it into stable bridge storage.
4. `eager_only_output=True` stores a dynamic Python result in
   `_EagerValueSlot`. Later eager steps can consume the current request's
   result, but a captured graph must not consume it.

A bridged tensor may safely be consumed by both a later eager step and a later
graph piece: both see the stable bridge. A single boundary cannot currently
return a mixed tree in which some leaves are eager-only and other leaves enter
a graph. Split that operation into separate boundaries until a real operator
justifies an operator-owned composite adapter.

Generic adapters reject non-strided tensors, unsupported views, and outputs
that alias boundary inputs. Alias-preserving behavior belongs to an
operator-owned adapter, not the generic tracer.

## Bridge Lifetime And Reuse

Every graph-visible eager output needs a stable address. A normal bridge stays
reserved for the plan lifetime.

An operator may set `reuse_bridge_after_next_step=True` only when that output
cannot be used after the immediately following graph or eager step. The pool
then makes its storage logically available to a later boundary without freeing
the physical allocation. Captured graphs may continue to retain the address.

The same pool can back mutually exclusive token-bucket plans because model
forwards are serialized and every plan remains resident. Concurrent plan
replay would require separate slots or an explicit lease and is unsupported.

## Standard Decoder Descriptor

`_StandardGraphDescriptor` identifies one complete plan with:

- the aggregate-token bucket;
- the graph-visible input names; and
- immutable extra forward arguments.

The exact request count is deliberately absent for the currently supported
Qwen3 and Qwen3.5 partitions. Every request-partition-dependent consumer is
eager, while captured tensor algebra depends only on the flattened token
bucket. This is a proven property of these operator partitions, not a generic
decoder assumption. A future captured batch-shaped operation must either add
the relevant capacity/layout fact to the descriptor or move that operation
eager.

The initial runtime uses a fixed 512-token stride up to the configured
`piecewise_cudagraph_max_tokens` cutoff. All candidate buckets are captured at
startup and retained; there is no runtime capture or eviction. The cutoff is
execution policy, not scheduler or paging capacity.

Current static eligibility requires:

- a model with `PiecewiseCudaGraphMixin`;
- CUDA execution with eager mode disabled;
- PCG support from every selected CUDA implementation.

TP and default-mode DP/EP are supported with one independent plan per rank.
DeepEP routed experts remain eager boundaries. The `DP_TP` layer mode is
rejected before boundary installation because its live per-rank splits and
collectives are not yet represented by the plan. This does not reject the
default attention-TP groups formed by a DP2/EP4 deployment.

Request-time selection rejects microbatch prefill, chunked prefill, and live
LoRA adapters because the current descriptor does not represent those modes.
Dynamic extra forward arguments are rejected; immutable scalar-like extras are
included in the descriptor.

Padding changes GEMM shapes and can change floating-point results. Compare PCG
with the equivalent bucket-shaped eager execution when validating mechanism
correctness, and separately measure the user-visible numerical and performance
effect against raw eager execution. The 512-token stride is provisional rather
than a universal profitability threshold.

## Failure And Reset Rules

- An unsupported or missing serving plan selects eager before PCG begins.
- Warmup, build, and replay failures propagate. Never retry the whole model
  forward after PCG may have written KV or state caches.
- A failed build publishes no partial plan. Active capture cleanup is
  best-effort while preserving the original exception.
- Plans are never evicted during serving.
- `CUDAGraphRunner.reset()` invalidates complete plans before weights, cache
  arenas, or other captured addresses can change.
- Steady replay uses stream ordering, not device-wide synchronization.

## Adding Piecewise Support

Keep the generic runner model- and operator-blind.

1. Decide which work actually depends on live request layout, metadata, state,
   host control flow, or unsupported collectives. Keep the remaining tensor
   algebra captured.
2. If the model follows the standard decoder input/output contract, opt it in
   with `PiecewiseCudaGraphMixin`. Do not add a runtime file per model.
3. Let every selected CUDA implementation report
   `supports_piecewise_cuda_graph()` and install its own wrapper through
   `enable_piecewise_cuda_graph()`. Capability installation is atomic through
   `CudaStepMetaPlan`.
4. Keep semantic metadata explicit through model, layer, and operator calls.
   Do not hide it in the graph runner, attach it to `StepContext` as a generic
   operator cache, or pass a model owner into a backend.
5. Choose one output policy. Use a normal bridge for fixed graph-visible
   tensors, a narrow operator-owned adapter for different semantics, or an
   eager-only slot for values consumed only by later eager steps.
6. Put every fact that changes captured shapes or boundary order in the
   descriptor. Do not key on metadata consumed only by eager boundaries.
7. Ensure startup warms and captures every size that serving may select.
8. Validate equal aggregate-token counts with different request partitions,
   prefix/history cache writes, state updates, reset/rebuild, step order,
   retained memory, and real latency.

Do not add model/operator branches, attention redispatch, synthetic metadata,
scheduler pages, runtime warming, fallback recovery, or signature/slicing DSLs
to `piecewise.py` or `standard.py`.

## Deliberately Unsupported

- Runtime capture and plan eviction.
- Concurrent build or replay on one model runner.
- Mixed eager-only and graph-visible leaves from one boundary result.
- Recursive binding of nested request objects.
- General alias/view-preserving output bridges.
- `DP_TP` layer execution.
- Dynamic extra standard-decoder forward arguments.
- Arbitrary data-dependent changes to boundary order.
- PCG on PyTorch 2.0; the non-owning CUDA view uses a private API available
  from PyTorch 2.1 onward.

These are explicit design boundaries. Add one only with a concrete operator,
clear ownership, focused correctness evidence, and measured performance.

## Review Checklist

When reviewing a graph-runner change, verify:

- Does one component clearly own every graph, pool, stream, bridge, buffer, and
  reset transition?
- Can any published step retain a dummy/startup request object?
- Does every captured tensor address remain valid for the plan lifetime?
- Can input values change boundary order without changing the descriptor?
- Is eager selected before side effects for unsupported/unprepared requests?
- Could an exception cause the complete forward to run twice?
- Is operator-specific slicing, metadata, state, or collective logic kept in
  the owning CUDA implementation?
- Does a new output shape need a bridge adapter, descriptor fact, or both?
- Is bridge reuse lifetime proven at the declaration site?
- Are continuous-batching layouts, cache/state mutation, reset, memory, and
  latency covered by evidence proportional to the change?

Early standalone PCG tests and experiments currently live in
`/home/yaoqian/space/tmp/lmdeploy_test/develop/pcg` while the interfaces are
still stabilizing.
